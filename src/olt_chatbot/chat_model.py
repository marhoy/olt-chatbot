"""Code for the chatbot model."""

import pickle
from operator import itemgetter
from typing import Any, Iterator

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from pydantic import BaseModel, Field

from typing import List

from langchain_core.runnables import RunnablePassthrough
from typing_extensions import Annotated, TypedDict

from olt_chatbot import config
from olt_chatbot.llm_models import EMBEDDING_MODELS, LLM_GENERATORS
#from tasks import generate_response

EMBEDDING = EMBEDDING_MODELS["text-embedding-ada-002"]

class CitedAnswer (BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used.""" # noqa:E501

    answer: str = Field(
        ...,
        description=(
            "The answer to the user question, which is based only on the given sources."
        ),
    )
    citations: list[str] = Field(
        ...,
        description="The URLs of the SPECIFIC sources which justify the answer."
    )


def write_docstores_to_disk(docs: list[Iterator[Document]]) -> None:
    """Store a vector db and BM25 retriever to disk."""
    text_splitter = text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    # We can not add documents to the BM25 retriever, so we need to have all chunks
    # # available as a list. Let's hope you have enough memory...

    chunks = text_splitter.split_documents(docs)
    
    logger.info("Creating vector retriever")
    Chroma.from_documents(
        filter_complex_metadata(chunks),
        embedding=EMBEDDING,
        persist_directory=config.CHROMA_DB_PATH,
    )

    logger.info("Creating BM25 retriever")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    with open(config.BM25_RETRIEVER_PATH, "wb") as file:
        pickle.dump(bm25_retriever, file)


def load_retriever_from_disk(k: int = 5) -> BaseRetriever:
    """Load a retriever from disk."""
    # Load Chroma db from disk
    logger.debug("Loading retriever from disk")
    vectordb = Chroma(
        persist_directory=config.CHROMA_DB_PATH, embedding_function=EMBEDDING
    )
    vector_retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})

    # Load BM25 retriever from disk
    with open(config.BM25_RETRIEVER_PATH, "rb") as file:
        bm25_retriever = pickle.load(file)
    # bm25_retriever.k = k  # Dette virker ikke lenger. Vet ikke hvordan man fikser det...

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.6, 0.4]
    )

    return ensemble_retriever


def get_chat_model (        
    model_name: str = "gpt-4o",
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Create a chat model with history."""
    logger.debug(f"Creating chat model with {model_name}")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(f"Source URL: {doc.metadata['source']}\nContent: {doc.page_content}" for doc in docs)

    llm = LLM_GENERATORS[model_name]

    retriever = load_retriever_from_disk()
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant working at Olympiatoppen. "
                    "You answer questions in Norwegian. "
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                (
                    "You are an assistant for question-answering tasks. "
                    "Give a user question and some context, answer the question."
                    "If you don't know the answer, just say that you "
                    "don't know.\n\n"
                    "Context: {context}\n\n"
                ),
            ),
            ("human", "{question}"
)
        ]
    )
    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=itemgetter("context") | RunnableLambda(format_docs)
        )
        | prompt
        | llm.bind_tools([CitedAnswer], tool_choice="CitedAnswer")
        | JsonOutputKeyToolsParser(key_name="CitedAnswer", first_tool_only=True)
    )
    chain = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    ).assign(cited_answer=rag_chain_from_docs)

    return chain
