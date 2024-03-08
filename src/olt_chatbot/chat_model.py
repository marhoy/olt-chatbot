"""Code for the chatbot model."""

from operator import itemgetter
from typing import Any

from bs4 import BeautifulSoup
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers.bm25 import BM25Retriever
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from loguru import logger

from olt_chatbot.llm_models import EMBEDDING_MODELS, LLM_GENERATORS

url = "https://olympiatoppen.no/"


def get_docs_from_url(url: str, max_depth: int = 1) -> list[Document]:
    """Get documents from a URL."""
    logger.info(f"Loading documents from {url}")
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=lambda x: BeautifulSoup(x, "html.parser").text,
        prevent_outside=True,
    )
    data = loader.load()
    return data


def create_retriever(docs: list[Document]) -> BaseRetriever:
    """Create a retriever from documents."""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=300
    )
    chunks = text_splitter.split_documents(docs)

    logger.info("Creating vector retriever")
    embedding = EMBEDDING_MODELS["text-embedding-ada-002"]
    vector_store = Chroma.from_documents(filter_complex_metadata(chunks), embedding)
    vector_retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 5}
    )

    logger.info("Creating BM25 retriever")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
    )

    return ensemble_retriever


def get_llm_chain(
    llm_name: str,
    url: str,
    max_depth: int = 3,
) -> Chain:
    """Create a chain with a retriever from a URL."""
    docs = get_docs_from_url(url, max_depth)
    retriever = create_retriever(docs)
    llm = LLM_GENERATORS[llm_name]
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm,
        retriever=retriever,  # return_source_documents=True
    )
    logger.info("Chain created")
    return chain


docs = get_docs_from_url("https://olympiatoppen.no/", max_depth=1)
retriever = create_retriever(docs)


def get_chat_model(
    model_name: str = "gpt-3.5",
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Create a chat model with history."""

    logger.debug(f"Creating chat model with {model_name}")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    llm = LLM_GENERATORS[model_name]

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
                    "Use the following pieces of retrieved context to answer the "
                    "question. If you don't know the answer, just say that you "
                    "don't know.\n\n"
                    "Question: {question}\n\n"
                    "Context: {context}\n\n"
                    "Answer:"
                ),
            ),
        ]
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=itemgetter("context") | RunnableLambda(format_docs)
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    chain = RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    ).assign(answer=rag_chain_from_docs)

    return chain
