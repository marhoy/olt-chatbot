"""Code for the LangChain Question-Answer chain."""

from operator import itemgetter
from typing import Any

from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
)
from loguru import logger
from pydantic import BaseModel, Field

from olt_chatbot.llm_models import LLM_GENERATORS
from olt_chatbot.retrievers import load_retriever_from_disk


# This is the response object from the model. It Forces the model to cite the sources
# used.
class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""  # noqa: E501

    answer: str = Field(
        ...,
        description=(
            "The answer to the user question, which is based only on the given sources."
        ),
    )
    citations: list[str] = Field(
        ...,
        description="The UUID of the <ARTICLE> which justify the answer.",
    )


def get_chain_with_history(
    model_name: str = "gpt-4o",
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Create a chat model with history."""
    logger.debug(f"Creating chat model with {model_name}")

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(
            f"Reference: {doc.metadata['source']}\nContent: {doc.page_content}"
            for doc in docs
        )

    llm = LLM_GENERATORS[model_name]
    retriever = load_retriever_from_disk()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful AI assistant working at Olympiatoppen. "
                    "You answer questions in Norwegian. "
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "human",
                (
                    "You are an assistant for question-answering tasks. "
                    "Given a user question and some context, answer the question."
                    "If you don't know the answer, just say that you "
                    "don't know.\n\n"
                    "Context: {context}\n\n"
                ),
            ),
            ("human", "{question}"),
        ]
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(
            context=itemgetter("context") | RunnableLambda(format_docs)
        )
        | prompt
        | llm.with_structured_output(CitedAnswer)
    )
    return RunnablePassthrough.assign(
        context=itemgetter("question") | retriever
    ).assign(cited_answer=rag_chain_from_docs)


def format_docs_with_id(docs: list[Document]) -> str:
    """Format the retrieved documents with an enumerated ID."""
    formatted = [
        (
            f'<ARTICLE UUID="{doc.metadata["chunk_id"]}">\n'
            f"\n{doc.page_content}"
            "\n\n</ARTICLE>"
        )
        for doc in docs
    ]
    return "\n\n".join(formatted)


def combine_docs(data: dict[str, Any]) -> list[Document]:
    """Combine new and old docs, ignoring duplicates."""
    old_docs = data.get("old_docs", [])
    new_docs = data.get("new_docs", [])
    combined_docs = {}
    for doc in old_docs + new_docs:
        chunk_id = doc.metadata.get("chunk_id", "")
        if chunk_id not in combined_docs:
            combined_docs[chunk_id] = doc

    return list(combined_docs.values())


def get_cited_rag_chain_for_streaming(
    llm_name: str = "gpt-4o-mini",
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Get the QA chain for Cited Answers."""
    llm = LLM_GENERATORS[llm_name]
    llm_with_tool = llm.bind_tools(
        [CitedAnswer],
        tool_choice="CitedAnswer",
    )

    retriever = load_retriever_from_disk(k=20)

    output_parser = JsonOutputKeyToolsParser(
        key_name="CitedAnswer", first_tool_only=True
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You're a helpful AI assistant, working at Olympiatoppen (OLT). "
                    "You answer questions in Norwegian. "
                    "Given a user question and some article snippets, answer the "
                    "question. If none of the articles answer the question, just "
                    "say you don't know.\n\n"
                    "Here are the articles:\n\n{context}"
                ),
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    return (
        RunnablePassthrough.assign(new_docs=itemgetter("question") | retriever)
        .assign(docs=RunnableLambda(combine_docs))
        .assign(context=itemgetter("docs") | RunnableLambda(format_docs_with_id))
        .assign(cited_answer=prompt | llm_with_tool | output_parser)
    )
