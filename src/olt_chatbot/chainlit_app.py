"""The logic for the Chainlit chatbot."""

from typing import Any

import chainlit as cl
from chainlit.user import User
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_core.runnables import Runnable, RunnableConfig
from loguru import logger

from olt_chatbot.chat_model import get_cited_rag_chain_for_streaming


@cl.on_chat_start
async def on_chat_start() -> None:
    """Runs when a new chat session is created."""
    logger.debug("Starting new chat session")
    chain = get_cited_rag_chain_for_streaming()
    cl.user_session.set("chain", chain)
    cl.user_session.set("chat_history", ChatMessageHistory())
    cl.user_session.set("chunks", [])


@cl.set_starters
async def set_starters(_user: User | None = None) -> list[cl.Starter]:
    """Set the starters for the chat."""
    return [
        cl.Starter(
            # label="OL",
            label="Talentutvikling",
            # message="Hvordan gjorde Norge det i OL i Paris 2024?",
            message="Kan du fortelle meg om OLT sin utviklingsfilosofi?",
            icon="/public/idea_blue.svg",
        ),
        cl.Starter(
            # label="Ernæring",
            label="Ernæring",
            message="Hva er Olympiatoppens holdninger til kosttilskudd?",
            # message="Kan du forklare spesifisitetsprinsippet?",
            icon="/public/idea_red.svg",
        ),
        cl.Starter(
            label="Trening",
            # label="Treningsplaner",
            message="Hvordan formtopper jeg inn mot konkurranser?",
            # message=("Hvordan kan jeg som trener lage en best mulig treningsplan for mine utøvere?",  # noqa: E501
            icon="/public/idea_green.svg",
        ),
        cl.Starter(
            label="Helse",
            # label="Mental trening",
            message="Hordan kan jeg redusere sjanse for smitte på reise?",
            icon="/public/idea_yellow.svg",
        ),
    ]


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Handle the user input and generate a response."""
    # Get objects from the user session
    chain: Runnable[dict[str, Any], dict[str, Any]] = cl.user_session.get("chain")
    chat_history: ChatMessageHistory = cl.user_session.get("chat_history")
    old_chunks = cl.user_session.get("chunks")

    # Create an async stream from the runnable
    async_stream = chain.astream(
        {
            "question": message.content,
            "chat_history": chat_history.messages,
            "old_docs": old_chunks,
        },
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    )

    # Create an empty message and send it to the UI
    msg = cl.Message(content="")
    await msg.send()

    # Loop through the async stream and update the message
    async for chunk in async_stream:
        if "docs" in chunk:
            retrieved_docs: list[Document] = chunk["docs"]
        elif "cited_answer" in chunk:
            msg.content = chunk["cited_answer"].get("answer", "")
            await msg.update()

    # Extract the cited uuids
    chunk_uuids: list[str] = chunk["cited_answer"].get("citations", [])

    # Debug info
    logger.debug(f"The context was based on {len(retrieved_docs)} chunks.")
    logger.debug(f"The following chunks are cited: {chunk_uuids}")

    # There could potentially be multiple chunks retrieved from the same document.
    # Create a dict of unique document metadata, where the key is the source field.
    cited_documents: dict[str, dict[str, Any]] = {}
    for uuid in chunk_uuids:
        for doc in retrieved_docs:
            if (
                doc.metadata.get("chunk_id") == uuid
                and doc.metadata.get("source") not in cited_documents
            ):
                cited_documents[doc.metadata["source"]] = doc.metadata

    # Add the citations to the message
    if cited_documents:
        msg.content += "\n\nKilder:\n\n"
        for metadata in cited_documents.values():
            if "title" in metadata and metadata["source"].startswith("http"):
                # HTML pages will be shown as links with a title
                markdown_string = f"[{metadata['title']}]({metadata['source']})"
            else:
                # PDFs will be shown as plain text without a link
                markdown_string = metadata["source"]

            # Create a numbered list of citations
            msg.content += f"1. {markdown_string}\n"
        await msg.update()

    # Update the chat history
    chat_history.add_user_message(message.content)
    chat_history.add_ai_message(chunk["cited_answer"].get("answer", ""))

    # Save things to the user session to be used for follow-up questions
    cl.user_session.set("chat_history", chat_history)
    cl.user_session.set("chunks", retrieved_docs)


if __name__ == "__main__":
    # For debugging purposes
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
