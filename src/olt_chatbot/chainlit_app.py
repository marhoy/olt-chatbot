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
    # #Your custom logic goes here...
    chain: Runnable[str, dict[str, Any]] = cl.user_session.get("chain")
    chat_history: ChatMessageHistory = cl.user_session.get("chat_history")

    # Create an async stream from the runnable
    async_stream = chain.astream(
        message.content,
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

    # Get the final answer, will be added to the history
    ai_answer = chunk["cited_answer"].get("answer", "")

    # Extract the citations and create elements for the message
    cited_doc_indices: list[str] = chunk["cited_answer"].get("citations", [])

    # Debug info
    logger.debug(f"The retriver found {len(retrieved_docs)} chunks")
    logger.debug(f"The following chunks are cited: {cited_doc_indices}")

    # There could potentially be multiple chunks retrieved from the same document.
    # Create a dict of unique document metadata, where the key is the source field.
    cited_metadata = {}
    for i_str in cited_doc_indices:
        try:
            i = int(i_str)
        except ValueError:
            continue
        if not 0 <= i <= len(retrieved_docs):
            continue
        metadata = retrieved_docs[i].metadata
        if metadata["source"] not in cited_metadata:
            cited_metadata[metadata["source"]] = metadata

    # Add the citations to the message
    if cited_metadata:
        msg.content += "\n\nKilder:\n\n"
        for metadata in cited_metadata.values():
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
    chat_history.add_ai_message(ai_answer)
    cl.user_session.set("chat_history", chat_history)


if __name__ == "__main__":
    # For debugging purposes
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
