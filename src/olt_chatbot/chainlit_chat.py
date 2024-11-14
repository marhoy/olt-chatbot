import os
print("Current Working Directory:", os.getcwd())

from typing import Any, cast
from venv import logger
import chainlit as cl
from chat_model import get_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from olt_chatbot.document_parsing import read_pdfs_from_fagstoff_folder
from olt_chatbot.document_parsing import get_docs_from_url
import os


@cl.on_chat_start
def start_chat():
    chain = get_chat_model()
    cl.user_session.set("chain",chain)
    cl.user_session.set("chat_history", ChatMessageHistory())

@cl.on_message
async def main(message: cl.Message):
    #Your custom logic goes here...
    chain = cl.user_session.get("chain")
    chat_history: ChatMessageHistory = cl.user_session.get("chat_history")
    

    # Generate the main response and dynamically reference the relevant PDF
    response = chain.invoke({"question": message.content, "chat_history": chat_history.messages})
    response_text = response["cited_answer"]["answer"]
 
    #Preoare cited URLS
    cited_urls = set()
    for citation in response["cited_answer"]["citations"]:
        cited_urls.add(citation)

    # Combine PDF and URL sources into a single content block
    combined_sources = "\n".join(list(cited_urls))


        # Send response back to the user with combined sources
    await cl.Message(
        content=f"{response_text}\n\nKilder:\n{combined_sources}"
    ).send()

    # Update chat history
    chat_history.add_user_message(message.content)
    #chat_history.add_ai_message(response['answer'])
    chat_history.add_ai_message (response_text)
    cl.user_session.set("chat_history", chat_history)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="OL",
            message="Hvordan gjorde Norge det i OL i Paris 2024?",
            icon="/public/learn.svg",
            ),

        cl.Starter(
            label="Ernæring",
            message="Hva er Olympiatoppens holdninger til kosttilskudd?",
            icon="/public/learn.svg",
            ),
        cl.Starter(
            label="Trening",
            message="Hvordan formtopper jeg inn mot konkurranser?",
            icon="/public/terminal.svg",
            ),
        cl.Starter(
            label="Helse",
            message="Hvordan kan jeg minske risiko for sykdom på reise med laget?",
            icon="/public/write.svg",
            )
        ]


if __name__ == "__main__":

    #For debugging purposes
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)