import chainlit as cl
from chat_model import get_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ChatMessageHistory


@cl.on_chat_start
def start_chat():
    chain = get_chat_model()
    cl.user_session.set("chain",chain)
    cl.user_session.set("chat_history", ChatMessageHistory())

@cl.on_message
async def main(message: cl.Message):
    # Your custom logic goes here...
    chain = cl.user_session.get("chain")
    chat_history: ChatMessageHistory = cl.user_session.get("chat_history")
    response = chain.invoke({"question": message.content, "chat_history": chat_history.messages})
    
    cited_urls = set()
    for doc in response["context"]:
        cited_urls.add(doc.metadata["source"])
    url_element = cl.Text(content="\n".join(list(cited_urls)))

    # Send a response back to the user
    await cl.Message(
        content=response['answer'],
        elements=[url_element]
    ).send()

    chat_history.add_user_message(message.content)
    chat_history.add_ai_message(response['answer'])
    cl.user_session.set("chat_history", chat_history)

@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="OL",
            message="Hvordan gjorde Norge det i OL i Paris 2024?.",
            icon="/logo_light.svg",
            ),

        cl.Starter(
            label="Ernæring",
            message="Hva er Olympiatoppens holdninger til kosttilskudd",
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
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)