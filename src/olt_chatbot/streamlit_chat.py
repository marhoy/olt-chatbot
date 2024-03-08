"""Code for the Streamlit app."""

from typing import Any, Generator, Iterator

import streamlit as st
from langchain.memory import ChatMessageHistory
from langchain_core.messages import AIMessage, AIMessageChunk
from loguru import logger

from olt_chatbot.chat_model import get_chat_model
from olt_chatbot.llm_models import LLM_GENERATORS

if __name__ == "__main__":
    logger.info("Starting / rerunning streamlit app")
    st.set_page_config(page_title="Chat with an LLM", page_icon="🤖")
    st.title("Chat with an LLM")

    # Initialize some state variables
    if "llm_name" not in st.session_state:
        st.session_state.llm_name = list(LLM_GENERATORS.keys())[1]
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = ChatMessageHistory()
    if "chat_model" not in st.session_state:
        st.session_state["chat_model"] = get_chat_model(st.session_state.llm_name)

    with st.sidebar:
        model_name = st.selectbox("Which LLM to use?", LLM_GENERATORS.keys(), index=1)
        if (model_name is not None) and (model_name != st.session_state.llm_name):
            logger.debug("Triggered model change by dropdown.")
            st.session_state.llm_name = model_name
            st.session_state.chat_model = get_chat_model(model_name)

        if st.button("Forget last question"):
            st.session_state.chat_history.messages = (
                st.session_state.chat_history.messages[:-2]
            )
        if st.button("Reset chat history", type="primary"):
            st.session_state.chat_history = ChatMessageHistory()

    # Display chat messages from history on app rerun
    for message in st.session_state.chat_history.messages:
        if isinstance(message, (AIMessage, AIMessageChunk)):
            msg_type = "ai"
            avatar = "🤖"
        else:
            msg_type = "user"
            avatar = "🙋🏼‍♂️"

        with st.chat_message(msg_type, avatar=avatar):
            st.markdown(message.content)
            if sources := message.additional_kwargs.get("sources", None):
                st.write(f"Referanser: {", ".join([source for source in sources])}.")

    if user_input := st.chat_input("Type something..."):
        with st.chat_message("user", avatar="🙋🏼‍♂️"):
            st.markdown(user_input)

        with st.chat_message("assistant", avatar="🤖"):
            # Create a new stream from the chat model
            stream = st.session_state.chat_model.stream(
                {
                    "question": user_input,
                    "chat_history": st.session_state.chat_history.messages,
                }
            )

            # First get the question, chat_history and context
            data: dict[str, Any] = dict()
            data |= next(stream)
            data |= next(stream)
            data |= next(stream)

            def get_answer_from_stream(
                stream: Iterator[dict[str, str]]
            ) -> Generator[str, None, None]:
                for item in stream:
                    yield item["answer"]

            # Stream the answer, write end result to the data dict
            data["answer"] = st.write_stream(get_answer_from_stream(stream))

            # Print the sources
            sources = set(chunk.metadata["source"] for chunk in data["context"])
            st.write(f"Referanser: {", ".join([source for source in sources])}.")

            # Add items to the history
            st.session_state.chat_history.add_user_message(data["question"])
            st.session_state.chat_history.add_ai_message(
                AIMessage(
                    content=data["answer"],
                    additional_kwargs={"sources": sources},
                )
            )
