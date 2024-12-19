"""Define the LLM models used by the date_autoreply bot."""

import os

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from olt_chatbot import config

# Set common environment variables for the Azure OpenAI API
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY.get_secret_value()

EMBEDDING_MODELS: dict[str, Embeddings] = {
    "text-embedding-ada-002": OpenAIEmbeddings(
        model="text-embedding-ada-002",
    ),
    "text-embedding-3-large": OpenAIEmbeddings(
        model="text-embedding-3-large",
    ),
}

LLM_GENERATORS: dict[str, ChatOpenAI] = {
    "gpt-3.5": ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        max_tokens=2048,
        temperature=0.0,
        seed=0,
    ),
    "gpt-4o-mini": ChatOpenAI(
        model="gpt-4o-mini",
        max_tokens=2048,
        temperature=0.0,
        seed=0,
    ),
    "gpt-4o": ChatOpenAI(
        model="gpt-4o",
        max_tokens=2048,
        temperature=0.0,
        seed=0,
    ),
}
