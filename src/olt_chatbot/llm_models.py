"""Define the LLM models used by the date_autoreply bot."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings


from olt_chatbot import config

EMBEDDING_MODELS: dict[str, Embeddings] = {
    "text-embedding-ada-002": OpenAIEmbeddings(
        model="text-embedding-ada-002",
        api_key=config.OPENAI_API_KEY.get_secret_value()
    )
}

LLM_GENERATORS: dict[str, BaseLLM | BaseChatModel] = {
    "gpt-3.5-turbo-instruct": OpenAI(
        model="gpt-3.5-turbo-instruct-0914",
        api_key=config.OPENAI_API_KEY.get_secret_value(),
        max_tokens=512,
        temperature=0.0,
    ),
    "gpt-3.5": ChatOpenAI(
        model="gpt-3.5-turbo-1106",
        api_key=config.OPENAI_API_KEY.get_secret_value(),
        max_tokens=2048,
        temperature=0.0,
    ),
    "gpt-4": ChatOpenAI(
        model="gpt-4-turbo-128k-1106",
        api_key=config.OPENAI_API_KEY.get_secret_value(),
        max_tokens=2048,
        temperature=0.0,
    ),
    "gpt-4o-mini": ChatOpenAI(
        model="gpt-4o-mini",
        api_key=config.OPENAI_API_KEY.get_secret_value(),
        max_tokens=2048,
        temperature=0.0,
    ),
    "gpt-4o": ChatOpenAI(
        model="gpt-4o",
        api_key=config.OPENAI_API_KEY.get_secret_value(),
        max_tokens=2048,
        temperature=0.0,
    ),
}
