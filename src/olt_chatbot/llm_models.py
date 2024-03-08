"""Define the LLM models used by the date_autoreply bot."""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_openai import AzureChatOpenAI, AzureOpenAI, AzureOpenAIEmbeddings

from olt_chatbot import config

EMBEDDING_MODELS: dict[str, Embeddings] = {
    "text-embedding-ada-002": AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
    )
}


LLM_GENERATORS: dict[str, BaseLLM | BaseChatModel] = {
    "gpt-3.5-turbo-instruct": AzureOpenAI(
        azure_deployment="gpt-35-turbo-instruct-0914",
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        max_tokens=512,
        temperature=0.0,
    ),
    "gpt-3.5": AzureChatOpenAI(
        azure_deployment="gpt-35-turbo-1106",
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        max_tokens=2048,
        temperature=0.0,
    ),
    "gpt-4": AzureChatOpenAI(
        azure_deployment="gpt-4-turbo-128k-1106",
        azure_endpoint=config.AZURE_OPENAI_ENDPOINT,
        api_key=config.AZURE_OPENAI_API_KEY,
        api_version=config.AZURE_OPENAI_API_VERSION,
        max_tokens=2048,
        temperature=0.0,
    ),
}
