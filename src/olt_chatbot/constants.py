"""Package wide constants.

Note: We're using Pydantic 2 for settings management. But since Pydantic 2 is not
supported by LangChain yet, we're using Pydantic 1 for data classes.
"""

import os
from pathlib import Path
from typing import cast

from dotenv import find_dotenv
from pydantic.v1 import BaseSettings, SecretStr

# Configuration values are read from (most important first):
# - Environment variables
# - Values defined in the .env-file $CONFIG_FILE (or "config.env" if unset)
# - Default values defined in the Settings class of this file
CONFIG_FILE = os.environ.get("CONFIG_FILE", find_dotenv("config.env", usecwd=True))


class Settings(BaseSettings):
    """Package wide configuration settings."""

    # Credentials for Azure OpenAI API
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_KEY: SecretStr = cast(SecretStr, "")
    AZURE_OPENAI_API_VERSION: str = "2023-09-01-preview"

    OUTPUT_DIRECTORY: Path = Path(__file__).parents[2].absolute() / "output"

    # Chroma DB
    @property
    def CHROMA_DB_PATH(self) -> str:
        return str(self.OUTPUT_DIRECTORY / "chroma-db")

    # BM25 Retriever
    @property
    def BM25_RETRIEVER_PATH(self) -> str:
        return str(self.OUTPUT_DIRECTORY / "bm25-retriever.pkl")

    class Config:
        """Class configuration."""

        env_file = CONFIG_FILE


settings = Settings()
