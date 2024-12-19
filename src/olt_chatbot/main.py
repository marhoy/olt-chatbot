"""Main entrypoints for the chatbot."""

import os
import shutil
from pathlib import Path

from chainlit.cli import run_chainlit
from chainlit.config import config as chainlit_config
from loguru import logger

from olt_chatbot import config
from olt_chatbot.retrievers import update_retriever_databases


def update_retrievers() -> None:
    """Update the retrievers."""
    # Delete the old retriever databases
    logger.info("Deleting old retriever databases")
    if config.OUTPUT_DIRECTORY.exists():
        shutil.rmtree(config.OUTPUT_DIRECTORY)

    # Create new retrievers
    update_retriever_databases()
    logger.success("Updating retrievers done!")


def start_chainlit_app() -> None:
    """Start the chainlit app."""
    # This is the path to the Chainlit app
    app_file = Path(__file__).parent.absolute() / "chainlit_app.py"

    # Use port 8888 unless specified otherwise
    os.environ["CHAINLIT_PORT"] = os.environ.get("CHAINLIT_PORT", "8888")

    # By default, we start in headless watch mode
    chainlit_config.run.watch = True
    chainlit_config.run.headless = True

    run_chainlit(str(app_file))
