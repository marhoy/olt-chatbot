from io import BytesIO
from typing import Iterator

import fitz
from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_core.documents import Document
from loguru import logger

url = "https://olympiatoppen.no/"


def get_docs_from_url(url: str, max_depth: int = 1) -> Iterator[Document]:
    """Get documents from a URL."""
    logger.info(f"Loading documents from {url}")
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=text_extractor,
        metadata_extractor=metadata_extractor,
        prevent_outside=True,
    )
    # Don't load all documents into a list: Return an iterator instead.
    return loader.lazy_load()


def text_extractor(string: str) -> str:
    """Extract text from an string that has been downloaded from a URL."""
    try:
        if "html" in string[:100]:
            return BeautifulSoup(string, "html.parser").text

        elif string.startswith("%PDF-"):
            logger.debug("Extracting text from PDF")
            with fitz.open("pdf", BytesIO(string.encode("utf-8"))) as file:
                text = "\n\n".join(page.get_textpage().extractText() for page in file)
            return text

        else:
            logger.warning("Ignoring unkown string: " + string[:100])

    except Exception as e:
        logger.warning(f"Error extracting text: {e}")
        logger.debug(f"String: {string[:100]}")

    return ""


def metadata_extractor(raw_html: str, url: str) -> dict[str, str]:
    """Extract metadata from a string that has been downloaded from a URL."""
    metadata = {"source": url}
    # TODO: Also extract title and description, depending on file type.
    return metadata
