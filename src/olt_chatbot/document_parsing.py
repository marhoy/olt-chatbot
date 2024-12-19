"""Code for crawling the web and reading PDF documents."""

import re
from collections.abc import Iterator
from io import BytesIO

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders.recursive_url_loader import (
    _metadata_extractor as langchain_metadata_extractor,
)
from langchain_core.documents import Document
from loguru import logger
from PyPDF2 import PdfReader

from olt_chatbot import config
from olt_chatbot.custom_recursiveurlloader import CustomRecursiveUrlLoader


def get_docs_from_url(url: str, max_depth: int = 1000) -> Iterator[Document]:
    """Get documents from a URL."""
    logger.info(f"Loading documents from {url}")
    loader = CustomRecursiveUrlLoader(
        url=url,
        max_depth=max_depth,
        extractor=text_extractor_from_response,
        metadata_extractor=metadata_extractor,
        prevent_outside=True,
    )
    # Don't load all documents into a list: Return an iterator instead.
    return loader.lazy_load()


def read_pdfs_from_fagstoff_folder() -> Iterator[Document]:
    """Read extra documents from PDFs in a folder."""
    for pdf_file in config.EXTRA_DOCUMENTS_DIRECTORY.glob("*.pdf"):
        logger.info(f"Reading local PDF file: {pdf_file.name}")
        pdf_text = extract_pdf_text(pdf_file.read_bytes())
        yield Document(page_content=pdf_text, metadata={"source": pdf_file.name})


def clean_text(text: str) -> str:
    """Clean text by removing multiple newlines and spaces."""
    # Change multiple newlines to double newlines
    text = re.sub(r"[\n\r]{2,}", "\n\n", text)
    # Change multiple spaces to single spaces
    return re.sub(r"([ \t\f])+", r"\1", text)


def text_extractor_from_response(response: requests.models.Response) -> str:
    """Extract text from an string that has been downloaded from a URL."""
    content_type = response.headers.get("Content-Type", "")
    try:
        if content_type.startswith("text/html"):
            logger.info("Extracting text from HTML")
            text = BeautifulSoup(response.text, "html.parser").text.strip()
            return clean_text(text)

        if content_type == "application/pdf":
            logger.info("Extracting text from PDF: " + response.url)
            text = clean_text(extract_pdf_text(response.content))
            if not text:
                raise RuntimeWarning(
                    "Reading PDF resulted in empty string, ignoring file."
                )
            return text

        logger.warning(f"Ignoring text of unknown content type: {content_type}")

    except Exception:
        logger.warning(f"Error extracting text from {response.url}")

    return ""


def metadata_extractor(
    raw_html: str, url: str, response: requests.Response
) -> dict[str, str]:
    """Extract metadata from a string that has been downloaded from a URL."""
    content_type = response.headers.get("Content-Type", "")
    if content_type.startswith("text/html"):
        metadata = langchain_metadata_extractor(raw_html, url, response)
    elif content_type == "application/pdf":
        metadata = {"source": url}
    else:
        logger.info(
            f"Ignoring metadata extraction for unknown content type: {content_type}."
        )
    return metadata


def extract_pdf_text(pdf_data: bytes) -> str:
    """Get the text from a PDF file, represented as a byte-string."""
    reader = PdfReader(BytesIO(pdf_data))
    return "".join(page.extract_text() for page in reader.pages)
