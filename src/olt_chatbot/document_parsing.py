from io import BytesIO
from typing import Iterator

import fitz
from bs4 import BeautifulSoup
#from olt_chatbot.customrecursiveurlloader import CustomRecursiveUrlLoader
from langchain_community.document_loaders import RecursiveUrlLoader

from langchain_core.documents import Document
from loguru import logger
from PyPDF2 import PdfReader
import os
import logging
import requests
import re
import pymupdf

logger = logging.getLogger(__name__)

def get_docs_from_url(url: str, max_depth: int = 1000) -> Iterator[Document]:
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


def text_extractor_new(response: requests.models.Response) -> str:
    """Extract text from an string that has been downloaded from a URL."""
    string = response.text
    try:
        if "html" in string[:100]:
            logger.info("Extractor: Extracting text from HTML")
            text = BeautifulSoup(string, "html.parser").text.strip()
            # Change multiple newlines to double newlines
            text = re.sub(r"[\n\r]{2,}", "\n\n", text)
            # Change multiple spaces to single spaces
            return re.sub(r"([ \t\f])+", r"\1", text)

        if response.headers["Content-Type"] == "application/pdf":
            logger.info("Extractor: Extracting text from PDF")
            with pymupdf.open("pdf", response.content) as file:
                pdf_content = "\n\n".join(
                    page.get_textpage().extractText() for page in file
                ).strip()
                if not pdf_content:
                    raise RuntimeWarning(
                        "Reading PDF resulted in empty string, ignoring file."
                    )
                return pdf_content

        logger.warning("Ignoring string of unknown type: " + string[:100])
    except Exception as e:
        logger.warning(f"Error extracting text: {e}")
        logger.debug(f"String: {string[:100]}")

    return ""

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


def read_pdfs_from_fagstoff_folder(folder_path='/Users/ingrideythorsdottir/Projects/olt-chatbot/fagstoff'):
    #pdf_texts = []
    pdf_sources = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            file_path = os.path.join(folder_path, file_name)
            pdf_text = read_pdf(file_path)
            pdf_sources.append((pdf_text, file_name))
    logger.info(f"pdf_sources: {pdf_sources}")  
    #return pdf_texts
    return pdf_sources

def read_pdf(file_path):
    with open(file_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    
