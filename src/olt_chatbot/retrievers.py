"""Code for the document retriever."""

import itertools
import uuid
from collections.abc import Iterator

# from langchain.retrievers.ensemble import EnsembleRetriever
from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from olt_chatbot import config
from olt_chatbot.document_parsing import (
    get_docs_from_url,
    read_pdfs_from_fagstoff_folder,
)
from olt_chatbot.llm_models import EMBEDDING_MODELS

EMBEDDING = EMBEDDING_MODELS["text-embedding-3-large"]


def update_retriever_databases() -> None:
    """Update the retriever databases."""
    all_docs = itertools.chain(
        get_docs_from_url("https://olympiatoppen.no/", max_depth=100),
        get_docs_from_url("https://olt-skala.nif.no/", max_depth=100),
        # get_docs_from_url("https://www.summit2028.no/", max_depth=100),
        # get_docs_from_url("https://www.teamnor.no/", max_depth=100),
        read_pdfs_from_fagstoff_folder(),
    )
    populate_and_write_retriever_to_disk(all_docs)


def populate_and_write_retriever_to_disk(docs: Iterator[Document]) -> None:
    """Store a vector db to disk."""
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )

    # We need to add the documents to the Chroma db in chunks. First create an empty
    # database collection:
    vector_store = Chroma(
        embedding_function=EMBEDDING,
        persist_directory=config.CHROMA_DB_PATH,
    )

    # Then loop over the documents in batches, create chunks, and add each chunk to the
    # database. The maximum number of chunks is 5461.
    for i, doc_batch in enumerate(itertools.batched(docs, 25), start=1):
        chunks = filter_complex_metadata(text_splitter.split_documents(doc_batch))
        for chunk_batch in itertools.batched(chunks, 5000):
            # Add a unique identifier to each chunk
            for chunk in chunk_batch:
                chunk.metadata["chunk_id"] = str(uuid.uuid4())
            logger.info(
                f"Chroma: Processing {len(chunk_batch)} chunks from batch {i} with "
                f"{len(doc_batch)} documents and {len(chunks)} chunks."
            )
            vector_store.add_documents(list(chunk_batch))


def load_retriever_from_disk(k: int = 15) -> BaseRetriever:
    """Load retriever(s) from disk."""
    # Load Chroma db from disk
    logger.debug("Loading retriever from disk")
    vectordb = Chroma(
        persist_directory=config.CHROMA_DB_PATH, embedding_function=EMBEDDING
    )
    return vectordb.as_retriever(search_kwargs={"k": k})
