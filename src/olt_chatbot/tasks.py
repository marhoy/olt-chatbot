
from olt_chatbot.chat_model import write_docstores_to_disk
from olt_chatbot.document_parsing import get_docs_from_url
import os
from document_parsing import read_pdfs_from_fagstoff_folder
import logging
from langchain_core.documents import Document


logger = logging.getLogger(__name__)

def generate_response(text):
    # Simple example, you might replace this with actual response generation logic
        return f"Chatbot response based on: {text}"


if __name__ == "__main__":
    # Process URL documents
    urls = ["https://olympiatoppen.no/", "https://olt-skala.nif.no/"]
    url_docs =[] # Collect website documents here
    for url in urls:
        url_docs.extend(list(get_docs_from_url(url, max_depth=100)))


    # Process PDF documents
    pdf_folder = 'fagstoff'
    pdf_sources = read_pdfs_from_fagstoff_folder(pdf_folder)
    pdf_docs = [Document(page_content=text, metadata={"source": file_name }) for text, file_name in pdf_sources]
    
    docs = url_docs + pdf_docs
    write_docstores_to_disk(docs)

