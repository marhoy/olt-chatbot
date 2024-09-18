from olt_chatbot.chat_model import write_docstores_to_disk
from olt_chatbot.document_parsing import get_docs_from_url


if __name__ == "__main__":
    urls = ["https://olympiatoppen.no/", "https://olt-skala.nif.no/"]
    docs = list ()
    for url in urls:
        docs.append(get_docs_from_url(url, max_depth=1))
                            
    write_docstores_to_disk(docs)
