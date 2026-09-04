from src.rag_engine.core.document import Document
from src.rag_engine.preprocessing.cleaning import clean_text
from src.rag_engine.preprocessing.hashing import content_hash


def clean_document(document: Document) -> Document:
    for page in document.pages:
        page["text"] = clean_text(page["text"])
        page["content_hash"] = content_hash(page["text"])

    return document