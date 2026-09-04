from src.rag_engine.core.document import Document


def deduplicate_pages(document: Document) -> Document:
    seen_hashes = set()
    unique_pages = []

    for page in document.pages:
        page_hash = page.get("content_hash")

        if page_hash is None:
            unique_pages.append(page)
            continue

        if page_hash in seen_hashes:
            continue

        seen_hashes.add(page_hash)
        unique_pages.append(page)

    document.pages = unique_pages

    return document