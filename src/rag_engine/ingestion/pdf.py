import pymupdf

from src.rag_engine.core.document import Document


def extract_pdf(path: str) -> Document:
    pdf = pymupdf.open(path)

    pages = []

    for page_number, page in enumerate(pdf):
        text = page.get_text()

        pages.append({
            "page_number": page_number + 1,
            "text": text,
        })

    document = Document(
        document_id=path,
        source_type="pdf",
        source_locator=path,
        pages=pages,
    )

    return document