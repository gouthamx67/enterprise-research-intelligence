from src.rag_engine.core.document import Document
from src.rag_engine.preprocessing.hashing import content_hash
from src.rag_engine.preprocessing.deduplication import deduplicate_pages


document = Document(
    document_id="test-document",
    source_type="test",
    source_locator="test",
    pages=[
        {
            "page_number": 1,
            "text": "This is page one.",
        },
        {
            "page_number": 2,
            "text": "This is page two.",
        },
        {
            "page_number": 3,
            "text": "This is page one.",
        },
    ],
)


for page in document.pages:
    page["content_hash"] = content_hash(page["text"])


print("Before:", len(document.pages))

document = deduplicate_pages(document)

print("After:", len(document.pages))

for page in document.pages:
    print(page["page_number"], page["text"])