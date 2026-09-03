from src.rag_engine.ingestion.pdf import extract_pdf


pdf_path = "data/raw/pdf/your_document.pdf"

document = extract_pdf(pdf_path)

print("Document ID:", document.document_id)
print("Source type:", document.source_type)
print("Source locator:", document.source_locator)

print("Number of pages:", len(document.pages))

for page in document.pages:
    print(f"\n--- PAGE {page['page_number']} ---\n")
    print(page["text"])