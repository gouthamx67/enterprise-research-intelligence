from src.rag_engine.ingestion.pdf import extract_pdf_text


PDF_PATH = "data/raw/pdf/your_document.pdf"


pages = extract_pdf_text(PDF_PATH)

print("Number of pages:", len(pages))

for page in pages:

    print(
        f"\n--- PAGE {page['page_number']} ---\n"
    )

    print(page["text"])

print("\n--- PAGE CHARACTER COUNTS ---")

for page in pages:

    print(
        f"Page {page['page_number']}: "
        f"{page['character_count']} characters"
    )