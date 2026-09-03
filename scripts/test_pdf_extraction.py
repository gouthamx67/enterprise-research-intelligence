from src.rag_engine.ingestion.pdf import extract_pdf


pdf_path = "data/raw/pdf/your_document.pdf"

pages = extract_pdf(pdf_path)

print("Number of pages:", len(pages))

for page in pages:
    print(f"\n--- PAGE {page['page_number']} ---\n")
    print(page["text"])


print("\n--- PAGE CHARACTER COUNTS ---")

for page in pages:
    print(
        f"Page {page['page_number']}: "
        f"{len(page['text'])} characters"
    )


import pymupdf

pdf = pymupdf.open(pdf_path)

page = pdf[0]

blocks = page.get_text("blocks")

print("\n--- BLOCKS ON PAGE 1 ---")

for block in blocks:
    print(block)


    print("\n--- DICT STRUCTURE FOR PAGE 1 ---")

page_dict = page.get_text("dict")

for block in page_dict["blocks"]:
    print("\nBLOCK")

    if "lines" not in block:
        print("No text lines in this block.")
        continue

    for line in block["lines"]:
        print("  LINE")

        for span in line["spans"]:
            print("    SPAN:")
            print("      text:", repr(span["text"]))
            print("      font:", span["font"])
            print("      size:", span["size"])
            print("      flags:", span["flags"])
            print("      bbox:", span["bbox"])