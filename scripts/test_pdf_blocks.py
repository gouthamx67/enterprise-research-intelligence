import pymupdf

from src.rag_engine.ingestion.pdf import extract_pdf_blocks
from src.rag_engine.preprocessing.structure import (
    get_dominant_font_size,
    get_block_font_size,
    looks_like_heading,
)


PDF_PATH = "data/raw/pdf/your_document.pdf"


pdf = pymupdf.open(PDF_PATH)

print("PDF opened successfully.")
print("Number of pages:", len(pdf))


for page_number, page in enumerate(pdf, start=1):

    blocks = extract_pdf_blocks(
        page,
        page_number,
    )

    dominant_size = get_dominant_font_size(blocks)

    print(f"\n===== PAGE {page_number} =====")
    print("Number of blocks:", len(blocks))
    print("Dominant font size:", dominant_size)

    for block in blocks:

        heading_candidate = looks_like_heading(
            block,
            dominant_size,
        )

        block.is_heading = heading_candidate

        font_size = get_block_font_size(block)

        print(
            f"\n--- BLOCK {block.block_number} ---"
        )

        print("Text:", block.text)
        print("Page:", block.page_number)
        print("Font size:", font_size)
        print("Heading candidate:", block.is_heading)
        print("BBox:", block.bbox)


pdf.close()

print("\nBlock extraction test completed.")