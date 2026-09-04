import pymupdf

from src.rag_engine.core.document import Block
from src.rag_engine.preprocessing.cleaning import (
    clean_text,
    should_remove_block,
)

def extract_pdf_text(pdf_path):
    """
    Extract text from every page of a PDF.
    """

    pdf = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(pdf):

        text = page.get_text()

        pages.append(
            {
                "page_number": page_number + 1,
                "text": text,
                "character_count": len(text),
            }
        )

    pdf.close()

    return pages


def extract_pdf_blocks(page, page_number):
    """
    Extract layout-aware blocks from a PDF page.
    """

    blocks = page.get_text("dict")["blocks"]

    extracted_blocks = []

    for block_number, block in enumerate(blocks):

        if "lines" not in block:
            continue

        block_text_parts = []
        spans = []

        for line in block["lines"]:

            for span in line["spans"]:

                block_text_parts.append(span["text"])

                spans.append(
                    {
                        "text": span["text"],
                        "font": span["font"],
                        "size": span["size"],
                        "flags": span["flags"],
                        "bbox": span["bbox"],
                    }
                )

        text = clean_text("".join(block_text_parts))

        if not text:
            continue

        if should_remove_block(text):
            continue

        extracted_blocks.append(
            Block(
                block_number=block_number,
                text=text,
                bbox=tuple(block["bbox"]),
                page_number=page_number,
                spans=spans,
            )
        )

    return extracted_blocks