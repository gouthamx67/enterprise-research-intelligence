import re
import unicodedata
from collections import Counter

from src.rag_engine.core.document import Page

def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters into a consistent representation.
    """
    return unicodedata.normalize("NFKC", text)


def normalize_whitespace(text: str) -> str:
    """
    Normalize excessive spaces while preserving line boundaries.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_text(text: str) -> str:
    """
    Apply conservative text cleaning.
    """

    text = normalize_unicode(text)
    text = normalize_whitespace(text)

    return text


def looks_like_page_number(text: str) -> bool:
    """
    Detect blocks that consist only of digits.

    This is intentionally simple for now.
    """

    text = text.strip()

    if not text:
        return False

    return text.isdigit()


def should_remove_block(text: str) -> bool:
    """
    Determine whether a block is currently considered
    an extraction artifact.
    """

    return looks_like_page_number(text)


def find_repeated_blocks(pages, minimum_occurrences=2):
    """
    Find block texts that occur repeatedly across pages.

    Detection and removal are intentionally separate.
    """

    block_texts = []

    for page in pages:

        for block in page.blocks:

            text = block.text.strip()

            if text:
                block_texts.append(text)

    counts = Counter(block_texts)

    return {
        text
        for text, count in counts.items()
        if count >= minimum_occurrences
    }


def is_near_top(block, page_height, threshold=0.15):
    """
    Determine whether a block is near the top of a page.
    """

    top = block.bbox[1]

    return top <= page_height * threshold


def is_near_bottom(block, page_height, threshold=0.85):
    """
    Determine whether a block is near the bottom of a page.
    """

    bottom = block.bbox[3]

    return bottom >= page_height * threshold


def classify_repeated_blocks(
    pages,
    page_heights,
    minimum_occurrences=2,
):
    """
    Classify repeated blocks as possible headers or footers.

    Repetition + consistent position is used as evidence.
    """

    if len(pages) != len(page_heights):
        raise ValueError(
            "pages and page_heights must contain the same number of items"
        )

    repeated_blocks = find_repeated_blocks(
        pages,
        minimum_occurrences=minimum_occurrences,
    )

    headers = set()
    footers = set()
    other_repeated = set()

    for text in repeated_blocks:

        positions = []

        for page, page_height in zip(pages, page_heights):

            for block in page.blocks:

                if block.text.strip() != text:
                    continue

                positions.append(
                    (
                        is_near_top(block, page_height),
                        is_near_bottom(block, page_height),
                    )
                )

        if not positions:
            continue

        top_count = sum(
            1
            for near_top, _ in positions
            if near_top
        )

        bottom_count = sum(
            1
            for _, near_bottom in positions
            if near_bottom
        )

        if top_count == len(positions):
            headers.add(text)

        elif bottom_count == len(positions):
            footers.add(text)

        else:
            other_repeated.add(text)

    return {
        "headers": headers,
        "footers": footers,
        "other_repeated": other_repeated,
    }


def remove_classified_headers_footers(
    pages,
    classification,
):
    """
    Remove blocks classified as headers or footers.

    Returns new page objects with the unwanted blocks removed.

    The original page objects are not modified.
    """

    removable = (
        classification["headers"]
        | classification["footers"]
    )

    cleaned_pages = []

    for page in pages:

        cleaned_blocks = [
            block
            for block in page.blocks
            if block.text.strip() not in removable
        ]

        cleaned_page = Page(
            page_number=page.page_number,
            blocks=cleaned_blocks,
        )

        cleaned_pages.append(cleaned_page)

    return cleaned_pages