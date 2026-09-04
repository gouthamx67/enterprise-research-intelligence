import re
from collections import Counter


def get_block_font_size(block):
    """
    Get the largest font size used inside a block.
    """

    sizes = [
        span["size"]
        for span in block.spans
        if span.get("size") is not None
    ]

    if not sizes:
        return None

    return max(sizes)


def get_dominant_font_size(blocks):
    """
    Find the most frequently occurring font size
    across a collection of blocks.
    """

    sizes = []

    for block in blocks:

        for span in block.spans:

            size = span.get("size")

            if size is not None:
                sizes.append(round(size, 1))

    if not sizes:
        return None

    counts = Counter(sizes)

    return counts.most_common(1)[0][0]


def looks_like_numbered_heading(text):
    """
    Detect common numbered heading patterns.
    """

    patterns = [
        r"^\d+\.\s+.+",
        r"^Item\s+\d+[A-Z]?\.",
        r"^\d+\.\d+\s+.+",
        r"^[IVX]+\.\s+.+",
    ]

    return any(
        re.match(pattern, text, re.IGNORECASE)
        for pattern in patterns
    )


def looks_like_heading(block, dominant_font_size):
    """
    Determine whether a block is a heading candidate.
    """

    text = block.text.strip()

    if not text:
        return False

    if looks_like_numbered_heading(text):
        return True

    block_font_size = get_block_font_size(block)

    if block_font_size is None:
        return False

    if dominant_font_size is None:
        return False

    if len(text) > 150:
        return False

    if block_font_size > dominant_font_size:
        return True

    return False