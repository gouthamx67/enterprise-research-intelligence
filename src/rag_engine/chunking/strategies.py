import re

import tiktoken


def fixed_size_chunks(text: str, chunk_size: int = 500):
    """
    Split text into fixed-size character chunks.

    chunk_size is measured in characters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    text = text.strip()

    return [
        text[start:start + chunk_size]
        for start in range(0, len(text), chunk_size)
        if text[start:start + chunk_size].strip()
    ]


def paragraph_chunks(text: str):
    """
    Split text according to paragraph boundaries.
    """

    text = text.strip()

    if not text:
        return []

    paragraphs = re.split(r"\n\s*\n", text)

    return [
        paragraph.strip()
        for paragraph in paragraphs
        if paragraph.strip()
    ]


def sentence_chunks(text: str):
    """
    Split text into sentence-level chunks.

    This is intentionally conservative and uses
    punctuation-based sentence boundaries.
    """

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def token_chunks(
    text: str,
    chunk_size: int = 100,
    encoding_name: str = "cl100k_base",
):
    """
    Split text into chunks containing at most
    chunk_size tokens.

    The tokenizer is explicitly specified so that
    token boundaries are determined by a real
    language-model tokenizer rather than character
    or whitespace approximations.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    text = text.strip()

    if not text:
        return []

    encoding = tiktoken.get_encoding(encoding_name)

    tokens = encoding.encode(text)

    chunks = []

    for start in range(0, len(tokens), chunk_size):
        token_slice = tokens[start:start + chunk_size]
        chunk = encoding.decode(token_slice).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def recursive_chunks(
    text: str,
    chunk_size: int = 500,
    separators=None,
):
    """
    Recursively split text while attempting to preserve
    natural textual structure.

    The default hierarchy is:

        paragraph
        newline
        sentence-like boundary
        word boundary
        character boundary

    chunk_size is measured in characters.

    This is a structural chunking strategy, not a
    token-based strategy.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    text = text.strip()

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if separators is None:
        separators = [
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    def split_recursively(current_text, separator_index):
        current_text = current_text.strip()

        if not current_text:
            return []

        if len(current_text) <= chunk_size:
            return [current_text]

        if separator_index >= len(separators):
            return [
                current_text[start:start + chunk_size].strip()
                for start in range(
                    0,
                    len(current_text),
                    chunk_size,
                )
                if current_text[start:start + chunk_size].strip()
            ]

        separator = separators[separator_index]

        if separator == "":
            return [
                current_text[start:start + chunk_size].strip()
                for start in range(
                    0,
                    len(current_text),
                    chunk_size,
                )
                if current_text[start:start + chunk_size].strip()
            ]

        pieces = current_text.split(separator)

        if len(pieces) == 1:
            return split_recursively(
                current_text,
                separator_index + 1,
            )

        chunks = []
        current_chunk = ""

        for piece in pieces:
            piece = piece.strip()

            if not piece:
                continue

            candidate = (
                piece
                if not current_chunk
                else current_chunk + separator + piece
            )

            if len(candidate) <= chunk_size:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(current_chunk.strip())

            if len(piece) <= chunk_size:
                current_chunk = piece
            else:
                nested_chunks = split_recursively(
                    piece,
                    separator_index + 1,
                )
                chunks.extend(nested_chunks)
                current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    return split_recursively(text, 0)