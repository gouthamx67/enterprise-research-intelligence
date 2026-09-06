from src.rag_engine.chunking.strategies import (
    fixed_size_chunks,
    paragraph_chunks,
    recursive_chunks,
    sentence_chunks,
    token_chunks,
)


TEXT = """
The company launched a new enterprise platform in 2025.
The platform combines analytics, automation, and collaboration.

In 2026, the company changed its product strategy.
It increased investment in AI capabilities and reduced investment
in several legacy products.

The company also changed its target customers.
It is now focusing more heavily on large enterprise accounts.
"""


def print_chunks(title, chunks):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)

    for index, chunk in enumerate(chunks, start=1):
        print(f"\n--- CHUNK {index} ---")
        print(chunk)


def main():
    fixed = fixed_size_chunks(
        TEXT,
        chunk_size=100,
    )

    paragraphs = paragraph_chunks(TEXT)

    sentences = sentence_chunks(TEXT)

    tokens = token_chunks(
        TEXT,
        chunk_size=30,
    )

    recursive = recursive_chunks(
        TEXT,
        chunk_size=100,
    )

    print_chunks(
        "FIXED-SIZE CHUNKS",
        fixed,
    )

    print_chunks(
        "PARAGRAPH CHUNKS",
        paragraphs,
    )

    print_chunks(
        "SENTENCE CHUNKS",
        sentences,
    )

    print_chunks(
        "TOKEN CHUNKS",
        tokens,
    )

    print_chunks(
        "RECURSIVE CHUNKS",
        recursive,
    )


if __name__ == "__main__":
    main()