from statistics import mean

from src.rag_engine.chunking.strategies import (
    fixed_size_chunks,
    paragraph_chunks,
    sentence_chunks,
    token_chunks,
    recursive_chunks,
)

from src.rag_engine.chunking.sliding_window import (
    sliding_window_chunks,
)

from src.rag_engine.chunking.parent_child import (
    parent_child_chunks,
)


TEXT = """
The company develops enterprise software for large organizations.

Revenue increased by 18% during the fiscal year. Operating income
increased by 12%, driven by higher subscription revenue and improved
operating leverage.

The company also changed its product strategy. Management shifted
investment toward artificial intelligence capabilities and platform
integration.

The new strategy focuses on expanding the core platform while adding
AI-powered features. The company expects these capabilities to improve
customer retention and increase adoption of additional products.

The company also reduced investment in several lower-priority products.
Resources were redirected toward products with stronger growth potential.

Management stated that the strategy will continue to evolve as customer
demand for AI-enabled enterprise software develops.
""".strip()


def calculate_sizes(chunks):
    if not chunks:
        return {
            "count": 0,
            "average": 0,
            "minimum": 0,
            "maximum": 0,
        }

    sizes = [len(chunk) for chunk in chunks]

    return {
        "count": len(chunks),
        "average": round(mean(sizes), 2),
        "minimum": min(sizes),
        "maximum": max(sizes),
    }


def print_result(name, chunks):
    stats = calculate_sizes(chunks)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"Chunks:   {stats['count']}")
    print(f"Average:  {stats['average']} characters")
    print(f"Minimum:  {stats['minimum']} characters")
    print(f"Maximum:  {stats['maximum']} characters")

    for index, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {index}:")
        print(chunk)


def main():
    print("=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    fixed = fixed_size_chunks(
        TEXT,
        chunk_size=200,
    )

    paragraphs = paragraph_chunks(TEXT)

    sentences = sentence_chunks(TEXT)

    tokens = token_chunks(
        TEXT,
        chunk_size=50,
    )

    recursive = recursive_chunks(
        TEXT,
        chunk_size=200,
    )

    sliding = sliding_window_chunks(
        TEXT,
        chunk_size=200,
        overlap=50,
    )

    parent_child = parent_child_chunks(
        TEXT,
        document_id="demo-document",
        source="demo.txt",
        source_type="text",
        parent_size=300,
        child_size=100,
    )

    print_result(
        "1. Fixed-size",
        fixed,
    )

    print_result(
        "2. Paragraph",
        paragraphs,
    )

    print_result(
        "3. Sentence",
        sentences,
    )

    print_result(
        "4. Token",
        tokens,
    )

    print_result(
        "5. Recursive",
        recursive,
    )

    print_result(
        "6. Sliding window",
        sliding,
    )

    print("\nParent-child")
    print("------------")
    print(f"Parent/child records: {len(parent_child)}")

    for item in parent_child:
        print(
            f"\nParent: {item.parent_id}"
            f"\nChild:  {item.child_id}"
            f"\nChild text: {item.child_text}"
        )


if __name__ == "__main__":
    main()