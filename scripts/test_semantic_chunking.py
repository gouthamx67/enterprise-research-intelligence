from src.rag_engine.chunking.semantic import (
    SemanticChunker,
)


TEXT = """
The company launched a new enterprise platform in 2025.
The platform combines analytics, automation, and collaboration.
The platform initially targeted mid-market customers.

In 2026, the company changed its product strategy.
It increased investment in AI capabilities and reduced investment
in several legacy products.
The company also changed its target customers.

The company is now focusing more heavily on large enterprise
accounts.
It is investing in enterprise-grade AI capabilities.
The strategy emphasizes automation and analytics.
"""


def main():
    chunker = SemanticChunker()

    chunks = chunker.chunk(
        TEXT,
        threshold=0.35,
    )

    print("SEMANTIC CHUNKS")
    print("================")

    for index, chunk in enumerate(chunks, start=1):
        print()
        print(f"--- CHUNK {index} ---")
        print(chunk)


if __name__ == "__main__":
    main()