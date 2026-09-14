from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.dense import DenseIndex


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="ai-001",
            text=(
                "The company increased investment "
                "in artificial intelligence."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="AI Strategy",
        ),
        build_chunk(
            chunk_id="platform-001",
            text=(
                "The company expanded platform "
                "integration across enterprise products."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Platform Strategy",
        ),
        build_chunk(
            chunk_id="finance-001",
            text=(
                "Revenue increased by 18 percent "
                "during the fiscal year."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Financial Results",
        ),
    ]


def print_results(
    title,
    results,
):
    print(f"\n{title}")
    print("-" * len(title))

    if not results:
        print("No results.")

        return

    for result in results:
        print(
            f"Rank {result.rank} | "
            f"Score {result.score:.4f} | "
            f"{result.chunk_id} | "
            f"{result.metadata.get('section_title')}"
        )


def main():
    print("=" * 70)
    print("DENSE SIMILARITY THRESHOLD")
    print("=" * 70)

    chunks = create_demo_chunks()

    index = DenseIndex()

    index.add_chunks(chunks)

    query = (
        "How is the company investing "
        "in artificial intelligence?"
    )

    print(f"\nQuery: {query}")

    # ---------------------------------------------------------
    # No threshold
    # ---------------------------------------------------------

    results_without_threshold = index.search(
        query,
        top_k=3,
    )

    print_results(
        "Without threshold",
        results_without_threshold,
    )

    # ---------------------------------------------------------
    # High threshold
    # ---------------------------------------------------------

    results_with_threshold = index.search(
        query,
        top_k=3,
        score_threshold=0.70,
    )

    print_results(
        "With threshold = 0.70",
        results_with_threshold,
    )

    # ---------------------------------------------------------
    # Impossible threshold for this demo
    # ---------------------------------------------------------

    no_results = index.search(
        query,
        top_k=3,
        score_threshold=1.0,
    )

    print_results(
        "With threshold = 1.0",
        no_results,
    )

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)

    assert len(
        results_without_threshold
    ) <= 3

    for result in results_with_threshold:
        assert result.score >= 0.70

    assert no_results == []

    print(
        "\nAll similarity-threshold assertions passed."
    )


if __name__ == "__main__":
    main()