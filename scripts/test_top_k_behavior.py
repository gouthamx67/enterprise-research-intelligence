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
            chunk_id="ai-002",
            text=(
                "Management expanded spending on "
                "machine learning capabilities."
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
                "integration across products."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Platform Strategy",
        ),
        build_chunk(
            chunk_id="finance-001",
            text=(
                "Revenue increased during the "
                "fiscal year."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Financial Results",
        ),
    ]


def main():
    print("=" * 70)
    print("TOP-K BEHAVIOR")
    print("=" * 70)

    chunks = create_demo_chunks()

    index = DenseIndex()

    index.add_chunks(chunks)

    query = (
        "How did the company change "
        "its AI investment?"
    )

    # ---------------------------------------------------------
    # top_k = 1
    # ---------------------------------------------------------

    results_k1 = index.search(
        query,
        top_k=1,
    )

    print("\ntop_k = 1")
    print("-" * 30)

    for result in results_k1:
        print(
            f"{result.rank} | "
            f"{result.chunk_id} | "
            f"{result.score:.4f}"
        )

    # ---------------------------------------------------------
    # top_k = 3
    # ---------------------------------------------------------

    results_k3 = index.search(
        query,
        top_k=3,
    )

    print("\ntop_k = 3")
    print("-" * 30)

    for result in results_k3:
        print(
            f"{result.rank} | "
            f"{result.chunk_id} | "
            f"{result.score:.4f}"
        )

    # ---------------------------------------------------------
    # top_k larger than corpus
    # ---------------------------------------------------------

    results_k10 = index.search(
        query,
        top_k=10,
    )

    print("\ntop_k = 10")
    print("-" * 30)

    for result in results_k10:
        print(
            f"{result.rank} | "
            f"{result.chunk_id} | "
            f"{result.score:.4f}"
        )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)

    assert len(results_k1) <= 1

    assert len(results_k3) <= 3

    assert len(results_k10) <= len(chunks)

    assert [
        result.rank
        for result in results_k3
    ] == [1, 2, 3]

    print(
        "\nAll top-k assertions passed."
    )


if __name__ == "__main__":
    main()