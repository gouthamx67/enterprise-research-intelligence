from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.sparse import BM25Retriever


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="acme-001",
            text=(
                "Acme increased investment in "
                "artificial intelligence."
            ),
            document_id="acme-report",
            source="acme-10k.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            start_page=25,
            end_page=25,
            metadata={
                "company": "Acme",
            },
        ),
        build_chunk(
            chunk_id="acme-002",
            text=(
                "Acme expanded platform integration "
                "across its enterprise products."
            ),
            document_id="acme-report",
            source="acme-10k.pdf",
            source_type="pdf",
            section_title="Platform Strategy",
            start_page=27,
            end_page=27,
            metadata={
                "company": "Acme",
            },
        ),
        build_chunk(
            chunk_id="globex-001",
            text=(
                "Globex increased investment in "
                "artificial intelligence."
            ),
            document_id="globex-report",
            source="globex-10k.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            start_page=30,
            end_page=30,
            metadata={
                "company": "Globex",
            },
        ),
    ]


def print_results(
    label,
    results,
):
    print(f"\n{label}")

    for result in results:
        print(
            f"Rank {result.rank} | "
            f"Score {result.score:.4f} | "
            f"{result.chunk_id} | "
            f"{result.metadata.get('company')}"
        )


def main():
    chunks = create_demo_chunks()

    sparse = BM25Retriever(
        chunks
    )

    dense = DenseIndex()

    dense.add_chunks(
        chunks
    )

    query = (
        "artificial intelligence investment"
    )

    print("=" * 70)
    print("FILTERED RETRIEVAL")
    print("=" * 70)

    print(f"\nQuery: {query}")

    filters = {
        "company": "Acme",
    }

    print(
        f"Filters: {filters}"
    )

    sparse_results = sparse.search(
        query,
        top_k=3,
        filters=filters,
    )

    dense_results = dense.search(
        query,
        top_k=3,
        filters=filters,
    )

    print_results(
        "\nBM25 results",
        sparse_results,
    )

    print_results(
        "\nDense results",
        dense_results,
    )

    print("\n" + "=" * 70)
    print("FILTER CHECK")
    print("=" * 70)

    all_results = (
        sparse_results
        + dense_results
    )

    for result in all_results:
        company = result.metadata.get(
            "company"
        )

        assert company == "Acme", (
            "A non-Acme result passed "
            "the metadata filter."
        )

    print(
        "\nAll returned results belong "
        "to Acme."
    )


if __name__ == "__main__":
    main()