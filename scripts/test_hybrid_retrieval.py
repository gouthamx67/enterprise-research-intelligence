from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.hybrid import (
    HybridRetriever,
)


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="acme-ai-001",
            text=(
                "Acme increased investment in "
                "artificial intelligence capabilities."
            ),
            document_id="acme-report",
            source="acme-10k.pdf",
            source_type="pdf",
            section_title="AI Strategy",
            start_page=25,
            end_page=25,
            metadata={
                "company": "Acme",
                "document_date": "2026-01-15",
            },
        ),
        build_chunk(
            chunk_id="acme-platform-001",
            text=(
                "Acme expanded platform integration "
                "across enterprise products."
            ),
            document_id="acme-report",
            source="acme-10k.pdf",
            source_type="pdf",
            section_title="Platform Strategy",
            start_page=27,
            end_page=27,
            metadata={
                "company": "Acme",
                "document_date": "2026-02-01",
            },
        ),
        build_chunk(
            chunk_id="acme-finance-001",
            text=(
                "Acme revenue increased by 18 percent "
                "during the fiscal year."
            ),
            document_id="acme-report",
            source="acme-10k.pdf",
            source_type="pdf",
            section_title="Financial Results",
            start_page=10,
            end_page=10,
            metadata={
                "company": "Acme",
                "document_date": "2026-01-15",
            },
        ),
        build_chunk(
            chunk_id="globex-ai-001",
            text=(
                "Globex increased investment in "
                "artificial intelligence capabilities."
            ),
            document_id="globex-report",
            source="globex-10k.pdf",
            source_type="pdf",
            section_title="AI Strategy",
            start_page=30,
            end_page=30,
            metadata={
                "company": "Globex",
                "document_date": "2026-01-20",
            },
        ),
    ]


def main():
    print("=" * 70)
    print("HYBRID RETRIEVAL")
    print("=" * 70)

    chunks = create_demo_chunks()

    retriever = HybridRetriever(
        chunks,
        rrf_k=60,
    )

    query = (
        "How did the company change "
        "its AI investment strategy?"
    )

    print(f"\nQuery: {query}")

    results = retriever.search(
        query,
        top_k=3,
    )

    print("\nHybrid results:")

    for result in results:
        print(
            "\n"
            f"Rank: {result.rank}\n"
            f"RRF score: {result.score:.6f}\n"
            f"Chunk: {result.chunk_id}\n"
            f"Company: "
            f"{result.metadata.get('company')}\n"
            f"Section: "
            f"{result.metadata.get('section_title')}\n"
            f"Text: {result.text}"
        )

    print("\n" + "=" * 70)
    print("FILTERED HYBRID RETRIEVAL")
    print("=" * 70)

    filtered_results = retriever.search(
        query,
        top_k=3,
        filters={
            "company": "Acme",
        },
    )

    print("\nFiltered results:")

    for result in filtered_results:
        print(
            f"Rank {result.rank} | "
            f"{result.chunk_id} | "
            f"{result.metadata.get('company')}"
        )

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)

    assert len(results) <= 3

    assert len(filtered_results) <= 3

    for result in filtered_results:
        assert (
            result.metadata.get("company")
            == "Acme"
        )

    print(
        "\nAll hybrid retrieval assertions passed."
    )


if __name__ == "__main__":
    main()