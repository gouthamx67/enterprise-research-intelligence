from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.sparse import BM25Retriever


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="chunk-001",
            text=(
                "The company increased revenue by 18 percent "
                "during the fiscal year."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Financial Results",
            start_page=10,
            end_page=10,
        ),
        build_chunk(
            chunk_id="chunk-002",
            text=(
                "Management changed the product strategy "
                "by increasing investment in artificial "
                "intelligence capabilities."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            start_page=25,
            end_page=25,
        ),
        build_chunk(
            chunk_id="chunk-003",
            text=(
                "The company expanded its platform "
                "integration strategy to connect multiple "
                "enterprise products."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Platform Strategy",
            start_page=27,
            end_page=27,
        ),
        build_chunk(
            chunk_id="chunk-004",
            text=(
                "Operating expenses increased as the company "
                "continued hiring across engineering teams."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Operating Expenses",
            start_page=31,
            end_page=31,
        ),
    ]


def main():
    chunks = create_demo_chunks()

    retriever = BM25Retriever(chunks)

    query = (
        "What changed in the company's "
        "product strategy?"
    )

    results = retriever.search(
        query,
        top_k=3,
    )

    print("=" * 60)
    print("BM25 RETRIEVAL")
    print("=" * 60)

    print(f"\nQuery: {query}")

    for result in results:
        print("\n" + "-" * 60)

        print(f"Rank:    {result.rank}")
        print(f"Score:   {result.score:.4f}")
        print(f"Chunk:   {result.chunk_id}")

        print(
            f"Section: "
            f"{result.metadata.get('section_title')}"
        )

        print(
            f"Pages:   "
            f"{result.metadata.get('start_page')}"
            f"-"
            f"{result.metadata.get('end_page')}"
        )

        print(f"Text:    {result.text}")


if __name__ == "__main__":
    main()