from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.dense import DenseIndex


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="chunk-001",
            text=(
                "The company increased investment "
                "in artificial intelligence."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            start_page=25,
            end_page=25,
        ),
        build_chunk(
            chunk_id="chunk-002",
            text=(
                "Management expanded spending "
                "on machine learning capabilities."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="AI Strategy",
            start_page=26,
            end_page=26,
        ),
        build_chunk(
            chunk_id="chunk-003",
            text=(
                "Revenue increased by 18 percent "
                "during the fiscal year."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Financial Results",
            start_page=10,
            end_page=10,
        ),
    ]


def main():
    chunks = create_demo_chunks()

    index = DenseIndex()

    index.add_chunks(chunks)

    print("=" * 60)
    print("DENSE INDEX")
    print("=" * 60)

    print(
        f"\nIndexed chunks: "
        f"{index.size}"
    )

    print(
        f"Embedding dimensions: "
        f"{index.dimensions}"
    )

    print("\nStored chunk IDs:")

    for item in index.embedded_chunks:
        print(
            f"- {item.chunk_id}"
        )

    print("\nMetadata example:")

    first = index.embedded_chunks[0]

    print(
        f"Source: "
        f"{first.metadata['source']}"
    )

    print(
        f"Section: "
        f"{first.metadata['section_title']}"
    )

    print(
        f"Page: "
        f"{first.metadata['start_page']}"
    )


if __name__ == "__main__":
    main()