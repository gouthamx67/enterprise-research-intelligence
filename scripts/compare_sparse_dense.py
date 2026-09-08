from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.sparse import BM25Retriever


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="chunk-001",
            text=(
                "The company increased investment in "
                "artificial intelligence capabilities."
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
                "Management expanded spending on "
                "machine learning capabilities and "
                "AI-powered enterprise software."
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
                "Revenue increased by 18 percent during "
                "the fiscal year."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Financial Results",
            start_page=10,
            end_page=10,
        ),
        build_chunk(
            chunk_id="chunk-004",
            text=(
                "Operating income increased by 12 percent "
                "because of improved operating leverage."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Financial Results",
            start_page=11,
            end_page=11,
        ),
        build_chunk(
            chunk_id="chunk-005",
            text=(
                "The company expanded platform integration "
                "to connect multiple enterprise products."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Platform Strategy",
            start_page=27,
            end_page=27,
        ),
    ]


def print_results(
    label,
    results,
):
    print(f"\n{label}")

    for result in results:
        print(
            f"  Rank {result.rank} | "
            f"Score {result.score:.4f} | "
            f"{result.chunk_id} | "
            f"{result.metadata.get('section_title')}"
        )


def main():
    print("=" * 70)
    print("SPARSE VS DENSE RETRIEVAL")
    print("=" * 70)

    chunks = create_demo_chunks()

    sparse_retriever = BM25Retriever(
        chunks
    )

    dense_retriever = DenseIndex()

    dense_retriever.add_chunks(
        chunks
    )

    queries = [
        "artificial intelligence investment",
        "How is the company increasing spending on AI?",
        "revenue growth",
        "changes to product strategy",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)

        sparse_results = (
            sparse_retriever.search(
                query,
                top_k=3,
            )
        )

        dense_results = (
            dense_retriever.search(
                query,
                top_k=3,
            )
        )

        print_results(
            "BM25 / Sparse",
            sparse_results,
        )

        print_results(
            "Dense / Embeddings",
            dense_results,
        )


if __name__ == "__main__":
    main()