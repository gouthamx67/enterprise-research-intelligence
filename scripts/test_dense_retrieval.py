from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.dense import (
    DenseIndex,
    cosine_similarity,
)


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
        build_chunk(
            chunk_id="chunk-004",
            text=(
                "The company opened several new "
                "regional offices."
            ),
            document_id="company-report",
            source="annual-report.pdf",
            source_type="pdf",
            section_title="Operations",
            start_page=40,
            end_page=40,
        ),
    ]


def main():
    print("=" * 60)
    print("DENSE RETRIEVAL")
    print("=" * 60)

    chunks = create_demo_chunks()

    index = DenseIndex()

    index.add_chunks(chunks)

    query = (
        "How did the company change "
        "its AI investment strategy?"
    )

    results = index.search(
        query,
        top_k=3,
    )

    print(f"\nQuery: {query}")

    print("\nTop results:")

    for result in results:
        print("\n" + "-" * 60)

        print(
            f"Rank:  {result.rank}"
        )

        print(
            f"Score: {result.score:.4f}"
        )

        print(
            f"Chunk: {result.chunk_id}"
        )

        print(
            f"Section: "
            f"{result.metadata.get('section_title')}"
        )

        print(
            f"Page: "
            f"{result.metadata.get('start_page')}"
        )

        print(
            f"Text: {result.text}"
        )

    print("\n" + "=" * 60)
    print("COSINE SIMILARITY SANITY CHECK")
    print("=" * 60)

    vector_a = index.embedded_chunks[0].embedding
    vector_b = index.embedded_chunks[1].embedding

    similarity = cosine_similarity(
        vector_a,
        np_matrix(vector_b),
    )

    print(
        f"\nSimilarity between chunk 1 "
        f"and chunk 2: "
        f"{similarity[0]:.4f}"
    )


def np_matrix(vector):
    """
    Convert one vector into a one-row matrix
    for the cosine_similarity function.
    """

    import numpy as np

    return np.asarray(
        vector
    ).reshape(1, -1)


if __name__ == "__main__":
    main()