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
                "Management expanded spending on machine "
                "learning capabilities and AI-powered "
                "enterprise software."
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


EVALUATION_DATASET = [
    {
        "query": "artificial intelligence investment",
        "relevant_chunks": {
            "chunk-001",
            "chunk-002",
        },
    },
    {
        "query": "revenue growth",
        "relevant_chunks": {
            "chunk-003",
        },
    },
    {
        "query": "operating income",
        "relevant_chunks": {
            "chunk-004",
        },
    },
    {
        "query": "platform integration strategy",
        "relevant_chunks": {
            "chunk-005",
        },
    },
]


def precision_at_k(
    retrieved_ids,
    relevant_ids,
    k,
):
    retrieved = retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_retrieved = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved
    )

    return relevant_retrieved / len(retrieved)


def recall_at_k(
    retrieved_ids,
    relevant_ids,
    k,
):
    retrieved = retrieved_ids[:k]

    if not relevant_ids:
        return 0.0

    relevant_retrieved = sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved
    )

    return relevant_retrieved / len(
        relevant_ids
    )


def evaluate_retriever(
    retriever,
    name,
    top_k=3,
):
    precisions = []
    recalls = []

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    for item in EVALUATION_DATASET:
        results = retriever.search(
            item["query"],
            top_k=top_k,
        )

        retrieved_ids = [
            result.chunk_id
            for result in results
        ]

        relevant_ids = item[
            "relevant_chunks"
        ]

        precision = precision_at_k(
            retrieved_ids,
            relevant_ids,
            top_k,
        )

        recall = recall_at_k(
            retrieved_ids,
            relevant_ids,
            top_k,
        )

        precisions.append(precision)
        recalls.append(recall)

        print(
            f"\nQuery: {item['query']}"
        )

        print(
            f"Expected: "
            f"{sorted(relevant_ids)}"
        )

        print(
            f"Retrieved: "
            f"{retrieved_ids}"
        )

        print(
            f"Precision@{top_k}: "
            f"{precision:.2f}"
        )

        print(
            f"Recall@{top_k}: "
            f"{recall:.2f}"
        )

    average_precision = (
        sum(precisions)
        / len(precisions)
    )

    average_recall = (
        sum(recalls)
        / len(recalls)
    )

    print("\n" + "-" * 70)

    print(
        f"Average Precision@{top_k}: "
        f"{average_precision:.2f}"
    )

    print(
        f"Average Recall@{top_k}: "
        f"{average_recall:.2f}"
    )


def main():
    chunks = create_demo_chunks()

    sparse_retriever = BM25Retriever(
        chunks
    )

    dense_retriever = DenseIndex()

    dense_retriever.add_chunks(
        chunks
    )

    evaluate_retriever(
        sparse_retriever,
        "BM25 / SPARSE RETRIEVAL",
    )

    evaluate_retriever(
        dense_retriever,
        "DENSE RETRIEVAL",
    )


if __name__ == "__main__":
    main()