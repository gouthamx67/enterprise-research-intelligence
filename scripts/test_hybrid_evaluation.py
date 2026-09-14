from dataclasses import dataclass

from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.hybrid import HybridRetriever
from src.rag_engine.retrieval.sparse import BM25Retriever


@dataclass
class EvaluationQuery:
    query: str
    relevant_chunk_ids: set[str]


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids cannot be empty")

    retrieved = set(retrieved_chunk_ids[:k])
    relevant_retrieved = retrieved.intersection(relevant_chunk_ids)

    return len(relevant_retrieved) / len(relevant_chunk_ids)


def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved = retrieved_chunk_ids[:k]

    if not retrieved:
        return 0.0

    relevant_retrieved = set(retrieved).intersection(relevant_chunk_ids)

    return len(relevant_retrieved) / len(retrieved)


def evaluate_retriever(
    retriever,
    evaluation_queries: list[EvaluationQuery],
    top_k: int,
) -> dict[str, float]:
    recall_scores = []
    precision_scores = []

    for item in evaluation_queries:
        results = retriever.search(
            item.query,
            top_k=top_k,
        )

        retrieved_ids = [
            result.chunk_id
            for result in results
        ]

        recall_scores.append(
            recall_at_k(
                retrieved_ids,
                item.relevant_chunk_ids,
                top_k,
            )
        )

        precision_scores.append(
            precision_at_k(
                retrieved_ids,
                item.relevant_chunk_ids,
                top_k,
            )
        )

    return {
        "recall": sum(recall_scores) / len(recall_scores),
        "precision": sum(precision_scores) / len(precision_scores),
    }


def print_query_results(
    retriever_name: str,
    retriever,
    evaluation_queries: list[EvaluationQuery],
    top_k: int,
) -> None:
    print(f"\n=== {retriever_name} ===")

    for item in evaluation_queries:
        results = retriever.search(
            item.query,
            top_k=top_k,
        )

        retrieved_ids = [
            result.chunk_id
            for result in results
        ]

        recall = recall_at_k(
            retrieved_ids,
            item.relevant_chunk_ids,
            top_k,
        )

        precision = precision_at_k(
            retrieved_ids,
            item.relevant_chunk_ids,
            top_k,
        )

        print(f"\nQuery: {item.query}")
        print(f"Relevant: {sorted(item.relevant_chunk_ids)}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Recall@{top_k}: {recall:.2f}")
        print(f"Precision@{top_k}: {precision:.2f}")


def main() -> None:
    chunks = [
        {
            "chunk_id": "chunk-001",
            "text": (
                "The company launched a new enterprise analytics "
                "platform focused on real-time dashboards."
            ),
        },
        {
            "chunk_id": "chunk-002",
            "text": (
                "The product strategy shifted toward enterprise "
                "customers and larger organizations."
            ),
        },
        {
            "chunk_id": "chunk-003",
            "text": (
                "The company introduced usage-based pricing for "
                "its analytics products."
            ),
        },
        {
            "chunk_id": "chunk-004",
            "text": (
                "The company expanded its platform with workflow "
                "automation and AI-assisted features."
            ),
        },
        {
            "chunk_id": "chunk-005",
            "text": (
                "Revenue increased as adoption of the enterprise "
                "product portfolio grew."
            ),
        },
        {
            "chunk_id": "chunk-006",
            "text": (
                "The company announced a partnership with a major "
                "cloud provider."
            ),
        },
    ]

    from src.rag_engine.core.document import Chunk

    chunk_objects = [
        Chunk(
            chunk_id=item["chunk_id"],
            text=item["text"],
            document_id="eval-doc",
            source="evaluation",
            source_type="test",
        )
        for item in chunks
    ]

    evaluation_queries = [
        EvaluationQuery(
            query="enterprise product strategy",
            relevant_chunk_ids={
                "chunk-002",
                "chunk-004",
            },
        ),
        EvaluationQuery(
            query="pricing strategy",
            relevant_chunk_ids={
                "chunk-003",
            },
        ),
        EvaluationQuery(
            query="new analytics platform",
            relevant_chunk_ids={
                "chunk-001",
            },
        ),
        EvaluationQuery(
            query="AI workflow features",
            relevant_chunk_ids={
                "chunk-004",
            },
        ),
    ]

    top_k = 3

    sparse = BM25Retriever(chunk_objects)

    dense = DenseIndex()
    dense.add_chunks(chunk_objects)

    hybrid = HybridRetriever(chunk_objects)

    retrievers = {
        "BM25": sparse,
        "Dense": dense,
        "Hybrid": hybrid,
    }

    for name, retriever in retrievers.items():
        print_query_results(
            name,
            retriever,
            evaluation_queries,
            top_k,
        )

    print("\n=== Aggregate Results ===")

    aggregate_results = {}

    for name, retriever in retrievers.items():
        metrics = evaluate_retriever(
            retriever,
            evaluation_queries,
            top_k,
        )

        aggregate_results[name] = metrics

        print(
            f"{name}: "
            f"Recall@{top_k}={metrics['recall']:.2f}, "
            f"Precision@{top_k}={metrics['precision']:.2f}"
        )

    best_recall = max(
        aggregate_results.items(),
        key=lambda item: item[1]["recall"],
    )

    best_precision = max(
        aggregate_results.items(),
        key=lambda item: item[1]["precision"],
    )

    print(
        f"\nBest Recall@{top_k}: "
        f"{best_recall[0]} "
        f"({best_recall[1]['recall']:.2f})"
    )

    print(
        f"Best Precision@{top_k}: "
        f"{best_precision[0]} "
        f"({best_precision[1]['precision']:.2f})"
    )


if __name__ == "__main__":
    main()