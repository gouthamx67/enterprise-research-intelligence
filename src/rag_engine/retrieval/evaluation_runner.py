from dataclasses import dataclass

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
)
from src.rag_engine.retrieval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass
class RetrievalEvaluationResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


def evaluate_retriever(
    retriever,
    dataset: EvaluationDataset,
    k: int = 5,
) -> RetrievalEvaluationResult:
    """
    Evaluate a retriever against an evaluation dataset.

    The retriever must expose:

        search(query, top_k=k)

    The evaluation dataset provides the ground-truth
    relevant chunk IDs.
    """

    if retriever is None:
        raise ValueError("retriever cannot be None")

    if not isinstance(dataset, EvaluationDataset):
        raise TypeError(
            "dataset must be an EvaluationDataset"
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than 0"
        )

    recalls = []
    precisions = []
    result_lists = []
    relevance_sets = []
    ndcg_scores = []

    for example in dataset:
        results = retriever.search(
            example.query,
            top_k=k,
        )

        retrieved_ids = [
            result.chunk_id
            for result in results
        ]

        relevant_ids = example.relevant_chunk_ids

        recalls.append(
            recall_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
        )

        precisions.append(
            precision_at_k(
                retrieved_ids,
                relevant_ids,
                k,
            )
        )

        result_lists.append(
            retrieved_ids
        )

        relevance_sets.append(
            relevant_ids
        )

        relevance_grades = {
            chunk_id: 1.0
            for chunk_id in relevant_ids
        }

        ndcg_scores.append(
            ndcg_at_k(
                retrieved_ids,
                relevance_grades,
                k,
            )
        )

    return RetrievalEvaluationResult(
        recall_at_k=sum(recalls) / len(recalls),
        precision_at_k=(
            sum(precisions) / len(precisions)
        ),
        mrr=mean_reciprocal_rank(
            result_lists,
            relevance_sets,
        ),
        ndcg_at_k=(
            sum(ndcg_scores)
            / len(ndcg_scores)
        ),
        k=k,
    )