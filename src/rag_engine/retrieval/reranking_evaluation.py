from dataclasses import dataclass

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
)
from src.rag_engine.retrieval.evaluation_runner import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)


@dataclass
class RerankingEvaluation:
    baseline: RetrievalEvaluationResult
    reranked: RetrievalEvaluationResult


def evaluate_reranking(
    baseline_retriever,
    reranked_retriever,
    dataset: EvaluationDataset,
    k: int = 5,
) -> RerankingEvaluation:
    """
    Compare a baseline retriever with a reranked
    retrieval pipeline.

    Both systems are evaluated against exactly the
    same evaluation dataset and at the same final K.
    """

    if baseline_retriever is None:
        raise ValueError(
            "baseline_retriever cannot be None"
        )

    if reranked_retriever is None:
        raise ValueError(
            "reranked_retriever cannot be None"
        )

    if not isinstance(
        dataset,
        EvaluationDataset,
    ):
        raise TypeError(
            "dataset must be an EvaluationDataset"
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than 0"
        )

    baseline = evaluate_retriever(
        baseline_retriever,
        dataset,
        k=k,
    )

    reranked = evaluate_retriever(
        reranked_retriever,
        dataset,
        k=k,
    )

    return RerankingEvaluation(
        baseline=baseline,
        reranked=reranked,
    )


def calculate_metric_improvement(
    evaluation: RerankingEvaluation,
    metric: str,
) -> float:
    """
    Calculate relative percentage improvement of the
    reranked system over the baseline.

    Returns:
        Percentage improvement.
    """

    metric_getters = {
        "Recall@K": lambda result: (
            result.recall_at_k
        ),
        "Precision@K": lambda result: (
            result.precision_at_k
        ),
        "MRR": lambda result: result.mrr,
        "nDCG@K": lambda result: (
            result.ndcg_at_k
        ),
    }

    if metric not in metric_getters:
        raise ValueError(
            f"unknown metric: {metric}"
        )

    baseline_value = metric_getters[
        metric
    ](evaluation.baseline)

    reranked_value = metric_getters[
        metric
    ](evaluation.reranked)

    if baseline_value == 0.0:
        if reranked_value == 0.0:
            return 0.0

        return float("inf")

    return (
        (reranked_value - baseline_value)
        / baseline_value
        * 100.0
    )


def print_reranking_evaluation(
    evaluation: RerankingEvaluation,
) -> None:
    """
    Print a side-by-side reranking evaluation.
    """

    baseline = evaluation.baseline
    reranked = evaluation.reranked

    print("Reranking Evaluation")
    print("=" * 70)

    print()

    print(
        f"{'System':<15}"
        f"{'Recall':>10}"
        f"{'Precision':>12}"
        f"{'MRR':>10}"
        f"{'nDCG':>10}"
    )

    print("-" * 57)

    print(
        f"{'Baseline':<15}"
        f"{baseline.recall_at_k:>10.4f}"
        f"{baseline.precision_at_k:>12.4f}"
        f"{baseline.mrr:>10.4f}"
        f"{baseline.ndcg_at_k:>10.4f}"
    )

    print(
        f"{'Reranked':<15}"
        f"{reranked.recall_at_k:>10.4f}"
        f"{reranked.precision_at_k:>12.4f}"
        f"{reranked.mrr:>10.4f}"
        f"{reranked.ndcg_at_k:>10.4f}"
    )

    print()

    print("Relative improvement")

    for metric in [
        "Recall@K",
        "Precision@K",
        "MRR",
        "nDCG@K",
    ]:
        improvement = calculate_metric_improvement(
            evaluation,
            metric,
        )

        if improvement == float("inf"):
            formatted = "inf"
        else:
            formatted = (
                f"{improvement:+.2f}%"
            )

        print(
            f"  {metric:<15}"
            f"{formatted}"
        )