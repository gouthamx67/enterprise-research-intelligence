from dataclasses import dataclass

from src.rag_engine.retrieval.comparison import (
    RetrievalComparison,
)


@dataclass
class MetricWinner:
    metric: str
    retriever: str
    value: float


@dataclass
class EvaluationReport:
    comparisons: list[RetrievalComparison]
    winners: list[MetricWinner]


def build_evaluation_report(
    comparisons: list[RetrievalComparison],
) -> EvaluationReport:
    """
    Build a report identifying the best retriever
    for each evaluation metric.
    """

    if not comparisons:
        raise ValueError(
            "comparisons cannot be empty"
        )

    metrics = {
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

    winners = []

    for metric_name, getter in metrics.items():
        winner = max(
            comparisons,
            key=lambda comparison: getter(
                comparison.result
            ),
        )

        winners.append(
            MetricWinner(
                metric=metric_name,
                retriever=winner.name,
                value=getter(
                    winner.result
                ),
            )
        )

    return EvaluationReport(
        comparisons=comparisons,
        winners=winners,
    )


def calculate_improvement(
    baseline: RetrievalComparison,
    candidate: RetrievalComparison,
    metric: str,
) -> float:
    """
    Calculate relative percentage improvement of a
    candidate over a baseline.

    Example:

        baseline MRR = 0.50
        candidate MRR = 0.60

        improvement = 20%
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
    ](baseline.result)

    candidate_value = metric_getters[
        metric
    ](candidate.result)

    if baseline_value == 0.0:
        if candidate_value == 0.0:
            return 0.0

        return float("inf")

    return (
        (candidate_value - baseline_value)
        / baseline_value
        * 100.0
    )


def print_evaluation_report(
    report: EvaluationReport,
) -> None:
    """
    Print a human-readable evaluation report.
    """

    print("Retrieval Evaluation Report")
    print("=" * 70)

    print()
    print("Results")
    print("-" * 70)

    print(
        f"{'Retriever':<15}"
        f"{'Recall':>10}"
        f"{'Precision':>12}"
        f"{'MRR':>10}"
        f"{'nDCG':>10}"
    )

    print("-" * 57)

    for comparison in report.comparisons:
        result = comparison.result

        print(
            f"{comparison.name:<15}"
            f"{result.recall_at_k:>10.4f}"
            f"{result.precision_at_k:>12.4f}"
            f"{result.mrr:>10.4f}"
            f"{result.ndcg_at_k:>10.4f}"
        )

    print()
    print("Metric Winners")
    print("-" * 70)

    for winner in report.winners:
        print(
            f"{winner.metric:<15}"
            f"{winner.retriever:<15}"
            f"{winner.value:.4f}"
        )