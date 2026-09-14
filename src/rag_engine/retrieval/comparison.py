from dataclasses import dataclass

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
)
from src.rag_engine.retrieval.evaluation_runner import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)


@dataclass
class RetrievalComparison:
    name: str
    result: RetrievalEvaluationResult


def compare_retrievers(
    retrievers: dict[str, object],
    dataset: EvaluationDataset,
    k: int = 5,
) -> list[RetrievalComparison]:
    """
    Evaluate multiple retrievers against the same dataset.

    Args:
        retrievers:
            Mapping of retriever name -> retriever object.

        dataset:
            Ground-truth evaluation dataset.

        k:
            Number of retrieved results to evaluate.

    Returns:
        Results ordered in the same order as the
        supplied retrievers dictionary.
    """

    if not retrievers:
        raise ValueError(
            "retrievers cannot be empty"
        )

    comparisons = []

    for name, retriever in retrievers.items():
        if not name.strip():
            raise ValueError(
                "retriever names cannot be empty"
            )

        result = evaluate_retriever(
            retriever,
            dataset,
            k=k,
        )

        comparisons.append(
            RetrievalComparison(
                name=name,
                result=result,
            )
        )

    return comparisons


def print_comparison_table(
    comparisons: list[RetrievalComparison],
) -> None:
    """
    Print retrieval evaluation results
    in a compact table.
    """

    if not comparisons:
        raise ValueError(
            "comparisons cannot be empty"
        )

    print(
        f"{'Retriever':<15}"
        f"{'Recall':>10}"
        f"{'Precision':>12}"
        f"{'MRR':>10}"
        f"{'nDCG':>10}"
    )

    print("-" * 57)

    for comparison in comparisons:
        result = comparison.result

        print(
            f"{comparison.name:<15}"
            f"{result.recall_at_k:>10.4f}"
            f"{result.precision_at_k:>12.4f}"
            f"{result.mrr:>10.4f}"
            f"{result.ndcg_at_k:>10.4f}"
        )