from dataclasses import dataclass

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
)
from src.rag_engine.retrieval.evaluation_runner import (
    RetrievalEvaluationResult,
)
from src.rag_engine.retrieval.reranking_evaluation import (
    RerankingEvaluation,
    calculate_metric_improvement,
    evaluate_reranking,
    print_reranking_evaluation,
)


@dataclass
class FakeResult:
    score: float
    rank: int
    chunk_id: str
    text: str
    metadata: dict


class FakeRetriever:
    """
    Deterministic retriever used to demonstrate
    the effect of reranking.
    """

    def __init__(self, results):
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        return self.results[query][:top_k]


def build_dataset():
    return EvaluationDataset(
        [
            EvaluationQuery(
                query="product strategy",
                relevant_chunk_ids={
                    "chunk-strategy",
                },
            ),
            EvaluationQuery(
                query="pricing strategy",
                relevant_chunk_ids={
                    "chunk-pricing",
                },
            ),
        ]
    )


def build_baseline():
    return FakeRetriever(
        {
            "product strategy": [
                FakeResult(
                    0.95,
                    1,
                    "chunk-noise-1",
                    "General company information.",
                    {},
                ),
                FakeResult(
                    0.90,
                    2,
                    "chunk-noise-2",
                    "General market information.",
                    {},
                ),
                FakeResult(
                    0.85,
                    3,
                    "chunk-strategy",
                    "The company changed its product strategy.",
                    {},
                ),
            ],
            "pricing strategy": [
                FakeResult(
                    0.95,
                    1,
                    "chunk-noise-3",
                    "General financial information.",
                    {},
                ),
                FakeResult(
                    0.90,
                    2,
                    "chunk-pricing",
                    "The company changed its pricing strategy.",
                    {},
                ),
                FakeResult(
                    0.80,
                    3,
                    "chunk-noise-4",
                    "General company information.",
                    {},
                ),
            ],
        }
    )


def build_reranked():
    """
    Simulates the same candidate pool after a
    successful reranking stage.
    """

    return FakeRetriever(
        {
            "product strategy": [
                FakeResult(
                    0.99,
                    1,
                    "chunk-strategy",
                    "The company changed its product strategy.",
                    {},
                ),
                FakeResult(
                    0.60,
                    2,
                    "chunk-noise-1",
                    "General company information.",
                    {},
                ),
                FakeResult(
                    0.50,
                    3,
                    "chunk-noise-2",
                    "General market information.",
                    {},
                ),
            ],
            "pricing strategy": [
                FakeResult(
                    0.98,
                    1,
                    "chunk-pricing",
                    "The company changed its pricing strategy.",
                    {},
                ),
                FakeResult(
                    0.60,
                    2,
                    "chunk-noise-3",
                    "General financial information.",
                    {},
                ),
                FakeResult(
                    0.50,
                    3,
                    "chunk-noise-4",
                    "General company information.",
                    {},
                ),
            ],
        }
    )


def main():
    dataset = build_dataset()

    baseline = build_baseline()
    reranked = build_reranked()

    evaluation = evaluate_reranking(
        baseline,
        reranked,
        dataset,
        k=3,
    )

    # ---------------------------------------------
    # Verify result types
    # ---------------------------------------------

    assert isinstance(
        evaluation,
        RerankingEvaluation,
    )

    assert isinstance(
        evaluation.baseline,
        RetrievalEvaluationResult,
    )

    assert isinstance(
        evaluation.reranked,
        RetrievalEvaluationResult,
    )

    # ---------------------------------------------
    # Baseline
    #
    # Query 1:
    # relevant at rank 3
    #
    # Query 2:
    # relevant at rank 2
    # ---------------------------------------------

    assert (
        evaluation.baseline.recall_at_k
        == 1.0
    )

    assert (
        evaluation.baseline.precision_at_k
        == 1 / 3
    )

    expected_baseline_mrr = (
        (1 / 3) + (1 / 2)
    ) / 2

    assert (
        evaluation.baseline.mrr
        == expected_baseline_mrr
    )

    # ---------------------------------------------
    # Reranked
    #
    # Both relevant chunks are rank 1.
    # ---------------------------------------------

    assert (
        evaluation.reranked.recall_at_k
        == 1.0
    )

    assert (
        evaluation.reranked.precision_at_k
        == 1 / 3
    )

    assert (
        evaluation.reranked.mrr
        == 1.0
    )

    # nDCG must improve because the relevant
    # documents moved to rank 1.
    assert (
        evaluation.reranked.ndcg_at_k
        > evaluation.baseline.ndcg_at_k
    )

    # ---------------------------------------------
    # MRR must improve
    # ---------------------------------------------

    assert (
        evaluation.reranked.mrr
        > evaluation.baseline.mrr
    )

    # ---------------------------------------------
    # Recall should remain unchanged
    #
    # Reranking cannot recover a document that
    # retrieval never produced.
    # ---------------------------------------------

    assert (
        evaluation.reranked.recall_at_k
        == evaluation.baseline.recall_at_k
    )

    # ---------------------------------------------
    # Calculate improvement
    # ---------------------------------------------

    mrr_improvement = (
        calculate_metric_improvement(
            evaluation,
            "MRR",
        )
    )

    expected_mrr_improvement = (
        (1.0 - expected_baseline_mrr)
        / expected_baseline_mrr
        * 100.0
    )

    assert abs(
        mrr_improvement
        - expected_mrr_improvement
    ) < 0.000001

    # ---------------------------------------------
    # Invalid metric
    # ---------------------------------------------

    try:
        calculate_metric_improvement(
            evaluation,
            "invalid",
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid dataset
    # ---------------------------------------------

    try:
        evaluate_reranking(
            baseline,
            reranked,
            None,
            k=3,
        )

        raise AssertionError(
            "Expected TypeError"
        )

    except TypeError:
        pass

    # ---------------------------------------------
    # Invalid K
    # ---------------------------------------------

    try:
        evaluate_reranking(
            baseline,
            reranked,
            dataset,
            k=0,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Display report
    # ---------------------------------------------

    print_reranking_evaluation(
        evaluation
    )

    print()

    print(
        "All reranking evaluation "
        "assertions passed."
    )


if __name__ == "__main__":
    main()