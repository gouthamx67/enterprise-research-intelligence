from src.rag_engine.retrieval.comparison import (
    RetrievalComparison,
)
from src.rag_engine.retrieval.evaluation_report import (
    build_evaluation_report,
    calculate_improvement,
    print_evaluation_report,
)
from src.rag_engine.retrieval.evaluation_runner import (
    RetrievalEvaluationResult,
)


def make_comparison(
    name: str,
    recall: float,
    precision: float,
    mrr: float,
    ndcg: float,
) -> RetrievalComparison:
    return RetrievalComparison(
        name=name,
        result=RetrievalEvaluationResult(
            recall_at_k=recall,
            precision_at_k=precision,
            mrr=mrr,
            ndcg_at_k=ndcg,
            k=5,
        ),
    )


def main() -> None:
    bm25 = make_comparison(
        "BM25",
        recall=0.60,
        precision=0.40,
        mrr=0.50,
        ndcg=0.55,
    )

    dense = make_comparison(
        "Dense",
        recall=0.70,
        precision=0.35,
        mrr=0.60,
        ndcg=0.65,
    )

    hybrid = make_comparison(
        "Hybrid",
        recall=0.80,
        precision=0.50,
        mrr=0.75,
        ndcg=0.78,
    )

    comparisons = [
        bm25,
        dense,
        hybrid,
    ]

    report = build_evaluation_report(
        comparisons
    )

    assert len(report.comparisons) == 3
    assert len(report.winners) == 4

    winners = {
        winner.metric: winner
        for winner in report.winners
    }

    assert (
        winners["Recall@K"].retriever
        == "Hybrid"
    )

    assert (
        winners["Precision@K"].retriever
        == "Hybrid"
    )

    assert (
        winners["MRR"].retriever
        == "Hybrid"
    )

    assert (
        winners["nDCG@K"].retriever
        == "Hybrid"
    )

    recall_improvement = (
        calculate_improvement(
            bm25,
            hybrid,
            "Recall@K",
        )
    )

    # (0.80 - 0.60) / 0.60 * 100
    assert abs(
        recall_improvement - 33.3333333333
    ) < 0.000001

    mrr_improvement = (
        calculate_improvement(
            bm25,
            hybrid,
            "MRR",
        )
    )

    # (0.75 - 0.50) / 0.50 * 100
    assert abs(
        mrr_improvement - 50.0
    ) < 0.000001

    zero_baseline = make_comparison(
        "Zero",
        recall=0.0,
        precision=0.0,
        mrr=0.0,
        ndcg=0.0,
    )

    nonzero_candidate = make_comparison(
        "Candidate",
        recall=0.5,
        precision=0.5,
        mrr=0.5,
        ndcg=0.5,
    )

    infinite_improvement = (
        calculate_improvement(
            zero_baseline,
            nonzero_candidate,
            "MRR",
        )
    )

    assert infinite_improvement == float(
        "inf"
    )

    zero_improvement = (
        calculate_improvement(
            zero_baseline,
            zero_baseline,
            "MRR",
        )
    )

    assert zero_improvement == 0.0

    try:
        build_evaluation_report([])

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    try:
        calculate_improvement(
            bm25,
            hybrid,
            "unknown",
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print_evaluation_report(report)

    print()
    print(
        "All evaluation report "
        "assertions passed."
    )


if __name__ == "__main__":
    main()