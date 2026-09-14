from src.rag_engine.retrieval.metrics import (
    dcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def main() -> None:
    relevant = {
        "chunk-a",
        "chunk-b",
        "chunk-c",
    }

    retrieved = [
        "chunk-x",
        "chunk-a",
        "chunk-y",
        "chunk-b",
        "chunk-c",
    ]

    # Top 3 contains only chunk-a from the
    # three relevant chunks.
    recall = recall_at_k(
        retrieved,
        relevant,
        k=3,
    )

    assert recall == 1 / 3

    precision = precision_at_k(
        retrieved,
        relevant,
        k=3,
    )

    assert precision == 1 / 3

    # First relevant result is at rank 2.
    rr = reciprocal_rank(
        retrieved,
        relevant,
    )

    assert rr == 1 / 2

    result_lists = [
        [
            "chunk-a",
            "chunk-x",
            "chunk-y",
        ],
        [
            "chunk-x",
            "chunk-b",
            "chunk-y",
        ],
        [
            "chunk-x",
            "chunk-y",
            "chunk-z",
        ],
    ]

    relevance_sets = [
        {"chunk-a"},
        {"chunk-b"},
        {"chunk-c"},
    ]

    mrr = mean_reciprocal_rank(
        result_lists,
        relevance_sets,
    )

    expected_mrr = (
        (1 / 1)
        + (1 / 2)
        + 0
    ) / 3

    assert mrr == expected_mrr

    grades = {
        "chunk-a": 3,
        "chunk-b": 2,
        "chunk-c": 1,
    }

    ranking = [
        "chunk-x",
        "chunk-a",
        "chunk-b",
    ]

    dcg = dcg_at_k(
        ranking,
        grades,
        k=3,
    )

    assert dcg > 0

    ndcg = ndcg_at_k(
        ranking,
        grades,
        k=3,
    )

    assert 0.0 <= ndcg <= 1.0

    ideal_ranking = [
        "chunk-a",
        "chunk-b",
        "chunk-c",
    ]

    ideal_ndcg = ndcg_at_k(
        ideal_ranking,
        grades,
        k=3,
    )

    assert ideal_ndcg == 1.0

    no_relevant_rr = reciprocal_rank(
        ["chunk-x", "chunk-y"],
        {"chunk-a"},
    )

    assert no_relevant_rr == 0.0

    no_relevant_mrr = mean_reciprocal_rank(
        [
            ["chunk-x"],
            ["chunk-y"],
        ],
        [
            {"chunk-a"},
            {"chunk-b"},
        ],
    )

    assert no_relevant_mrr == 0.0

    try:
        recall_at_k(
            retrieved,
            relevant,
            k=0,
        )

        raise AssertionError(
            "Expected ValueError for invalid k"
        )

    except ValueError:
        pass

    try:
        precision_at_k(
            retrieved,
            set(),
            k=3,
        )

        # Precision technically has no requirement
        # for a non-empty relevance set, so this is
        # intentionally allowed.
    except ValueError:
        raise AssertionError(
            "Precision should handle an empty "
            "relevance set"
        )

    try:
        mean_reciprocal_rank(
            [],
            [],
        )

        raise AssertionError(
            "Expected ValueError for empty result lists"
        )

    except ValueError:
        pass

    print("Recall@3:")
    print(f"  {recall:.4f}")

    print("\nPrecision@3:")
    print(f"  {precision:.4f}")

    print("\nReciprocal Rank:")
    print(f"  {rr:.4f}")

    print("\nMRR:")
    print(f"  {mrr:.4f}")

    print("\nnDCG@3:")
    print(f"  {ndcg:.4f}")

    print("\nIdeal nDCG@3:")
    print(f"  {ideal_ndcg:.4f}")

    print(
        "\nAll retrieval metric "
        "assertions passed."
    )


if __name__ == "__main__":
    main()