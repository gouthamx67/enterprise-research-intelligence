import math


def recall_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """
    Fraction of relevant chunks retrieved in the top-k results.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    if not relevant_chunk_ids:
        raise ValueError(
            "relevant_chunk_ids cannot be empty"
        )

    retrieved = retrieved_chunk_ids[:k]

    relevant_retrieved = set(retrieved).intersection(
        relevant_chunk_ids
    )

    return (
        len(relevant_retrieved)
        / len(relevant_chunk_ids)
    )


def precision_at_k(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
    k: int,
) -> float:
    """
    Fraction of retrieved top-k chunks that are relevant.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    retrieved = retrieved_chunk_ids[:k]

    if not retrieved:
        return 0.0

    relevant_retrieved = set(retrieved).intersection(
        relevant_chunk_ids
    )

    return len(relevant_retrieved) / len(retrieved)


def reciprocal_rank(
    retrieved_chunk_ids: list[str],
    relevant_chunk_ids: set[str],
) -> float:
    """
    Reciprocal rank of the first relevant result.

    Returns 0.0 when no relevant result is retrieved.
    """

    if not relevant_chunk_ids:
        raise ValueError(
            "relevant_chunk_ids cannot be empty"
        )

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(
    result_lists: list[list[str]],
    relevance_sets: list[set[str]],
) -> float:
    """
    Mean Reciprocal Rank across multiple queries.
    """

    if not result_lists:
        raise ValueError(
            "result_lists cannot be empty"
        )

    if len(result_lists) != len(relevance_sets):
        raise ValueError(
            "result_lists and relevance_sets "
            "must have the same length"
        )

    scores = [
        reciprocal_rank(
            retrieved,
            relevant,
        )
        for retrieved, relevant in zip(
            result_lists,
            relevance_sets,
        )
    ]

    return sum(scores) / len(scores)


def dcg_at_k(
    retrieved_chunk_ids: list[str],
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain at k.

    Relevance grades may be binary or graded.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    score = 0.0

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids[:k],
        start=1,
    ):
        relevance = relevance_grades.get(
            chunk_id,
            0.0,
        )

        score += (
            (2**relevance - 1)
            / math.log2(rank + 1)
        )

    return score


def ndcg_at_k(
    retrieved_chunk_ids: list[str],
    relevance_grades: dict[str, float],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Returns 0.0 when there is no relevant gain.
    """

    if k <= 0:
        raise ValueError("k must be greater than 0")

    actual_dcg = dcg_at_k(
        retrieved_chunk_ids,
        relevance_grades,
        k,
    )

    ideal_chunk_ids = sorted(
        relevance_grades,
        key=lambda chunk_id: (
            relevance_grades[chunk_id]
        ),
        reverse=True,
    )

    ideal_dcg = dcg_at_k(
        ideal_chunk_ids,
        relevance_grades,
        k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg