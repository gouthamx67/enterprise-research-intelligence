from dataclasses import dataclass

from src.rag_engine.retrieval.models import (
    RetrievalResult,
)
from src.rag_engine.retrieval.reranker import (
    CrossEncoderReranker,
)


@dataclass
class FakeCrossEncoder:
    """
    Deterministic fake cross-encoder used to test
    reranking logic without downloading a model.
    """

    def predict(self, pairs):
        scores = []

        for query, text in pairs:
            text_lower = text.lower()

            # Strongest match.
            if (
                "product strategy" in text_lower
                or "strategy and roadmap" in text_lower
            ):
                scores.append(0.95)

            # Secondary relevant match.
            elif "pricing strategy" in text_lower:
                scores.append(0.80)

            # Irrelevant candidate.
            else:
                scores.append(0.10)

        return scores


def build_candidates():
    return [
        RetrievalResult(
            score=0.90,
            rank=1,
            chunk_id="chunk-pricing",
            text=(
                "The company changed its pricing "
                "strategy."
            ),
            metadata={
                "section": "pricing"
            },
        ),
        RetrievalResult(
            score=0.88,
            rank=2,
            chunk_id="chunk-noise",
            text=(
                "The company operates several "
                "regional offices."
            ),
            metadata={
                "section": "company"
            },
        ),
        RetrievalResult(
            score=0.82,
            rank=3,
            chunk_id="chunk-strategy",
            text=(
                "The company changed its product "
                "strategy and roadmap."
            ),
            metadata={
                "section": "strategy"
            },
        ),
    ]


def build_test_reranker():
    """
    Build a CrossEncoderReranker while replacing
    its real model with a deterministic fake model.
    """

    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model_name = "fake-model"
    reranker.model = FakeCrossEncoder()

    return reranker


def main():
    reranker = build_test_reranker()

    candidates = build_candidates()

    query = (
        "What changed in product strategy?"
    )

    results = reranker.rerank(
        query,
        candidates,
        top_k=3,
    )

    assert len(results) == 3

    # -------------------------------------------------
    # Verify reranking order
    # -------------------------------------------------

    assert (
        results[0].chunk_id
        == "chunk-strategy"
    )

    assert (
        results[1].chunk_id
        == "chunk-pricing"
    )

    assert (
        results[2].chunk_id
        == "chunk-noise"
    )

    # -------------------------------------------------
    # Verify new ranks
    # -------------------------------------------------

    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[2].rank == 3

    # -------------------------------------------------
    # Verify original ranks were preserved
    # -------------------------------------------------

    assert (
        results[0].original_rank == 3
    )

    assert (
        results[1].original_rank == 1
    )

    assert (
        results[2].original_rank == 2
    )

    # -------------------------------------------------
    # Verify scores
    # -------------------------------------------------

    assert results[0].score == 0.95
    assert results[1].score == 0.80
    assert results[2].score == 0.10

    assert (
        results[0].score
        >= results[1].score
        >= results[2].score
    )

    # -------------------------------------------------
    # Verify original retrieval scores remain
    # -------------------------------------------------

    assert (
        results[0].original_score
        == 0.82
    )

    assert (
        results[1].original_score
        == 0.90
    )

    assert (
        results[2].original_score
        == 0.88
    )

    # -------------------------------------------------
    # Verify metadata survives reranking
    # -------------------------------------------------

    assert (
        results[0].metadata["section"]
        == "strategy"
    )

    assert (
        results[1].metadata["section"]
        == "pricing"
    )

    assert (
        results[2].metadata["section"]
        == "company"
    )

    # -------------------------------------------------
    # Verify top_k
    # -------------------------------------------------

    top_two = reranker.rerank(
        query,
        candidates,
        top_k=2,
    )

    assert len(top_two) == 2

    assert (
        top_two[0].chunk_id
        == "chunk-strategy"
    )

    assert (
        top_two[1].chunk_id
        == "chunk-pricing"
    )

    # -------------------------------------------------
    # Empty candidates
    # -------------------------------------------------

    empty_results = reranker.rerank(
        query,
        [],
    )

    assert empty_results == []

    # -------------------------------------------------
    # Invalid query
    # -------------------------------------------------

    try:
        reranker.rerank(
            "",
            candidates,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # -------------------------------------------------
    # Invalid top_k
    # -------------------------------------------------

    try:
        reranker.rerank(
            query,
            candidates,
            top_k=0,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # -------------------------------------------------
    # Display results
    # -------------------------------------------------

    print("Initial retrieval ranking:")

    for candidate in candidates:
        print(
            f"  {candidate.rank}. "
            f"{candidate.chunk_id} "
            f"(score={candidate.score})"
        )

    print("\nAfter reranking:")

    for result in results:
        print(
            f"  {result.rank}. "
            f"{result.chunk_id} "
            f"(rerank_score="
            f"{result.score:.4f}, "
            f"original_rank="
            f"{result.original_rank})"
        )

    print(
        "\nAll reranker assertions passed."
    )


if __name__ == "__main__":
    main()