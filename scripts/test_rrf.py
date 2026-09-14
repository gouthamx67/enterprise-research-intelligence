from src.rag_engine.retrieval.hybrid import (
    reciprocal_rank_fusion,
)
from src.rag_engine.retrieval.models import (
    RetrievalResult,
)


def make_result(
    chunk_id,
    rank,
    score,
):
    return RetrievalResult(
        score=score,
        rank=rank,
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        metadata={
            "source": "demo.pdf",
        },
    )


def main():
    print("=" * 70)
    print("RECIPROCAL RANK FUSION")
    print("=" * 70)

    bm25_results = [
        make_result(
            "chunk-A",
            rank=1,
            score=8.4,
        ),
        make_result(
            "chunk-C",
            rank=2,
            score=6.2,
        ),
        make_result(
            "chunk-B",
            rank=3,
            score=4.8,
        ),
    ]

    dense_results = [
        make_result(
            "chunk-B",
            rank=1,
            score=0.91,
        ),
        make_result(
            "chunk-A",
            rank=2,
            score=0.87,
        ),
        make_result(
            "chunk-D",
            rank=3,
            score=0.72,
        ),
    ]

    fused_results = reciprocal_rank_fusion(
        [
            bm25_results,
            dense_results,
        ],
        k=60,
        top_k=4,
    )

    print("\nBM25:")
    for result in bm25_results:
        print(
            f"Rank {result.rank} | "
            f"{result.chunk_id} | "
            f"{result.score:.4f}"
        )

    print("\nDense:")
    for result in dense_results:
        print(
            f"Rank {result.rank} | "
            f"{result.chunk_id} | "
            f"{result.score:.4f}"
        )

    print("\nFused:")
    for result in fused_results:
        print(
            f"Rank {result.rank} | "
            f"{result.chunk_id} | "
            f"RRF={result.score:.6f}"
        )

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)

    fused_ids = [
        result.chunk_id
        for result in fused_results
    ]

    assert set(fused_ids) == {
        "chunk-A",
        "chunk-B",
        "chunk-C",
        "chunk-D",
    }

    assert len(fused_results) == 4

    # A appears at rank 1 in BM25
    # and rank 2 in dense retrieval.
    #
    # B appears at rank 3 in BM25
    # and rank 1 in dense retrieval.
    #
    # Both should therefore receive contributions
    # from both retrieval systems.

    a_score = next(
        result.score
        for result in fused_results
        if result.chunk_id == "chunk-A"
    )

    b_score = next(
        result.score
        for result in fused_results
        if result.chunk_id == "chunk-B"
    )

    assert a_score > 0
    assert b_score > 0

    print(
        "\nAll RRF assertions passed."
    )


if __name__ == "__main__":
    main()