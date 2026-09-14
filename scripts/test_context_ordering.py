from dataclasses import dataclass

from src.rag_engine.retrieval.context_ordering import (
    ContextOrderer,
    ContextOrderingResult,
    OrderedContext,
    order_context,
)


@dataclass
class FakeContext:
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict


def build_contexts():
    return [
        FakeContext(
            chunk_id="chunk-003",
            text="Evidence about pricing.",
            score=0.70,
            rank=3,
            metadata={
                "source": "website.html",
                "document_date": "2025-06-01",
            },
        ),
        FakeContext(
            chunk_id="chunk-001",
            text="Evidence about product strategy.",
            score=0.95,
            rank=1,
            metadata={
                "source": "filing.pdf",
                "document_date": "2025-01-01",
            },
        ),
        FakeContext(
            chunk_id="chunk-002",
            text="Evidence about product roadmap.",
            score=0.85,
            rank=2,
            metadata={
                "source": "filing.pdf",
                "document_date": "2025-03-01",
            },
        ),
        FakeContext(
            chunk_id="chunk-004",
            text="Evidence about customers.",
            score=0.80,
            rank=4,
            metadata={
                "source": "website.html",
                "document_date": "2025-02-01",
            },
        ),
    ]


def main():

    contexts = build_contexts()

    orderer = ContextOrderer()

    # ---------------------------------------------
    # rank_order
    # ---------------------------------------------

    result = orderer.order(
        contexts,
        strategy="rank_order",
    )

    assert isinstance(
        result,
        ContextOrderingResult,
    )

    assert [
        item.chunk_id
        for item in result.contexts
    ] == [
        "chunk-001",
        "chunk-002",
        "chunk-003",
        "chunk-004",
    ]

    # ---------------------------------------------
    # score_desc
    # ---------------------------------------------

    result = orderer.order(
        contexts,
        strategy="score_desc",
    )

    assert [
        item.chunk_id
        for item in result.contexts
    ] == [
        "chunk-001",
        "chunk-002",
        "chunk-004",
        "chunk-003",
    ]

    # ---------------------------------------------
    # source_grouped
    # ---------------------------------------------

    result = orderer.order(
        contexts,
        strategy="source_grouped",
    )

    assert [
        item.chunk_id
        for item in result.contexts
    ] == [
        "chunk-001",
        "chunk-002",
        "chunk-004",
        "chunk-003",
    ]

    # ---------------------------------------------
    # chronological
    # ---------------------------------------------

    result = orderer.order(
        contexts,
        strategy="chronological",
    )

    assert [
        item.chunk_id
        for item in result.contexts
    ] == [
        "chunk-001",
        "chunk-004",
        "chunk-002",
        "chunk-003",
    ]

    # ---------------------------------------------
    # Evidence itself is unchanged
    # ---------------------------------------------

    assert (
        result.contexts[0].text
        == "Evidence about product strategy."
    )

    # ---------------------------------------------
    # Metadata preserved
    # ---------------------------------------------

    assert (
        result.contexts[0].metadata["source"]
        == "filing.pdf"
    )

    assert (
        result.contexts[0].metadata[
            "document_date"
        ]
        == "2025-01-01"
    )

    # ---------------------------------------------
    # Score and rank preserved
    # ---------------------------------------------

    assert result.contexts[0].score == 0.95
    assert result.contexts[0].rank == 1

    # ---------------------------------------------
    # Counts
    # ---------------------------------------------

    assert result.original_count == 4
    assert result.ordered_count == 4

    # ---------------------------------------------
    # Convenience function
    # ---------------------------------------------

    result = order_context(
        contexts,
        strategy="score_desc",
    )

    assert isinstance(
        result.contexts[0],
        OrderedContext,
    )

    assert (
        result.contexts[0].chunk_id
        == "chunk-001"
    )

    # ---------------------------------------------
    # Empty input
    # ---------------------------------------------

    result = orderer.order([])

    assert result.original_count == 0
    assert result.ordered_count == 0
    assert result.contexts == []

    # ---------------------------------------------
    # None input
    # ---------------------------------------------

    try:
        orderer.order(None)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid strategy
    # ---------------------------------------------

    try:
        orderer.order(
            contexts,
            strategy="invalid",
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print(
        "All context ordering "
        "assertions passed."
    )


if __name__ == "__main__":
    main()