from dataclasses import dataclass

from src.rag_engine.retrieval.context_deduplication import (
    ContextDeduplicator,
    ContextDeduplicationResult,
    DeduplicatedContext,
    deduplicate_context,
    normalize_for_deduplication,
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
            chunk_id="chunk-001",
            text=(
                "The company launched its "
                "AI platform in 2025."
            ),
            score=0.95,
            rank=1,
            metadata={
                "source": "filing.pdf",
                "page": 10,
            },
        ),
        FakeContext(
            chunk_id="chunk-002",
            text=(
                "The company launched its "
                "AI platform in 2025."
            ),
            score=0.90,
            rank=2,
            metadata={
                "source": "transcript.txt",
                "page": 4,
            },
        ),
        FakeContext(
            chunk_id="chunk-003",
            text=(
                "  The   Company launched its "
                "AI platform in 2025.  "
            ),
            score=0.85,
            rank=3,
            metadata={
                "source": "website.html",
            },
        ),
        FakeContext(
            chunk_id="chunk-004",
            text=(
                "The company changed its "
                "pricing model."
            ),
            score=0.80,
            rank=4,
            metadata={
                "source": "filing.pdf",
                "page": 12,
            },
        ),
    ]


def main():

    contexts = build_contexts()

    # ---------------------------------------------
    # Normalization
    # ---------------------------------------------

    normalized = normalize_for_deduplication(
        "  The   COMPANY launched its AI platform.  "
    )

    assert (
        normalized
        == "the company launched its ai platform."
    )

    # ---------------------------------------------
    # Basic deduplication
    # ---------------------------------------------

    deduplicator = ContextDeduplicator()

    result = deduplicator.deduplicate(
        contexts
    )

    assert isinstance(
        result,
        ContextDeduplicationResult,
    )

    assert result.original_count == 4

    assert result.deduplicated_count == 2

    assert result.removed_count == 2

    # ---------------------------------------------
    # First occurrence survives
    # ---------------------------------------------

    assert [
        item.chunk_id
        for item in result.selected
    ] == [
        "chunk-001",
        "chunk-004",
    ]

    # ---------------------------------------------
    # Original text is preserved
    # ---------------------------------------------

    assert (
        result.selected[0].text
        == contexts[0].text.strip()
    )

    # ---------------------------------------------
    # Highest-ranked occurrence survives
    # ---------------------------------------------

    assert (
        result.selected[0].score
        == 0.95
    )

    assert (
        result.selected[0].rank
        == 1
    )

    # ---------------------------------------------
    # Metadata is preserved
    # ---------------------------------------------

    assert (
        result.selected[0].metadata["source"]
        == "filing.pdf"
    )

    assert (
        result.selected[0].metadata["page"]
        == 10
    )

    # ---------------------------------------------
    # Remaining unique context preserved
    # ---------------------------------------------

    assert (
        result.selected[1].chunk_id
        == "chunk-004"
    )

    assert (
        result.selected[1].text
        == "The company changed its pricing model."
    )

    # ---------------------------------------------
    # Convenience function
    # ---------------------------------------------

    result = deduplicate_context(
        contexts
    )

    assert result.deduplicated_count == 2

    assert isinstance(
        result.selected[0],
        DeduplicatedContext,
    )

    # ---------------------------------------------
    # Empty input
    # ---------------------------------------------

    result = deduplicator.deduplicate([])

    assert result.original_count == 0
    assert result.deduplicated_count == 0
    assert result.removed_count == 0
    assert result.selected == []

    # ---------------------------------------------
    # None input
    # ---------------------------------------------

    try:
        deduplicator.deduplicate(None)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid normalization input
    # ---------------------------------------------

    try:
        normalize_for_deduplication(None)

        raise AssertionError(
            "Expected TypeError"
        )

    except TypeError:
        pass

    print(
        "All context deduplication "
        "assertions passed."
    )


if __name__ == "__main__":
    main()