from dataclasses import dataclass

from src.rag_engine.retrieval.context_selection import (
    ContextSelectionResult,
    ContextSelector,
    SelectedContext,
    select_context,
)


@dataclass
class FakeCandidate:
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict


def build_candidates():
    return [
        FakeCandidate(
            chunk_id="chunk-001",
            text=(
                "The company launched a new "
                "enterprise analytics platform."
            ),
            score=0.95,
            rank=1,
            metadata={
                "source": "filing-2025.pdf",
                "page": 10,
            },
        ),
        FakeCandidate(
            chunk_id="chunk-002",
            text=(
                "The product roadmap expanded "
                "to include AI workflow features."
            ),
            score=0.90,
            rank=2,
            metadata={
                "source": "transcript-2025.txt",
                "page": 4,
            },
        ),
        FakeCandidate(
            chunk_id="chunk-003",
            text=(
                "The company introduced a new "
                "usage-based pricing model."
            ),
            score=0.85,
            rank=3,
            metadata={
                "source": "website.html",
            },
        ),
        FakeCandidate(
            chunk_id="chunk-004",
            text=(
                "This is general company information."
            ),
            score=0.50,
            rank=4,
            metadata={
                "source": "company.html",
            },
        ),
    ]


def main():
    candidates = build_candidates()

    # ---------------------------------------------
    # Basic selection
    # ---------------------------------------------

    selector = ContextSelector(
        max_chunks=2
    )

    result = selector.select(
        query="What changed in product strategy?",
        candidates=candidates,
    )

    assert isinstance(
        result,
        ContextSelectionResult,
    )

    assert result.candidate_count == 4
    assert result.selected_count == 2

    assert [
        item.chunk_id
        for item in result.selected
    ] == [
        "chunk-001",
        "chunk-002",
    ]

    # ---------------------------------------------
    # Ranking order is preserved
    # ---------------------------------------------

    assert (
        result.selected[0].rank
        == 1
    )

    assert (
        result.selected[1].rank
        == 2
    )

    assert (
        result.selected[0].score
        == 0.95
    )

    # ---------------------------------------------
    # Metadata is preserved
    # ---------------------------------------------

    assert (
        result.selected[0].metadata["source"]
        == "filing-2025.pdf"
    )

    assert (
        result.selected[1].metadata["source"]
        == "transcript-2025.txt"
    )

    # ---------------------------------------------
    # Character count
    # ---------------------------------------------

    expected_characters = sum(
        len(item.text)
        for item in result.selected
    )

    assert (
        result.total_characters
        == expected_characters
    )

    # ---------------------------------------------
    # Character budget
    # ---------------------------------------------

    first_length = len(
        candidates[0].text
    )

    second_length = len(
        candidates[1].text
    )

    selector = ContextSelector(
        max_chunks=10,
        max_characters=first_length,
    )

    result = selector.select(
        query="product strategy",
        candidates=candidates,
    )

    assert result.selected_count == 1

    assert (
        result.selected[0].chunk_id
        == "chunk-001"
    )

    # ---------------------------------------------
    # Empty candidate text is skipped
    # ---------------------------------------------

    candidates_with_empty = [
        FakeCandidate(
            chunk_id="empty",
            text="   ",
            score=1.0,
            rank=1,
            metadata={},
        ),
        FakeCandidate(
            chunk_id="valid",
            text="Useful evidence.",
            score=0.9,
            rank=2,
            metadata={},
        ),
    ]

    selector = ContextSelector(
        max_chunks=5
    )

    result = selector.select(
        query="test",
        candidates=candidates_with_empty,
    )

    assert result.selected_count == 1

    assert (
        result.selected[0].chunk_id
        == "valid"
    )

    # ---------------------------------------------
    # max_chunks validation
    # ---------------------------------------------

    try:
        ContextSelector(max_chunks=0)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # max_characters validation
    # ---------------------------------------------

    try:
        ContextSelector(
            max_chunks=5,
            max_characters=0,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Empty query validation
    # ---------------------------------------------

    try:
        selector.select(
            query="   ",
            candidates=candidates,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # None candidates validation
    # ---------------------------------------------

    try:
        selector.select(
            query="test",
            candidates=None,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Convenience function
    # ---------------------------------------------

    result = select_context(
        query="product strategy",
        candidates=candidates,
        max_chunks=3,
    )

    assert isinstance(
        result.selected[0],
        SelectedContext,
    )

    assert result.selected_count == 3

    print(
        "All context selection "
        "assertions passed."
    )


if __name__ == "__main__":
    main()