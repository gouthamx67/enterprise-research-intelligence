from datetime import date

from src.rag_engine.failures.detectors import (
    detect_citation_error,
    detect_context_overload,
    detect_duplicate_evidence,
    detect_hallucination,
    detect_insufficient_evidence,
    detect_missing_context,
    detect_poor_chunking,
    detect_retrieval_failure,
    detect_stale_information,
    detect_wrong_context,
)

from src.rag_engine.failures.models import FailureType

from src.rag_engine.retrieval.models import RetrievalResult


def main():
    # ---------------------------------------------------------
    # Retrieval failure
    # ---------------------------------------------------------

    finding = detect_retrieval_failure([])

    assert finding is not None
    assert finding.failure_type == (
        FailureType.RETRIEVAL_FAILURE
    )

    # ---------------------------------------------------------
    # Insufficient evidence
    # ---------------------------------------------------------

    results = [
        RetrievalResult(
            score=0.30,
            rank=1,
            chunk_id="low",
            text="Weak evidence.",
        )
    ]

    finding = detect_insufficient_evidence(
        results,
        minimum_score=0.50,
        minimum_results=2,
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.INSUFFICIENT_EVIDENCE
    )

    # ---------------------------------------------------------
    # Context overload
    # ---------------------------------------------------------

    results = [
        RetrievalResult(
            score=0.9,
            rank=1,
            chunk_id=str(i),
            text="x" * 100,
        )
        for i in range(11)
    ]

    finding = detect_context_overload(
        results,
        max_chunks=10,
        max_characters=12000,
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.CONTEXT_OVERLOAD
    )

    # ---------------------------------------------------------
    # Duplicate evidence
    # ---------------------------------------------------------

    duplicate_results = [
        RetrievalResult(
            score=0.9,
            rank=1,
            chunk_id="a",
            text="Product strategy changed.",
        ),
        RetrievalResult(
            score=0.8,
            rank=2,
            chunk_id="b",
            text="  PRODUCT STRATEGY changed. ",
        ),
    ]

    finding = detect_duplicate_evidence(
        duplicate_results
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.DUPLICATE_EVIDENCE
    )

    # ---------------------------------------------------------
    # Wrong context
    # ---------------------------------------------------------

    finding = detect_wrong_context(
        "product strategy enterprise",
        [
            RetrievalResult(
                score=0.9,
                rank=1,
                chunk_id="wrong",
                text="Weather forecast tomorrow.",
            )
        ],
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.WRONG_CONTEXT
    )

    # ---------------------------------------------------------
    # Missing context
    # ---------------------------------------------------------

    finding = detect_missing_context(
        "product strategy enterprise",
        [
            RetrievalResult(
                score=0.9,
                rank=1,
                chunk_id="x",
                text="The office opened in London.",
            )
        ],
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.MISSING_CONTEXT
    )

    # ---------------------------------------------------------
    # Poor chunking
    # ---------------------------------------------------------

    finding = detect_poor_chunking(
        [
            RetrievalResult(
                score=0.9,
                rank=1,
                chunk_id="huge",
                text="x" * 6000,
            )
        ],
        max_chunk_characters=5000,
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.POOR_CHUNKING
    )

    # ---------------------------------------------------------
    # Stale information
    # ---------------------------------------------------------

    finding = detect_stale_information(
        [
            RetrievalResult(
                score=0.9,
                rank=1,
                chunk_id="old",
                text="Old strategy.",
                metadata={
                    "document_date": "2024-01-01"
                },
            )
        ],
        current_date=date(2026, 1, 1),
        max_age_days=365,
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.STALE_INFORMATION
    )

    # ---------------------------------------------------------
    # Hallucination
    # ---------------------------------------------------------

    finding = detect_hallucination(
        answer="The company became a space company.",
        evidence=[
            "The product strategy changed toward enterprise customers."
        ],
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.HALLUCINATION
    )

    # ---------------------------------------------------------
    # Citation error
    # ---------------------------------------------------------

    finding = detect_citation_error(
        citations=["C1", "C99"],
        valid_citation_ids={"C1", "C2"},
    )

    assert finding is not None
    assert finding.failure_type == (
        FailureType.CITATION_ERROR
    )

    print("Failure detector test passed.")


if __name__ == "__main__":
    main()