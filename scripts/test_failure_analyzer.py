from datetime import date

from src.rag_engine.failures.analyzer import (
    RAGFailureAnalyzer,
)

from src.rag_engine.failures.models import (
    FailureType,
)

from src.rag_engine.retrieval.models import (
    RetrievalResult,
)


def main():
    analyzer = RAGFailureAnalyzer()

    results = [
        RetrievalResult(
            score=0.90,
            rank=1,
            chunk_id="C1",
            text=(
                "The product strategy changed toward "
                "enterprise customers."
            ),
            metadata={
                "document_date": "2026-01-01",
            },
        ),
        RetrievalResult(
            score=0.85,
            rank=2,
            chunk_id="C2",
            text=(
                "AI capabilities became central "
                "to the product roadmap."
            ),
            metadata={
                "document_date": "2026-02-01",
            },
        ),
    ]

    result = analyzer.analyze(
        query="product strategy enterprise",
        results=results,
        answer=(
            "The product strategy changed toward "
            "enterprise customers and AI became "
            "central to the roadmap."
        ),
        citations=["C1", "C2"],
        valid_citation_ids={"C1", "C2"},
        current_date=date(2026, 3, 1),
    )

    assert result.query == (
        "product strategy enterprise"
    )

    assert result.has_failures is False

    # ---------------------------------------------------------
    # Failure case
    # ---------------------------------------------------------

    bad_result = analyzer.analyze(
        query="product strategy enterprise",
        results=[],
        answer="The company became a space company.",
        citations=["C99"],
        valid_citation_ids={"C1"},
    )

    assert bad_result.has_failures is True

    assert bad_result.contains(
        FailureType.RETRIEVAL_FAILURE
    )

    assert bad_result.contains(
        FailureType.INSUFFICIENT_EVIDENCE
    )

    assert bad_result.contains(
        FailureType.HALLUCINATION
    )

    assert bad_result.contains(
        FailureType.CITATION_ERROR
    )

    print("Failure analyzer test passed.")
    print(
        f"Detected failures: "
        f"{bad_result.failure_count}"
    )


if __name__ == "__main__":
    main()