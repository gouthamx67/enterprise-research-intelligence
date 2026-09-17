from src.rag_engine.failures.models import (
    FailureAnalysisResult,
    FailureFinding,
    FailureType,
)


def main():
    finding = FailureFinding(
        failure_type=FailureType.RETRIEVAL_FAILURE,
        severity="high",
        message="No results.",
    )

    result = FailureAnalysisResult(
        query="test query",
        findings=[finding],
    )

    assert result.has_failures is True
    assert result.failure_count == 1
    assert result.contains(
        FailureType.RETRIEVAL_FAILURE
    )

    empty = FailureAnalysisResult(
        query="test query",
    )

    assert empty.has_failures is False
    assert empty.failure_count == 0

    print("Failure model test passed.")


if __name__ == "__main__":
    main()