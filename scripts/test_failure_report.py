from src.rag_engine.failures.models import (
    FailureAnalysisResult,
    FailureFinding,
    FailureType,
)

from src.rag_engine.failures.report import (
    build_failure_report,
)


def main():
    result = FailureAnalysisResult(
        query="product strategy",
        findings=[
            FailureFinding(
                failure_type=FailureType.HALLUCINATION,
                severity="high",
                message="Unsupported answer.",
            ),
            FailureFinding(
                failure_type=FailureType.DUPLICATE_EVIDENCE,
                severity="low",
                message="Duplicate chunks.",
            ),
        ],
    )

    report = build_failure_report(result)

    assert report.status == "failed"
    assert report.failure_count == 2
    assert report.high_severity_count == 1
    assert report.low_severity_count == 1
    assert report.medium_severity_count == 0

    assert len(report.failures) == 2

    assert report.failures[0]["type"] == (
        "hallucination"
    )

    print("Failure report test passed.")
    print()
    print(f"Status: {report.status}")
    print(f"Failures: {report.failure_count}")
    print(f"High: {report.high_severity_count}")
    print(f"Medium: {report.medium_severity_count}")
    print(f"Low: {report.low_severity_count}")


if __name__ == "__main__":
    main()