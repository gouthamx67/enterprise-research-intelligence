from dataclasses import dataclass

from src.rag_engine.failures.models import (
    FailureAnalysisResult,
    FailureType,
)


@dataclass
class FailureReport:
    query: str
    status: str
    failure_count: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    failures: list[dict]


def build_failure_report(
    result: FailureAnalysisResult,
) -> FailureReport:
    high = sum(
        finding.severity == "high"
        for finding in result.findings
    )

    medium = sum(
        finding.severity == "medium"
        for finding in result.findings
    )

    low = sum(
        finding.severity == "low"
        for finding in result.findings
    )

    failures = [
        {
            "type": finding.failure_type.value,
            "severity": finding.severity,
            "message": finding.message,
            "evidence": list(finding.evidence),
            "metadata": dict(finding.metadata),
        }
        for finding in result.findings
    ]

    status = (
        "failed"
        if result.has_failures
        else "healthy"
    )

    return FailureReport(
        query=result.query,
        status=status,
        failure_count=result.failure_count,
        high_severity_count=high,
        medium_severity_count=medium,
        low_severity_count=low,
        failures=failures,
    )


def print_failure_report(
    report: FailureReport,
) -> None:
    print("RAG Failure Report")
    print("==================")
    print(f"Status: {report.status}")
    print(f"Failures: {report.failure_count}")
    print(f"High: {report.high_severity_count}")
    print(f"Medium: {report.medium_severity_count}")
    print(f"Low: {report.low_severity_count}")

    for failure in report.failures:
        print()
        print(
            f"[{failure['severity'].upper()}] "
            f"{failure['type']}"
        )
        print(failure["message"])

        if failure["evidence"]:
            print(
                "Evidence: "
                + ", ".join(failure["evidence"])
            )