from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureType(str, Enum):
    RETRIEVAL_FAILURE = "retrieval_failure"
    POOR_CHUNKING = "poor_chunking"
    MISSING_CONTEXT = "missing_context"
    WRONG_CONTEXT = "wrong_context"
    CONTEXT_OVERLOAD = "context_overload"
    HALLUCINATION = "hallucination"
    CITATION_ERROR = "citation_error"
    STALE_INFORMATION = "stale_information"
    DUPLICATE_EVIDENCE = "duplicate_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


@dataclass
class FailureFinding:
    failure_type: FailureType
    severity: str
    message: str
    evidence: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureAnalysisResult:
    query: str
    findings: list[FailureFinding] = field(default_factory=list)

    @property
    def has_failures(self) -> bool:
        return len(self.findings) > 0

    @property
    def failure_count(self) -> int:
        return len(self.findings)

    def contains(
        self,
        failure_type: FailureType,
    ) -> bool:
        return any(
            finding.failure_type == failure_type
            for finding in self.findings
        )