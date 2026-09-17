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

from src.rag_engine.failures.models import (
    FailureAnalysisResult,
    FailureFinding,
)
from src.rag_engine.retrieval.models import RetrievalResult


class RAGFailureAnalyzer:
    """
    Runs the complete RAG failure-detection suite.
    """

    def analyze(
        self,
        query: str,
        results: list[RetrievalResult],
        answer: str | None = None,
        citations: list[str] | None = None,
        valid_citation_ids: set[str] | None = None,
        current_date: date | None = None,
    ) -> FailureAnalysisResult:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        findings: list[FailureFinding] = []

        detectors = [
            detect_retrieval_failure(results),
            detect_insufficient_evidence(results),
            detect_context_overload(results),
            detect_duplicate_evidence(results),
            detect_wrong_context(query, results),
            detect_missing_context(query, results),
            detect_poor_chunking(results),
        ]

        if current_date is not None:
            detectors.append(
                detect_stale_information(
                    results,
                    current_date,
                )
            )

        if answer is not None:
            detectors.append(
                detect_hallucination(
                    answer,
                    [
                        result.text
                        for result in results
                    ],
                )
            )

        if (
            citations is not None
            and valid_citation_ids is not None
        ):
            detectors.append(
                detect_citation_error(
                    citations,
                    valid_citation_ids,
                )
            )

        for finding in detectors:
            if finding is not None:
                findings.append(finding)

        return FailureAnalysisResult(
            query=query.strip(),
            findings=findings,
        )