from dataclasses import dataclass, field

from src.rag_engine.retrieval.models import RetrievalResult


@dataclass
class RetrievalAssessment:
    query: str
    sufficient: bool
    relevant_count: int
    total_count: int
    average_score: float
    reason: str


@dataclass
class CorrectiveResult:
    original_results: list[RetrievalResult]
    accepted_results: list[RetrievalResult]
    assessment: RetrievalAssessment
    should_retry: bool


class CorrectiveRetriever:
    """
    Evaluates retrieved evidence and determines whether
    retrieval should be accepted or corrected.

    This implementation is intentionally deterministic.
    """

    def __init__(
        self,
        minimum_score: float = 0.50,
        minimum_relevant_results: int = 2,
    ):
        if not 0 <= minimum_score <= 1:
            raise ValueError(
                "minimum_score must be between 0 and 1"
            )

        if minimum_relevant_results <= 0:
            raise ValueError(
                "minimum_relevant_results must be positive"
            )

        self.minimum_score = minimum_score
        self.minimum_relevant_results = minimum_relevant_results

    def assess(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> RetrievalAssessment:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if not results:
            return RetrievalAssessment(
                query=query.strip(),
                sufficient=False,
                relevant_count=0,
                total_count=0,
                average_score=0.0,
                reason="No retrieval results were returned.",
            )

        relevant = [
            result
            for result in results
            if result.score >= self.minimum_score
        ]

        average_score = sum(
            result.score for result in results
        ) / len(results)

        sufficient = (
            len(relevant) >= self.minimum_relevant_results
        )

        if sufficient:
            reason = "Retrieved evidence is sufficient."
        else:
            reason = (
                "Retrieved evidence is insufficient; "
                "additional retrieval is required."
            )

        return RetrievalAssessment(
            query=query.strip(),
            sufficient=sufficient,
            relevant_count=len(relevant),
            total_count=len(results),
            average_score=average_score,
            reason=reason,
        )

    def evaluate(
        self,
        query: str,
        results: list[RetrievalResult],
    ) -> CorrectiveResult:
        assessment = self.assess(query, results)

        if assessment.sufficient:
            accepted = list(results)
        else:
            accepted = []

        return CorrectiveResult(
            original_results=list(results),
            accepted_results=accepted,
            assessment=assessment,
            should_retry=not assessment.sufficient,
        )