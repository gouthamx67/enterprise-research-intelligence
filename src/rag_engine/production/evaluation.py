from dataclasses import dataclass

from src.rag_engine.pipeline.models import RAGResponse


@dataclass(frozen=True)
class ProductionEvaluation:
    query: str
    success: bool
    confidence: float
    retrieved_count: int
    context_count: int
    citation_count: int
    failure_count: int

    @property
    def citation_coverage(self) -> float:
        if self.context_count == 0:
            return 0.0

        return min(
            self.citation_count
            / self.context_count,
            1.0,
        )


class ProductionEvaluator:
    """
    Converts a RAGResponse into production-facing quality metrics.
    """

    def evaluate(
        self,
        response: RAGResponse,
    ) -> ProductionEvaluation:
        metadata = response.metadata

        context_count = int(
            metadata.get(
                "context_count",
                0,
            )
        )

        citation_count = 0

        if response.generated_answer is not None:
            citation_count = len(
                response.generated_answer.citation_ids
            )

        return ProductionEvaluation(
            query=response.query,
            success=response.success,
            confidence=response.confidence,
            retrieved_count=int(
                metadata.get(
                    "retrieved_count",
                    0,
                )
            ),
            context_count=context_count,
            citation_count=citation_count,
            failure_count=response.failure_analysis.failure_count,
        )


def summarize_evaluations(
    evaluations: list[ProductionEvaluation],
) -> dict[str, float]:
    if not evaluations:
        return {
            "count": 0,
            "success_rate": 0.0,
            "average_confidence": 0.0,
            "average_citation_coverage": 0.0,
        }

    count = len(evaluations)

    return {
        "count": float(count),
        "success_rate": (
            sum(
                item.success
                for item in evaluations
            )
            / count
        ),
        "average_confidence": (
            sum(
                item.confidence
                for item in evaluations
            )
            / count
        ),
        "average_citation_coverage": (
            sum(
                item.citation_coverage
                for item in evaluations
            )
            / count
        ),
    }