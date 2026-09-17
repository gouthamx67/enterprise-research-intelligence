from dataclasses import dataclass
from typing import Any

from src.rag_engine.pipeline.models import RAGResponse
from src.rag_engine.production.config import (
    ProductionConfig,
    load_production_config,
)
from src.rag_engine.production.service import (
    ProductionRAGService,
    ProductionRequest,
)


@dataclass
class ResearchAnswer:
    """
    User-facing representation of a research answer.
    """

    answer: str
    citations: list[str]
    confidence: float
    success: bool
    failure_count: int
    metadata: dict[str, Any]


class EnterpriseResearchIntelligence:
    """
    Top-level application facade.

    The rest of the project contains the individual RAG components.
    This class provides one stable interface for an application layer.
    """

    def __init__(
        self,
        pipeline,
        config: ProductionConfig | None = None,
    ):
        self.config = (
            config
            or load_production_config()
        )

        self.service = ProductionRAGService(
            pipeline=pipeline,
            config=self.config,
        )

    def ask(
        self,
        question: str,
        candidate_k: int | None = None,
        final_k: int | None = None,
    ) -> ResearchAnswer:
        response: RAGResponse = self.service.run(
            ProductionRequest(
                query=question,
                candidate_k=candidate_k,
                final_k=final_k,
            )
        )

        if response.generated_answer is None:
            answer = (
                "I could not construct an answer "
                "from the available evidence."
            )

            citations: list[str] = []

        else:
            answer = response.generated_answer.answer
            citations = list(
                response.generated_answer.citation_ids
            )

        return ResearchAnswer(
            answer=answer,
            citations=citations,
            confidence=response.confidence,
            success=response.success,
            failure_count=(
                response.failure_analysis.failure_count
            ),
            metadata=dict(response.metadata),
        )