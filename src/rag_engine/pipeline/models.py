from dataclasses import dataclass, field
from typing import Any

from src.rag_engine.advanced.adaptive import AdaptiveDecision
from src.rag_engine.advanced.agentic import AgentPlan
from src.rag_engine.failures.models import FailureAnalysisResult
from src.rag_engine.retrieval.reranker import RerankedResult


@dataclass
class GeneratedAnswer:
    answer: str
    citation_ids: list[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    query: str

    plan: AgentPlan
    routing: AdaptiveDecision

    retrieved_results: list[RerankedResult]

    generated_answer: GeneratedAnswer | None

    failure_analysis: FailureAnalysisResult

    confidence: float

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def success(self) -> bool:
        return (
            self.generated_answer is not None
            and not self.failure_analysis.has_failures
        )