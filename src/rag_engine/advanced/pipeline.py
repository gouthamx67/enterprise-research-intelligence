from dataclasses import dataclass, field

from src.rag_engine.retrieval.models import RetrievalResult

from src.rag_engine.advanced.agentic import (
    AgentPlan,
    AgenticPlanner,
)

from src.rag_engine.advanced.adaptive import (
    AdaptiveDecision,
    AdaptiveRouter,
)

from src.rag_engine.advanced.corrective import (
    CorrectiveResult,
    CorrectiveRetriever,
)

from src.rag_engine.advanced.self_rag import (
    SelfRAGEvaluation,
    SelfRAGCritic,
)


@dataclass
class AdvancedRAGResult:
    query: str
    plan: AgentPlan
    routing: AdaptiveDecision
    retrieval_results: list[RetrievalResult]
    correction: CorrectiveResult
    self_rag: SelfRAGEvaluation | None = None
    metadata: dict = field(default_factory=dict)


class AdvancedRAGPipeline:
    """
    Orchestrates the advanced RAG control flow.

    Retrieval itself is injected so the pipeline can use
    the hybrid retriever already built earlier.
    """

    def __init__(
        self,
        retriever,
        planner: AgenticPlanner | None = None,
        router: AdaptiveRouter | None = None,
        corrective: CorrectiveRetriever | None = None,
        critic: SelfRAGCritic | None = None,
    ):
        if retriever is None:
            raise ValueError("retriever cannot be None")

        self.retriever = retriever
        self.planner = planner or AgenticPlanner()
        self.router = router or AdaptiveRouter()
        self.corrective = (
            corrective
            or CorrectiveRetriever()
        )
        self.critic = critic or SelfRAGCritic()

    def retrieve_step(
        self,
        query: str,
        strategy: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """
        Execute one retrieval step.

        The current retriever is expected to expose search().
        Strategy selection is retained as metadata because the
        actual retrieval implementation is injected.
        """

        results = self.retriever.search(
            query,
            top_k=top_k,
        )

        return results

    def run(
        self,
        query: str,
        top_k: int = 5,
        answer: str | None = None,
    ) -> AdvancedRAGResult:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be positive")

        plan = self.planner.plan(query)
        routing = self.router.route(query)

        all_results: dict[str, RetrievalResult] = {}

        for step in plan.steps:
            results = self.retrieve_step(
                query=step.query,
                strategy=step.strategy,
                top_k=top_k,
            )

            for result in results:
                existing = all_results.get(
                    result.chunk_id
                )

                if (
                    existing is None
                    or result.score > existing.score
                ):
                    all_results[result.chunk_id] = result

        retrieval_results = sorted(
            all_results.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        correction = self.corrective.evaluate(
            query,
            retrieval_results,
        )

        self_rag = None

        if answer is not None:
            evidence = [
                result.text
                for result in correction.accepted_results
            ]

            self_rag = self.critic.evaluate(
                answer,
                evidence,
            )

        return AdvancedRAGResult(
            query=query.strip(),
            plan=plan,
            routing=routing,
            retrieval_results=retrieval_results,
            correction=correction,
            self_rag=self_rag,
            metadata={
                "planner_steps": len(plan.steps),
                "retrieval_count": len(retrieval_results),
                "accepted_count": len(
                    correction.accepted_results
                ),
            },
        )