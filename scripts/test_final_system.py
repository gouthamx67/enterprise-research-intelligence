from src.rag_engine.final_system import (
    EnterpriseResearchIntelligence,
)
from src.rag_engine.pipeline.models import (
    GeneratedAnswer,
    RAGResponse,
)
from src.rag_engine.production.config import (
    ProductionConfig,
)
from src.rag_engine.failures.models import (
    FailureAnalysisResult,
)


class FakePipeline:
    """
    Deterministic pipeline used only for testing the
    final application facade.
    """

    def run(
        self,
        query,
        candidate_k,
        final_k,
    ):
        return RAGResponse(
            query=query,
            plan=None,
            routing=None,
            retrieved_results=[],
            generated_answer=GeneratedAnswer(
                answer=(
                    "The company expanded its enterprise "
                    "product strategy."
                ),
                citation_ids=[
                    "C1",
                    "C2",
                ],
            ),
            failure_analysis=FailureAnalysisResult(
                query=query,
                findings=[],
            ),
            confidence=0.91,
            metadata={
                "retrieved_count": 10,
                "context_count": 2,
                "routing_strategy": "hybrid",
            },
        )


def main():
    config = ProductionConfig(
        candidate_k=10,
        final_k=5,
        cache_enabled=False,
        retry_attempts=0,
    )

    system = EnterpriseResearchIntelligence(
        pipeline=FakePipeline(),
        config=config,
    )

    result = system.ask(
        "What changed in this company's product strategy?"
    )

    assert result.success is True

    assert (
        "enterprise"
        in result.answer.lower()
    )

    assert result.citations == [
        "C1",
        "C2",
    ]

    assert result.confidence == 0.91

    assert result.failure_count == 0

    print(
        "Final system test passed."
    )

    print()
    print(
        f"Answer:       {result.answer}"
    )
    print(
        f"Citations:    {result.citations}"
    )
    print(
        f"Confidence:   {result.confidence:.3f}"
    )
    print(
        f"Failures:     {result.failure_count}"
    )


if __name__ == "__main__":
    main()