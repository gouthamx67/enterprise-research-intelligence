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


class DemoPipeline:
    """
    Demonstration pipeline.

    In the real application this object is replaced by the
    EndToEndRAGPipeline configured with the project's retrieval,
    reranking, context, and generation components.
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
                    "The product strategy shifted toward "
                    "enterprise customers, broader platform "
                    "capabilities, and expanded monetization."
                ),
                citation_ids=[
                    "C1",
                    "C2",
                    "C3",
                ],
            ),
            failure_analysis=FailureAnalysisResult(
                query=query,
                findings=[],
            ),
            confidence=0.94,
            metadata={
                "retrieved_count": 20,
                "context_count": 5,
                "routing_strategy": "hybrid",
                "citation_valid": True,
            },
        )


def main():
    config = ProductionConfig(
        candidate_k=20,
        final_k=5,
        cache_enabled=False,
    )

    system = EnterpriseResearchIntelligence(
        pipeline=DemoPipeline(),
        config=config,
    )

    question = (
        "What changed in this company's product "
        "strategy over the last 12 months?"
    )

    result = system.ask(question)

    print("=" * 70)
    print("ENTERPRISE RESEARCH INTELLIGENCE")
    print("=" * 70)

    print()
    print("QUESTION")
    print(question)

    print()
    print("ANSWER")
    print(result.answer)

    print()
    print("CITATIONS")
    for citation in result.citations:
        print(f"- {citation}")

    print()
    print("CONFIDENCE")
    print(f"{result.confidence:.3f}")

    print()
    print("FAILURES")
    print(result.failure_count)

    print()
    print("METADATA")

    for key, value in result.metadata.items():
        print(f"- {key}: {value}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()