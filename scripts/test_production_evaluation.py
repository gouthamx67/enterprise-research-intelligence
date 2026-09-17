from src.rag_engine.pipeline.models import (
    GeneratedAnswer,
    RAGResponse,
)
from src.rag_engine.production.evaluation import (
    ProductionEvaluator,
    summarize_evaluations,
)
from src.rag_engine.failures.models import (
    FailureAnalysisResult,
)


def main():
    response = RAGResponse(
        query="What changed?",
        plan=None,
        routing=None,
        retrieved_results=[],
        generated_answer=GeneratedAnswer(
            answer="The strategy changed.",
            citation_ids=[
                "C1",
                "C2",
            ],
        ),
        failure_analysis=FailureAnalysisResult(
            query="What changed?",
            findings=[],
        ),
        confidence=0.9,
        metadata={
            "retrieved_count": 5,
            "context_count": 2,
        },
    )

    evaluator = ProductionEvaluator()

    evaluation = evaluator.evaluate(
        response
    )

    assert evaluation.success is True
    assert evaluation.confidence == 0.9
    assert evaluation.retrieved_count == 5
    assert evaluation.context_count == 2
    assert evaluation.citation_count == 2
    assert evaluation.citation_coverage == 1.0

    summary = summarize_evaluations(
        [evaluation]
    )

    assert summary["count"] == 1.0
    assert summary["success_rate"] == 1.0

    print(
        "Production evaluation test passed."
    )


if __name__ == "__main__":
    main()