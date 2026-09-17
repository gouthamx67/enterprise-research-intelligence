from src.rag_engine.failures.models import (
    FailureAnalysisResult,
)
from src.rag_engine.pipeline.models import (
    GeneratedAnswer,
    RAGResponse,
)
from src.rag_engine.production.cache import (
    QueryCache,
)
from src.rag_engine.production.config import (
    ProductionConfig,
)
from src.rag_engine.production.observability import (
    Observability,
)
from src.rag_engine.production.service import (
    ProductionRAGService,
    ProductionRequest,
)


class FakePipeline:
    def __init__(self):
        self.calls = 0

    def run(
        self,
        query,
        candidate_k,
        final_k,
    ):
        self.calls += 1

        return RAGResponse(
            query=query,
            plan=None,
            routing=None,
            retrieved_results=[],
            generated_answer=GeneratedAnswer(
                answer=(
                    "Product strategy changed."
                ),
                citation_ids=["C1"],
            ),
            failure_analysis=FailureAnalysisResult(
                query=query,
                findings=[],
            ),
            confidence=0.95,
            metadata={
                "retrieved_count": 5,
                "context_count": 1,
            },
        )


def main():
    pipeline = FakePipeline()

    config = ProductionConfig(
        candidate_k=10,
        final_k=5,
        cache_enabled=True,
        cache_ttl_seconds=60,
        cache_max_entries=10,
        retry_attempts=0,
    )

    observability = Observability()

    service = ProductionRAGService(
        pipeline=pipeline,
        config=config,
        cache=QueryCache(
            ttl_seconds=60,
            max_entries=10,
        ),
        observability=observability,
    )

    request = ProductionRequest(
        query="What changed in product strategy?"
    )

    first = service.run(request)

    assert first.success is True
    assert pipeline.calls == 1

    second = service.run(request)

    assert second.success is True

    # Second request should come from cache.
    assert pipeline.calls == 1

    metrics = observability.metrics

    assert metrics.request_count == 1
    assert metrics.success_count == 1
    assert metrics.cache_misses == 1
    assert metrics.cache_hits == 1

    print(
        "Production service test passed."
    )

    print()
    print(
        f"Pipeline calls:     {pipeline.calls}"
    )
    print(
        f"Cache hits:         {metrics.cache_hits}"
    )
    print(
        f"Cache misses:       {metrics.cache_misses}"
    )
    print(
        f"Average latency:    "
        f"{metrics.average_latency_ms:.2f} ms"
    )


if __name__ == "__main__":
    main()