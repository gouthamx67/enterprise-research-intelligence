from time import sleep

from src.rag_engine.production.observability import (
    Observability,
)


def main():
    observability = Observability()

    start = observability.start_request()

    with observability.tracer.measure(
        "retrieval"
    ):
        sleep(0.001)

    observability.record_response(
        retrieval_count=5,
        context_count=3,
        citation_count=3,
        failure_count=0,
    )

    observability.finish_request(
        start_time=start,
        success=True,
    )

    metrics = observability.metrics

    assert metrics.request_count == 1
    assert metrics.success_count == 1
    assert metrics.failure_count == 0
    assert metrics.retrieval_count == 5
    assert metrics.context_count == 3
    assert metrics.citation_count == 3
    assert metrics.failure_findings == 0
    assert metrics.average_latency_ms >= 0

    assert len(
        observability.tracer.events
    ) == 1

    assert (
        observability.tracer.events[0].name
        == "retrieval"
    )

    print(
        "Production observability test passed."
    )


if __name__ == "__main__":
    main()