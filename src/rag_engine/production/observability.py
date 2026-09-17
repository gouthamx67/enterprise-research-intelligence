from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


@dataclass
class PipelineMetrics:
    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    cache_hits: int = 0
    cache_misses: int = 0

    total_latency_ms: float = 0.0

    retrieval_count: int = 0
    context_count: int = 0
    citation_count: int = 0

    failure_findings: int = 0

    @property
    def average_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0

        return (
            self.total_latency_ms
            / self.request_count
        )


@dataclass
class TraceEvent:
    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class PipelineTracer:
    """
    Lightweight tracing implementation.

    Each request can record named operations such as:

        planning
        retrieval
        reranking
        context_construction
        generation
    """

    def __init__(self):
        self.events: list[TraceEvent] = []

    def record(
        self,
        name: str,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                name=name,
                duration_ms=duration_ms,
                metadata=dict(
                    metadata or {}
                ),
            )
        )

    def measure(self, name: str):
        return _TraceTimer(
            tracer=self,
            name=name,
        )

    def clear(self) -> None:
        self.events.clear()


class _TraceTimer:
    def __init__(
        self,
        tracer: PipelineTracer,
        name: str,
    ):
        self.tracer = tracer
        self.name = name
        self.start = 0.0

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        duration_ms = (
            perf_counter() - self.start
        ) * 1000

        self.tracer.record(
            self.name,
            duration_ms,
        )


class Observability:
    def __init__(self):
        self.metrics = PipelineMetrics()
        self.tracer = PipelineTracer()

    def start_request(self) -> float:
        self.metrics.request_count += 1
        return perf_counter()

    def finish_request(
        self,
        start_time: float,
        success: bool,
    ) -> None:
        elapsed_ms = (
            perf_counter() - start_time
        ) * 1000

        self.metrics.total_latency_ms += (
            elapsed_ms
        )

        if success:
            self.metrics.success_count += 1
        else:
            self.metrics.failure_count += 1

    def record_cache_hit(self) -> None:
        self.metrics.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.metrics.cache_misses += 1

    def record_response(
        self,
        retrieval_count: int,
        context_count: int,
        citation_count: int,
        failure_count: int,
    ) -> None:
        self.metrics.retrieval_count += (
            retrieval_count
        )

        self.metrics.context_count += (
            context_count
        )

        self.metrics.citation_count += (
            citation_count
        )

        self.metrics.failure_findings += (
            failure_count
        )