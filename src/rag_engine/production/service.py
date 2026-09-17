from dataclasses import dataclass
import hashlib

from src.rag_engine.pipeline.models import RAGResponse
from src.rag_engine.production.cache import QueryCache
from src.rag_engine.production.config import (
    ProductionConfig,
)
from src.rag_engine.production.observability import (
    Observability,
)
from src.rag_engine.production.reliability import (
    ReliablePipeline,
    RetryConfig,
)


@dataclass(frozen=True)
class ProductionRequest:
    query: str
    candidate_k: int | None = None
    final_k: int | None = None


class ProductionRAGService:
    """
    Production-facing wrapper around the end-to-end RAG pipeline.

    Responsibilities:

        - input validation
        - configuration
        - caching
        - retries
        - observability
        - pipeline execution
    """

    def __init__(
        self,
        pipeline,
        config: ProductionConfig | None = None,
        cache: QueryCache | None = None,
        observability: Observability | None = None,
    ):
        if pipeline is None:
            raise ValueError(
                "pipeline cannot be None"
            )

        self.config = (
            config or ProductionConfig()
        )

        self.cache = cache

        if (
            self.cache is None
            and self.config.cache_enabled
        ):
            self.cache = QueryCache(
                ttl_seconds=(
                    self.config.cache_ttl_seconds
                ),
                max_entries=(
                    self.config.cache_max_entries
                ),
            )

        self.observability = (
            observability
            or Observability()
        )

        self.pipeline = ReliablePipeline(
            pipeline,
            RetryConfig(
                attempts=(
                    self.config.retry_attempts
                ),
            ),
        )

    def _validate_query(
        self,
        query: str,
    ) -> str:
        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        query = query.strip()

        if len(query) > self.config.max_query_length:
            raise ValueError(
                "query exceeds maximum allowed length"
            )

        return query

    def _cache_key(
        self,
        request: ProductionRequest,
    ) -> str:
        raw = (
            f"{request.query}|"
            f"{request.candidate_k}|"
            f"{request.final_k}"
        )

        return hashlib.sha256(
            raw.encode("utf-8")
        ).hexdigest()

    def run(
        self,
        request: ProductionRequest,
    ) -> RAGResponse:
        query = self._validate_query(
            request.query
        )

        candidate_k = (
            request.candidate_k
            if request.candidate_k is not None
            else self.config.candidate_k
        )

        final_k = (
            request.final_k
            if request.final_k is not None
            else self.config.final_k
        )

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        if final_k <= 0:
            raise ValueError(
                "final_k must be greater than 0"
            )

        if final_k > candidate_k:
            raise ValueError(
                "final_k cannot be greater than candidate_k"
            )

        normalized_request = ProductionRequest(
            query=query,
            candidate_k=candidate_k,
            final_k=final_k,
        )

        cache_key = self._cache_key(
            normalized_request
        )

        if self.cache is not None:
            cached = self.cache.get(
                cache_key
            )

            if cached is not None:
                self.observability.record_cache_hit()
                return cached

            self.observability.record_cache_miss()

        start_time = (
            self.observability.start_request()
        )

        try:
            response = self.pipeline.run(
                query=query,
                candidate_k=candidate_k,
                final_k=final_k,
            )

            self.observability.record_response(
                retrieval_count=int(
                    response.metadata.get(
                        "retrieved_count",
                        0,
                    )
                ),
                context_count=int(
                    response.metadata.get(
                        "context_count",
                        0,
                    )
                ),
                citation_count=(
                    len(
                        response.generated_answer.citation_ids
                    )
                    if response.generated_answer
                    is not None
                    else 0
                ),
                failure_count=(
                    response.failure_analysis.failure_count
                ),
            )

            self.observability.finish_request(
                start_time=start_time,
                success=response.success,
            )

            if (
                self.cache is not None
                and response.success
            ):
                self.cache.set(
                    cache_key,
                    response,
                )

            return response

        except Exception:
            self.observability.finish_request(
                start_time=start_time,
                success=False,
            )
            raise