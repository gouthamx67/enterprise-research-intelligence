from dataclasses import dataclass, field
from typing import Any

from src.rag_engine.retrieval.models import RetrievalResult
from src.rag_engine.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedResult,
)
from src.rag_engine.retrieval.reranking_config import (
    RerankingConfig,
)


@dataclass
class RerankingPipelineResult:
    query: str
    candidate_count: int
    final_count: int
    results: list[RerankedResult] = field(
        default_factory=list
    )


class RetrievalRerankingPipeline:
    def __init__(
        self,
        retriever,
        reranker,
        config: RerankingConfig | None = None,
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        if reranker is None:
            raise ValueError(
                "reranker cannot be None"
            )

        self.retriever = retriever
        self.reranker = reranker
        self.config = (
            config
            if config is not None
            else RerankingConfig()
        )

    def search(
        self,
        query: str,
        candidate_k: int | None = None,
        final_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RerankingPipelineResult:

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        candidate_k = (
            self.config.candidate_k
            if candidate_k is None
            else candidate_k
        )

        final_k = (
            self.config.final_k
            if final_k is None
            else final_k
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

        candidates = self.retriever.search(
            query.strip(),
            top_k=candidate_k,
            filters=filters,
        )

        reranked = self.reranker.rerank(
            query.strip(),
            candidates,
            top_k=final_k,
        )

        return RerankingPipelineResult(
            query=query.strip(),
            candidate_count=len(candidates),
            final_count=len(reranked),
            results=reranked,
        )


def build_reranking_pipeline(
    retriever,
    reranker: CrossEncoderReranker,
    config: RerankingConfig | None = None,
):
    return RetrievalRerankingPipeline(
        retriever=retriever,
        reranker=reranker,
        config=config,
    )