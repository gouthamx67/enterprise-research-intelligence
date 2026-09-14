from typing import Any

from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.models import (
    RetrievalResult,
)
from src.rag_engine.retrieval.sparse import (
    BM25Retriever,
)


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 5,
) -> list[RetrievalResult]:
    """
    Combine multiple ranked retrieval result lists
    using Reciprocal Rank Fusion (RRF).
    """

    if k <= 0:
        raise ValueError(
            "k must be greater than 0"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be greater than 0"
        )

    fused_scores = {}

    result_lookup = {}

    for results in result_lists:
        for result in results:
            chunk_id = result.chunk_id

            fused_scores[chunk_id] = (
                fused_scores.get(
                    chunk_id,
                    0.0,
                )
                + 1.0
                / (k + result.rank)
            )

            result_lookup[
                chunk_id
            ] = result

    ranked_chunks = sorted(
        fused_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    fused_results = []

    for rank, (
        chunk_id,
        fusion_score,
    ) in enumerate(
        ranked_chunks[:top_k],
        start=1,
    ):
        original_result = result_lookup[
            chunk_id
        ]

        fused_results.append(
            RetrievalResult(
                score=fusion_score,
                rank=rank,
                chunk_id=chunk_id,
                text=original_result.text,
                metadata=original_result.metadata.copy(),
            )
        )

    return fused_results


class HybridRetriever:
    """
    Hybrid retriever combining BM25 and dense retrieval.

    BM25 provides lexical matching.

    Dense retrieval provides semantic matching.

    Reciprocal Rank Fusion combines their rankings.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        rrf_k: int = 60,
    ):
        if not chunks:
            raise ValueError(
                "chunks cannot be empty"
            )

        if rrf_k <= 0:
            raise ValueError(
                "rrf_k must be greater than 0"
            )

        self.chunks = chunks

        self.rrf_k = rrf_k

        self.sparse = BM25Retriever(
            chunks
        )

        self.dense = DenseIndex()

        self.dense.add_chunks(
            chunks
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        dense_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """
        Perform hybrid retrieval.

        BM25 and dense retrieval operate over the same
        metadata-filtered corpus.

        Their ranked results are combined with RRF.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        sparse_results = self.sparse.search(
            query,
            top_k=top_k,
            filters=filters,
        )

        dense_results = self.dense.search(
            query,
            top_k=top_k,
            filters=filters,
            score_threshold=dense_threshold,
        )

        return reciprocal_rank_fusion(
            [
                sparse_results,
                dense_results,
            ],
            k=self.rrf_k,
            top_k=top_k,
        )