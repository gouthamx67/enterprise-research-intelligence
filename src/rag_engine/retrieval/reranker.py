from dataclasses import dataclass, field
from typing import Any

from sentence_transformers import CrossEncoder

from src.rag_engine.retrieval.models import RetrievalResult


@dataclass
class RerankedResult:
    """
    A retrieval result after reranking.
    """

    score: float
    rank: int
    chunk_id: str
    text: str
    original_score: float
    original_rank: int
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


class CrossEncoderReranker:
    """
    Rerank retrieval candidates using a
    cross-encoder model.

    The model receives pairs:

        [query, candidate_text]

    and returns a relevance score for each pair.
    """

    def __init__(
        self,
        model_name: str = (
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
    ):
        if not model_name or not model_name.strip():
            raise ValueError(
                "model_name cannot be empty"
            )

        self.model_name = model_name

        self.model = CrossEncoder(
            model_name
        )

    def score(
        self,
        query: str,
        candidates: list[RetrievalResult],
    ) -> list[float]:
        """
        Score every candidate against the query.
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not candidates:
            return []

        pairs = [
            [
                query.strip(),
                candidate.text,
            ]
            for candidate in candidates
        ]

        scores = self.model.predict(
            pairs
        )

        return [
            float(score)
            for score in scores
        ]

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """
        Rerank candidates according to cross-encoder
        relevance scores.
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if not candidates:
            return []

        scores = self.score(
            query,
            candidates,
        )

        reranked = []

        for candidate, score in zip(
            candidates,
            scores,
        ):
            reranked.append(
                RerankedResult(
                    score=score,
                    rank=0,
                    chunk_id=candidate.chunk_id,
                    text=candidate.text,
                    original_score=candidate.score,
                    original_rank=candidate.rank,
                    metadata=candidate.metadata.copy(),
                )
            )

        reranked.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        if top_k is not None:
            reranked = reranked[:top_k]

        final_results = []

        for rank, result in enumerate(
            reranked,
            start=1,
        ):
            final_results.append(
                RerankedResult(
                    score=result.score,
                    rank=rank,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    original_score=result.original_score,
                    original_rank=result.original_rank,
                    metadata=result.metadata.copy(),
                )
            )

        return final_results