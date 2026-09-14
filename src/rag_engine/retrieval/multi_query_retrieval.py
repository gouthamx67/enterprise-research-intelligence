from dataclasses import dataclass, field
from typing import Any

from src.rag_engine.retrieval.models import RetrievalResult
from src.rag_engine.retrieval.multi_query import MultiQueryGenerator


@dataclass
class MultiQueryRetrievalResult:
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(default_factory=dict)
    source_queries: list[str] = field(default_factory=list)


class MultiQueryRetriever:
    """
    Runs retrieval for multiple generated queries and merges the results.

    Duplicate chunks are represented only once in the final result while
    preserving the queries that retrieved each chunk.
    """

    def __init__(
        self,
        retriever,
        query_generator: MultiQueryGenerator | None = None,
    ):
        if retriever is None:
            raise ValueError("retriever cannot be None")

        self.retriever = retriever
        self.query_generator = (
            query_generator
            if query_generator is not None
            else MultiQueryGenerator()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[MultiQueryRetrievalResult]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        generated_queries = self.query_generator.generate(
            query
        ).queries

        result_groups = []

        for generated_query in generated_queries:
            results = self.retriever.search(
                generated_query,
                top_k=top_k,
            )

            result_groups.append(
                (
                    generated_query,
                    results,
                )
            )

        merged = self._merge_results(result_groups)

        return self._rank_results(merged, top_k)

    @staticmethod
    def _merge_results(
        result_groups: list[
            tuple[str, list[RetrievalResult]]
        ],
    ) -> dict[str, MultiQueryRetrievalResult]:
        merged: dict[
            str,
            MultiQueryRetrievalResult,
        ] = {}

        for generated_query, results in result_groups:
            for result in results:
                existing = merged.get(result.chunk_id)

                if existing is None:
                    merged[result.chunk_id] = (
                        MultiQueryRetrievalResult(
                            chunk_id=result.chunk_id,
                            text=result.text,
                            score=result.score,
                            rank=result.rank,
                            metadata=result.metadata.copy(),
                            source_queries=[
                                generated_query
                            ],
                        )
                    )

                else:
                    if (
                        generated_query
                        not in existing.source_queries
                    ):
                        existing.source_queries.append(
                            generated_query
                        )

                    existing.score += result.score

        return merged

    @staticmethod
    def _rank_results(
        merged_results: dict[
            str,
            MultiQueryRetrievalResult,
        ],
        top_k: int,
    ) -> list[MultiQueryRetrievalResult]:
        ranked = sorted(
            merged_results.values(),
            key=lambda result: (
                len(result.source_queries),
                result.score,
            ),
            reverse=True,
        )

        final_results = ranked[:top_k]

        for rank, result in enumerate(
            final_results,
            start=1,
        ):
            result.rank = rank

        return final_results