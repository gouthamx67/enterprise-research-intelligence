from dataclasses import dataclass

from src.rag_engine.retrieval.models import (
    RetrievalResult,
)
from src.rag_engine.retrieval.reranker import (
    CrossEncoderReranker,
)
from src.rag_engine.retrieval.reranking_pipeline import (
    RetrievalRerankingPipeline,
    build_reranking_pipeline,
)


@dataclass
class FakeRetriever:
    """
    Deterministic first-stage retriever.
    """

    calls: list

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters=None,
    ):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "filters": filters,
            }
        )

        candidates = [
            RetrievalResult(
                score=0.90,
                rank=1,
                chunk_id="chunk-1",
                text=(
                    "The company has several "
                    "regional offices."
                ),
                metadata={
                    "company": "Acme",
                },
            ),
            RetrievalResult(
                score=0.88,
                rank=2,
                chunk_id="chunk-2",
                text=(
                    "The company changed its "
                    "product strategy."
                ),
                metadata={
                    "company": "Acme",
                },
            ),
            RetrievalResult(
                score=0.85,
                rank=3,
                chunk_id="chunk-3",
                text=(
                    "The company changed its "
                    "pricing strategy."
                ),
                metadata={
                    "company": "Acme",
                },
            ),
            RetrievalResult(
                score=0.80,
                rank=4,
                chunk_id="chunk-4",
                text=(
                    "The company operates "
                    "globally."
                ),
                metadata={
                    "company": "Acme",
                },
            ),
            RetrievalResult(
                score=0.75,
                rank=5,
                chunk_id="chunk-5",
                text=(
                    "The company expanded its "
                    "product roadmap."
                ),
                metadata={
                    "company": "Acme",
                },
            ),
        ]

        return candidates[:top_k]


class FakeCrossEncoder:
    """
    Deterministic reranker.

    It deliberately changes the first-stage ranking.
    """

    def predict(self, pairs):
        scores = []

        for query, text in pairs:
            text_lower = text.lower()

            if (
                "product strategy"
                in text_lower
            ):
                scores.append(0.99)

            elif (
                "product roadmap"
                in text_lower
            ):
                scores.append(0.90)

            elif (
                "pricing strategy"
                in text_lower
            ):
                scores.append(0.75)

            else:
                scores.append(0.10)

        return scores


def build_test_reranker():
    reranker = CrossEncoderReranker.__new__(
        CrossEncoderReranker
    )

    reranker.model_name = "fake-model"
    reranker.model = FakeCrossEncoder()

    return reranker


def main():
    retriever = FakeRetriever(
        calls=[]
    )

    reranker = build_test_reranker()

    pipeline = RetrievalRerankingPipeline(
        retriever=retriever,
        reranker=reranker,
    )

    query = (
        "What changed in product strategy?"
    )

    result = pipeline.search(
        query,
        candidate_k=5,
        final_k=2,
    )

    # ---------------------------------------------
    # Verify pipeline metadata
    # ---------------------------------------------

    assert result.query == query

    assert result.candidate_count == 5

    assert result.final_count == 2

    assert len(result.results) == 2

    # ---------------------------------------------
    # Verify reranking changed the order
    # ---------------------------------------------

    assert (
        result.results[0].chunk_id
        == "chunk-2"
    )

    assert (
        result.results[1].chunk_id
        == "chunk-5"
    )

    # ---------------------------------------------
    # Verify final ranks
    # ---------------------------------------------

    assert result.results[0].rank == 1
    assert result.results[1].rank == 2

    # ---------------------------------------------
    # Verify original retrieval ranks survive
    # ---------------------------------------------

    assert (
        result.results[0].original_rank
        == 2
    )

    assert (
        result.results[1].original_rank
        == 5
    )

    # ---------------------------------------------
    # Verify retriever received candidate_k
    # ---------------------------------------------

    assert len(retriever.calls) == 1

    assert (
        retriever.calls[0]["query"]
        == query
    )

    assert (
        retriever.calls[0]["top_k"]
        == 5
    )

    # ---------------------------------------------
    # Verify metadata filters propagate
    # ---------------------------------------------

    retriever.calls.clear()

    filters = {
        "company": "Acme"
    }

    pipeline.search(
        query,
        candidate_k=5,
        final_k=2,
        filters=filters,
    )

    assert len(retriever.calls) == 1

    assert (
        retriever.calls[0]["filters"]
        == filters
    )

    # ---------------------------------------------
    # Verify final_k <= candidate_k
    # ---------------------------------------------

    try:
        pipeline.search(
            query,
            candidate_k=2,
            final_k=3,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid candidate_k
    # ---------------------------------------------

    try:
        pipeline.search(
            query,
            candidate_k=0,
            final_k=1,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid final_k
    # ---------------------------------------------

    try:
        pipeline.search(
            query,
            candidate_k=5,
            final_k=0,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid query
    # ---------------------------------------------

    try:
        pipeline.search(
            "",
            candidate_k=5,
            final_k=2,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Factory function
    # ---------------------------------------------

    factory_pipeline = (
        build_reranking_pipeline(
            retriever,
            reranker,
        )
    )

    assert isinstance(
        factory_pipeline,
        RetrievalRerankingPipeline,
    )

    print(
        "Candidate generation:"
    )

    print(
        "  candidate_k = 5"
    )

    print(
        "  candidates retrieved = 5"
    )

    print(
        "\nFinal reranking:"
    )

    print(
        "  final_k = 2"
    )

    print(
        "\nFinal results:"
    )

    for item in result.results:
        print(
            f"  {item.rank}. "
            f"{item.chunk_id} "
            f"(rerank_score="
            f"{item.score:.4f}, "
            f"original_rank="
            f"{item.original_rank})"
        )

    print(
        "\nAll reranking pipeline "
        "assertions passed."
    )


if __name__ == "__main__":
    main()