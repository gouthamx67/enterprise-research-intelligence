from src.rag_engine.pipeline.rag import (
    EndToEndRAGPipeline,
)

from src.rag_engine.retrieval.models import (
    RetrievalResult,
)

from src.rag_engine.retrieval.reranker import (
    RerankedResult,
)

from src.rag_engine.retrieval.context_construction import (
    ContextConstructionPipeline,
)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(
        self,
        query,
        top_k=20,
    ):
        self.calls.append(query)

        normalized = query.lower()

        if "previous" in normalized:
            return [
                RetrievalResult(
                    score=0.91,
                    rank=1,
                    chunk_id="old-001",
                    text=(
                        "The previous strategy focused "
                        "on smaller customers."
                    ),
                    metadata={
                        "source": "2025_report.pdf",
                        "source_type": "pdf",
                        "document_date": "2025-01-15",
                    },
                ),
            ]

        if "new products" in normalized:
            return [
                RetrievalResult(
                    score=0.89,
                    rank=1,
                    chunk_id="product-001",
                    text=(
                        "The company launched new "
                        "enterprise AI capabilities."
                    ),
                    metadata={
                        "source": "2026_report.pdf",
                        "source_type": "pdf",
                        "document_date": "2026-01-15",
                    },
                ),
            ]

        if "target customers" in normalized:
            return [
                RetrievalResult(
                    score=0.87,
                    rank=1,
                    chunk_id="customer-001",
                    text=(
                        "The company increasingly "
                        "targets enterprise customers."
                    ),
                    metadata={
                        "source": "2026_transcript.pdf",
                        "source_type": "transcript",
                        "document_date": "2026-02-01",
                    },
                ),
            ]

        return [
            RetrievalResult(
                score=0.95,
                rank=1,
                chunk_id="strategy-001",
                text=(
                    "The product strategy changed "
                    "toward enterprise customers."
                ),
                metadata={
                    "source": "2026_report.pdf",
                    "source_type": "pdf",
                    "document_date": "2026-01-15",
                },
            ),
            RetrievalResult(
                score=0.90,
                rank=2,
                chunk_id="strategy-002",
                text=(
                    "AI capabilities became central "
                    "to the product roadmap."
                ),
                metadata={
                    "source": "2026_report.pdf",
                    "source_type": "pdf",
                    "document_date": "2026-01-15",
                },
            ),
        ]


class FakeReranker:
    def rerank(
        self,
        query,
        candidates,
        top_k=None,
    ):
        ordered = sorted(
            candidates,
            key=lambda item: item.score,
            reverse=True,
        )

        if top_k is not None:
            ordered = ordered[:top_k]

        return [
            RerankedResult(
                score=result.score,
                rank=index,
                chunk_id=result.chunk_id,
                text=result.text,
                original_score=result.score,
                original_rank=result.rank,
                metadata=result.metadata,
            )
            for index, result in enumerate(
                ordered,
                start=1,
            )
        ]


def main():
    retriever = FakeRetriever()
    reranker = FakeReranker()

    context_pipeline = ContextConstructionPipeline(
        max_chunks=5,
        max_sentences_per_chunk=2,
        min_relevance_score=0.0,
        ordering_strategy="rank_order",
    )

    pipeline = EndToEndRAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_pipeline=context_pipeline,
    )

    query = (
        "What changed in this company's "
        "product strategy over the last 12 months?"
    )

    response = pipeline.run(
        query=query,
        candidate_k=10,
        final_k=5,
    )

    # =========================================================
    # Planner
    # =========================================================

    assert len(response.plan.steps) == 4

    # =========================================================
    # Adaptive routing
    # =========================================================

    assert response.routing.strategy == "hybrid"

    # =========================================================
    # Retrieval
    # =========================================================

    assert len(retriever.calls) == 4
    assert len(response.retrieved_results) == 5

    # =========================================================
    # Context
    # =========================================================

    assert response.metadata["context_count"] > 0

    # =========================================================
    # Generation
    # =========================================================

    assert response.generated_answer is not None

    assert (
        "enterprise"
        in response.generated_answer.answer.lower()
    )

    assert (
        "strategy"
        in response.generated_answer.answer.lower()
    )

    # =========================================================
    # Citations
    # =========================================================

    assert len(
        response.generated_answer.citation_ids
    ) > 0

    assert response.metadata["citation_valid"] is True

    # =========================================================
    # Confidence
    # =========================================================

    assert 0.0 <= response.confidence <= 1.0

    # =========================================================
    # Failure analysis
    # =========================================================

    assert (
        response.failure_analysis.has_failures
        is False
    )

    assert response.success is True

    # =========================================================
    # Metadata
    # =========================================================

    assert (
        response.metadata["candidate_count"]
        == 5
    )

    assert (
        response.metadata["reranked_count"]
        == 5
    )

    print("End-to-end RAG test passed.")
    print()
    print(f"Planner steps:       {len(response.plan.steps)}")
    print(f"Retrieval calls:     {len(retriever.calls)}")
    print(
        "Retrieved/reranked: "
        f"{len(response.retrieved_results)}"
    )
    print(
        "Final contexts:     "
        f"{response.metadata['context_count']}"
    )
    print(
        "Citations:          "
        f"{len(response.generated_answer.citation_ids)}"
    )
    print(
        "Confidence:         "
        f"{response.confidence:.3f}"
    )
    print(
        "Failures:           "
        f"{response.failure_analysis.failure_count}"
    )
    print(
        "Success:            "
        f"{response.success}"
    )


if __name__ == "__main__":
    main()