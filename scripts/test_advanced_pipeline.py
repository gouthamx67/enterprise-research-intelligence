from src.rag_engine.advanced.pipeline import AdvancedRAGPipeline
from src.rag_engine.retrieval.models import RetrievalResult


class FakeRetriever:
    """
    Deterministic retriever for integration testing.
    """

    def __init__(self):
        self.calls = []

    def search(self, query, top_k=5):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
            }
        )

        query_lower = query.lower()

        if "previous" in query_lower:
            return [
                RetrievalResult(
                    score=0.88,
                    rank=1,
                    chunk_id="historical-001",
                    text=(
                        "The previous product strategy "
                        "focused primarily on smaller customers."
                    ),
                ),
                RetrievalResult(
                    score=0.81,
                    rank=2,
                    chunk_id="historical-002",
                    text=(
                        "The earlier roadmap emphasized "
                        "self-service products."
                    ),
                ),
            ]

        if "new products" in query_lower:
            return [
                RetrievalResult(
                    score=0.91,
                    rank=1,
                    chunk_id="product-001",
                    text=(
                        "The company launched new enterprise "
                        "AI capabilities."
                    ),
                ),
            ]

        if "target customers" in query_lower:
            return [
                RetrievalResult(
                    score=0.86,
                    rank=1,
                    chunk_id="customer-001",
                    text=(
                        "The company increasingly targets "
                        "enterprise customers."
                    ),
                ),
            ]

        return [
            RetrievalResult(
                score=0.94,
                rank=1,
                chunk_id="strategy-001",
                text=(
                    "The product strategy changed toward "
                    "enterprise customers."
                ),
            ),
            RetrievalResult(
                score=0.89,
                rank=2,
                chunk_id="strategy-002",
                text=(
                    "AI capabilities became central to "
                    "the product roadmap."
                ),
            ),
        ]


def main():
    retriever = FakeRetriever()

    pipeline = AdvancedRAGPipeline(
        retriever=retriever,
    )

    query = (
        "What changed in this company's "
        "product strategy over the last 12 months?"
    )

    result = pipeline.run(
        query=query,
        top_k=5,
        answer=(
            "The product strategy changed toward "
            "enterprise customers and AI capabilities "
            "became central to the roadmap."
        ),
    )

    # ---------------------------------------------------------
    # Agentic planning
    # ---------------------------------------------------------

    assert len(result.plan.steps) == 4

    # ---------------------------------------------------------
    # Adaptive routing
    # ---------------------------------------------------------

    assert result.routing.strategy == "hybrid"

    # ---------------------------------------------------------
    # Retrieval execution
    # ---------------------------------------------------------

    assert len(retriever.calls) == 4

    assert len(result.retrieval_results) == 6

    # ---------------------------------------------------------
    # Corrective RAG
    # ---------------------------------------------------------

    assert result.correction.assessment.sufficient is True
    assert result.correction.should_retry is False

    assert (
        len(result.correction.accepted_results)
        == len(result.retrieval_results)
    )

    # ---------------------------------------------------------
    # Self-RAG
    # ---------------------------------------------------------

    assert result.self_rag is not None
    assert result.self_rag.supported is True
    assert result.self_rag.useful is True

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    assert result.metadata["planner_steps"] == 4
    assert (
        result.metadata["retrieval_count"]
        == len(result.retrieval_results)
    )
    assert (
        result.metadata["accepted_count"]
        == len(result.correction.accepted_results)
    )

    # ---------------------------------------------------------
    # Verify evidence IDs
    # ---------------------------------------------------------

    ids = {
        item.chunk_id
        for item in result.retrieval_results
    }

    assert "strategy-001" in ids
    assert "historical-001" in ids
    assert "product-001" in ids
    assert "customer-001" in ids

    print("Advanced RAG pipeline test passed.")
    print()
    print(f"Planner steps:       {len(result.plan.steps)}")
    print(f"Retrieval calls:     {len(retriever.calls)}")
    print(f"Unique evidence:     {len(result.retrieval_results)}")
    print(
        "Accepted evidence:  "
        f"{len(result.correction.accepted_results)}"
    )
    print(
        "Self-RAG supported:  "
        f"{result.self_rag.supported}"
    )
    print(
        "Self-RAG score:      "
        f"{result.self_rag.score:.3f}"
    )


if __name__ == "__main__":
    main()