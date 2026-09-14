from dataclasses import dataclass

from src.rag_engine.retrieval.comparison import (
    compare_retrievers,
    print_comparison_table,
)
from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
)


@dataclass
class FakeResult:
    chunk_id: str


class FakeRetriever:
    def __init__(self, results):
        self.results = results

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        return self.results[query][:top_k]


def build_dataset() -> EvaluationDataset:
    return EvaluationDataset(
        [
            EvaluationQuery(
                query="product strategy",
                relevant_chunk_ids={
                    "chunk-a",
                    "chunk-b",
                },
            ),
            EvaluationQuery(
                query="pricing strategy",
                relevant_chunk_ids={
                    "chunk-c",
                },
            ),
            EvaluationQuery(
                query="AI capabilities",
                relevant_chunk_ids={
                    "chunk-d",
                },
            ),
        ]
    )


def build_retrievers() -> dict[str, object]:
    bm25 = FakeRetriever(
        {
            "product strategy": [
                FakeResult("chunk-a"),
                FakeResult("chunk-x"),
                FakeResult("chunk-b"),
            ],
            "pricing strategy": [
                FakeResult("chunk-c"),
                FakeResult("chunk-y"),
                FakeResult("chunk-z"),
            ],
            "AI capabilities": [
                FakeResult("chunk-x"),
                FakeResult("chunk-d"),
                FakeResult("chunk-y"),
            ],
        }
    )

    dense = FakeRetriever(
        {
            "product strategy": [
                FakeResult("chunk-x"),
                FakeResult("chunk-a"),
                FakeResult("chunk-y"),
            ],
            "pricing strategy": [
                FakeResult("chunk-y"),
                FakeResult("chunk-c"),
                FakeResult("chunk-z"),
            ],
            "AI capabilities": [
                FakeResult("chunk-d"),
                FakeResult("chunk-x"),
                FakeResult("chunk-y"),
            ],
        }
    )

    hybrid = FakeRetriever(
        {
            "product strategy": [
                FakeResult("chunk-a"),
                FakeResult("chunk-b"),
                FakeResult("chunk-x"),
            ],
            "pricing strategy": [
                FakeResult("chunk-c"),
                FakeResult("chunk-y"),
                FakeResult("chunk-z"),
            ],
            "AI capabilities": [
                FakeResult("chunk-d"),
                FakeResult("chunk-x"),
                FakeResult("chunk-y"),
            ],
        }
    )

    return {
        "BM25": bm25,
        "Dense": dense,
        "Hybrid": hybrid,
    }


def main() -> None:
    dataset = build_dataset()
    retrievers = build_retrievers()

    comparisons = compare_retrievers(
        retrievers,
        dataset,
        k=3,
    )

    assert len(comparisons) == 3

    assert comparisons[0].name == "BM25"
    assert comparisons[1].name == "Dense"
    assert comparisons[2].name == "Hybrid"

    for comparison in comparisons:
        result = comparison.result

        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.precision_at_k <= 1.0
        assert 0.0 <= result.mrr <= 1.0
        assert 0.0 <= result.ndcg_at_k <= 1.0
        assert result.k == 3

    # Hybrid was deliberately constructed to
    # perform best on this deterministic test corpus.
    assert (
        comparisons[2].result.mrr
        >= comparisons[0].result.mrr
    )

    assert (
        comparisons[2].result.mrr
        >= comparisons[1].result.mrr
    )

    print("Retrieval comparison:")
    print()

    print_comparison_table(
        comparisons
    )

    print()
    print(
        "All retrieval comparison "
        "assertions passed."
    )


if __name__ == "__main__":
    main()