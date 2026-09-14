from dataclasses import dataclass

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
)
from src.rag_engine.retrieval.evaluation_runner import (
    evaluate_retriever,
)


@dataclass
class FakeResult:
    chunk_id: str


class FakeRetriever:
    """
    Small deterministic retriever used to test
    the evaluation runner without embeddings.
    """

    def __init__(self):
        self.results = {
            "query one": [
                FakeResult("chunk-a"),
                FakeResult("chunk-x"),
                FakeResult("chunk-y"),
            ],
            "query two": [
                FakeResult("chunk-x"),
                FakeResult("chunk-b"),
                FakeResult("chunk-y"),
            ],
            "query three": [
                FakeResult("chunk-z"),
                FakeResult("chunk-q"),
                FakeResult("chunk-r"),
            ],
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        return self.results[query][:top_k]


def main() -> None:
    dataset = EvaluationDataset(
        [
            EvaluationQuery(
                query="query one",
                relevant_chunk_ids={
                    "chunk-a",
                },
            ),
            EvaluationQuery(
                query="query two",
                relevant_chunk_ids={
                    "chunk-b",
                },
            ),
            EvaluationQuery(
                query="query three",
                relevant_chunk_ids={
                    "chunk-c",
                },
            ),
        ]
    )

    retriever = FakeRetriever()

    result = evaluate_retriever(
        retriever,
        dataset,
        k=3,
    )

    # Query one:
    # relevant at rank 1
    #
    # Query two:
    # relevant at rank 2
    #
    # Query three:
    # no relevant result
    #
    # Recall:
    # 1 + 1 + 0
    # ---------
    #     3
    #
    # = 2/3
    assert result.recall_at_k == 2 / 3

    # Each query retrieves 3 results.
    # One relevant result appears in two queries.
    #
    # Precision = 2 relevant / 9 retrieved
    assert result.precision_at_k == 2 / 9

    # MRR:
    #
    # query one = 1/1
    # query two = 1/2
    # query three = 0
    #
    # (1 + 0.5 + 0) / 3
    assert result.mrr == 0.5

    assert result.k == 3

    # nDCG should be between 0 and 1.
    assert 0.0 <= result.ndcg_at_k <= 1.0

    try:
        evaluate_retriever(
            retriever,
            dataset,
            k=0,
        )

        raise AssertionError(
            "Expected ValueError for invalid k"
        )

    except ValueError:
        pass

    try:
        evaluate_retriever(
            retriever,
            None,
            k=3,
        )

        raise AssertionError(
            "Expected ValueError for invalid dataset"
        )

    except (ValueError, TypeError):
        pass

    print("Evaluation result:")
    print(
        f"  Recall@3:    "
        f"{result.recall_at_k:.4f}"
    )
    print(
        f"  Precision@3: "
        f"{result.precision_at_k:.4f}"
    )
    print(
        f"  MRR:         "
        f"{result.mrr:.4f}"
    )
    print(
        f"  nDCG@3:      "
        f"{result.ndcg_at_k:.4f}"
    )

    print(
        "\nAll evaluation runner "
        "assertions passed."
    )


if __name__ == "__main__":
    main()