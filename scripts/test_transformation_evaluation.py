from dataclasses import dataclass

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
)
from src.rag_engine.retrieval.transformation_evaluation import (
    TransformedQueryRetriever,
    evaluate_query_transformations,
    print_transformation_table,
)


@dataclass
class FakeResult:
    score: float
    rank: int
    chunk_id: str
    text: str
    metadata: dict


class FakeRetriever:
    """
    Deterministic retriever for testing the
    transformation evaluation framework.
    """

    def __init__(self):
        self.results = {
            "original product question": [
                FakeResult(
                    0.90,
                    1,
                    "chunk-noise",
                    "Noise",
                    {},
                ),
                FakeResult(
                    0.80,
                    2,
                    "chunk-a",
                    "Product strategy evidence",
                    {},
                ),
                FakeResult(
                    0.70,
                    3,
                    "chunk-noise-2",
                    "More noise",
                    {},
                ),
            ],
            "rewritten product strategy question": [
                FakeResult(
                    0.95,
                    1,
                    "chunk-a",
                    "Product strategy evidence",
                    {},
                ),
                FakeResult(
                    0.85,
                    2,
                    "chunk-b",
                    "Additional strategy evidence",
                    {},
                ),
                FakeResult(
                    0.60,
                    3,
                    "chunk-noise",
                    "Noise",
                    {},
                ),
            ],
            "product roadmap question": [
                FakeResult(
                    0.92,
                    1,
                    "chunk-b",
                    "Roadmap evidence",
                    {},
                ),
                FakeResult(
                    0.88,
                    2,
                    "chunk-a",
                    "Product strategy evidence",
                    {},
                ),
                FakeResult(
                    0.50,
                    3,
                    "chunk-noise",
                    "Noise",
                    {},
                ),
            ],
            "pricing question": [
                FakeResult(
                    0.95,
                    1,
                    "chunk-c",
                    "Pricing evidence",
                    {},
                ),
                FakeResult(
                    0.50,
                    2,
                    "chunk-noise",
                    "Noise",
                    {},
                ),
            ],
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
    ):
        if query not in self.results:
            return []

        return self.results[query][:top_k]


def build_dataset():
    return EvaluationDataset(
        [
            EvaluationQuery(
                query="original product question",
                relevant_chunk_ids={
                    "chunk-a",
                    "chunk-b",
                },
            ),
            EvaluationQuery(
                query="pricing question",
                relevant_chunk_ids={
                    "chunk-c",
                },
            ),
        ]
    )


def test_single_query_transformation():
    retriever = FakeRetriever()

    transformed = TransformedQueryRetriever(
        retriever,
        lambda query: (
            "rewritten product strategy question"
            if query == "original product question"
            else query
        ),
    )

    results = transformed.search(
        "original product question",
        top_k=3,
    )

    assert len(results) == 3

    assert results[0].chunk_id == "chunk-a"
    assert results[1].chunk_id == "chunk-b"

    assert results[0].rank == 1
    assert results[1].rank == 2

    assert (
        results[0].score
        >= results[1].score
    )


def test_multi_query_transformation():
    retriever = FakeRetriever()

    transformed = TransformedQueryRetriever(
        retriever,
        lambda query: [
            "rewritten product strategy question",
            "product roadmap question",
        ],
    )

    results = transformed.search(
        "original product question",
        top_k=3,
    )

    chunk_ids = [
        result.chunk_id
        for result in results
    ]

    # chunk-a appears in both queries but must
    # only occur once in the merged results.
    assert chunk_ids.count("chunk-a") == 1

    assert "chunk-a" in chunk_ids
    assert "chunk-b" in chunk_ids

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )


def test_evaluate_transformations():
    retriever = FakeRetriever()
    dataset = build_dataset()

    transformations = {
        "Original": lambda query: query,
        "Rewriting": lambda query: (
            "rewritten product strategy question"
            if query == "original product question"
            else query
        ),
        "Multi-query": lambda query: (
            [
                "rewritten product strategy question",
                "product roadmap question",
            ]
            if query == "original product question"
            else [query]
        ),
    }

    evaluations = evaluate_query_transformations(
        retriever,
        dataset,
        transformations,
        k=3,
    )

    assert len(evaluations) == 3

    names = [
        evaluation.name
        for evaluation in evaluations
    ]

    assert names == [
        "Original",
        "Rewriting",
        "Multi-query",
    ]

    for evaluation in evaluations:
        result = evaluation.result

        assert 0.0 <= result.recall_at_k <= 1.0
        assert 0.0 <= result.precision_at_k <= 1.0
        assert 0.0 <= result.mrr <= 1.0
        assert 0.0 <= result.ndcg_at_k <= 1.0

    original = evaluations[0].result
    rewriting = evaluations[1].result
    multi_query = evaluations[2].result

    # Rewriting should improve the first relevant
    # result for the deterministic test corpus.
    assert (
        rewriting.mrr
        > original.mrr
    )

    # Multi-query should retrieve both relevant
    # chunks for the product question.
    assert (
        multi_query.recall_at_k
        >= original.recall_at_k
    )

    print()
    print(
        "Query transformation evaluation:"
    )
    print()

    print_transformation_table(
        evaluations
    )


def test_invalid_transformer():
    retriever = FakeRetriever()

    try:
        TransformedQueryRetriever(
            retriever,
            None,
        )

        raise AssertionError(
            "Expected TypeError"
        )

    except TypeError:
        pass


def test_invalid_transformation_output():
    retriever = FakeRetriever()

    transformed = TransformedQueryRetriever(
        retriever,
        lambda query: 123,
    )

    try:
        transformed.search(
            "original product question"
        )

        raise AssertionError(
            "Expected TypeError"
        )

    except TypeError:
        pass


def main():
    test_single_query_transformation()
    test_multi_query_transformation()
    test_evaluate_transformations()
    test_invalid_transformer()
    test_invalid_transformation_output()

    print()
    print(
        "All query transformation "
        "evaluation assertions passed."
    )


if __name__ == "__main__":
    main()