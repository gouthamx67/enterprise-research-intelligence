from src.rag_engine.advanced.corrective import CorrectiveRetriever
from src.rag_engine.retrieval.models import RetrievalResult


def make_results():
    return [
        RetrievalResult(
            score=0.90,
            rank=1,
            chunk_id="a",
            text="Product strategy changed.",
        ),
        RetrievalResult(
            score=0.80,
            rank=2,
            chunk_id="b",
            text="Enterprise customers became important.",
        ),
        RetrievalResult(
            score=0.20,
            rank=3,
            chunk_id="c",
            text="Unrelated information.",
        ),
    ]


def main():
    corrective = CorrectiveRetriever(
        minimum_score=0.50,
        minimum_relevant_results=2,
    )

    results = make_results()

    assessment = corrective.assess(
        "product strategy",
        results,
    )

    assert assessment.sufficient is True
    assert assessment.relevant_count == 2
    assert assessment.total_count == 3

    result = corrective.evaluate(
        "product strategy",
        results,
    )

    assert result.should_retry is False
    assert len(result.accepted_results) == 3

    empty = corrective.evaluate(
        "product strategy",
        [],
    )

    assert empty.should_retry is True
    assert len(empty.accepted_results) == 0

    print("Corrective RAG test passed.")


if __name__ == "__main__":
    main()