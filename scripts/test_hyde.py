from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.hyde import (
    HyDERetriever,
    HypotheticalDocumentGenerator,
)


class FakeGenerator:
    def generate(self, query: str):
        return type(
            "HypotheticalDocument",
            (),
            {
                "query": query,
                "text": (
                    "The company expanded its enterprise "
                    "product platform with new AI features."
                ),
            },
        )()


def main() -> None:
    chunks = [
        Chunk(
            chunk_id="chunk-001",
            text=(
                "The company expanded its enterprise "
                "platform with AI-assisted features."
            ),
            document_id="doc-001",
            source="test",
            source_type="test",
        ),
        Chunk(
            chunk_id="chunk-002",
            text=(
                "The company introduced a new office "
                "location in Europe."
            ),
            document_id="doc-002",
            source="test",
            source_type="test",
        ),
        Chunk(
            chunk_id="chunk-003",
            text=(
                "The company changed its pricing model "
                "to usage-based billing."
            ),
            document_id="doc-003",
            source="test",
            source_type="test",
        ),
    ]

    dense_index = DenseIndex()
    dense_index.add_chunks(chunks)

    generator = HypotheticalDocumentGenerator()

    hypothetical = generator.generate(
        "What changed in the product strategy?"
    )

    assert hypothetical.query == (
        "What changed in the product strategy?"
    )

    assert "product strategy" in hypothetical.text
    assert "product roadmap" in hypothetical.text
    assert "customer segments" in hypothetical.text
    assert "product positioning" in hypothetical.text

    hyde = HyDERetriever(
        dense_index=dense_index,
    )

    results = hyde.search(
        "What changed in the product strategy?",
        top_k=2,
    )

    assert len(results) == 2

    result_ids = [
        result.chunk_id
        for result in results
    ]

    assert len(result_ids) == len(
        set(result_ids)
    )

    assert all(
        chunk_id in {
            "chunk-001",
            "chunk-002",
            "chunk-003",
        }
        for chunk_id in result_ids
    )

    assert results[0].score >= results[1].score

    fake_hyde = HyDERetriever(
        dense_index=dense_index,
        generator=FakeGenerator(),
    )

    fake_results = fake_hyde.search(
        "completely unrelated query",
        top_k=2,
    )

    assert len(fake_results) == 2

    fake_result_ids = [
        result.chunk_id
        for result in fake_results
    ]

    assert len(fake_result_ids) == len(
        set(fake_result_ids)
    )

    assert fake_results[0].score >= fake_results[1].score

    try:
        hyde.search("")

        raise AssertionError(
            "Expected ValueError for empty query"
        )

    except ValueError:
        pass

    try:
        hyde.search(
            "test query",
            top_k=0,
        )

        raise AssertionError(
            "Expected ValueError for invalid top_k"
        )

    except ValueError:
        pass

    print("Hypothetical document:")
    print(hypothetical.text)

    print("\nHyDE results:")

    for result in results:
        print(
            f"{result.rank}. "
            f"{result.chunk_id} "
            f"score={result.score:.4f}"
        )

    print("\nImportant:")
    print(
        "The test does not assume a specific semantic "
        "ranking because embedding models determine "
        "the ranking."
    )

    print("\nAll HyDE assertions passed.")


if __name__ == "__main__":
    main()