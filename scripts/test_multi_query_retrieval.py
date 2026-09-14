from dataclasses import dataclass

from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.models import RetrievalResult
from src.rag_engine.retrieval.multi_query_retrieval import (
    MultiQueryRetriever,
)


class FakeQueryGenerator:
    def generate(self, query: str):
        return type(
            "GeneratedQueries",
            (),
            {
                "queries": [
                    "query one",
                    "query two",
                    "query three",
                ]
            },
        )()


class FakeRetriever:
    RESULTS = {
        "query one": [
            RetrievalResult(
                score=0.90,
                rank=1,
                chunk_id="chunk-a",
                text="Text A",
                metadata={"source": "doc-a"},
            ),
            RetrievalResult(
                score=0.80,
                rank=2,
                chunk_id="chunk-b",
                text="Text B",
                metadata={"source": "doc-b"},
            ),
            RetrievalResult(
                score=0.70,
                rank=3,
                chunk_id="chunk-c",
                text="Text C",
                metadata={"source": "doc-c"},
            ),
        ],
        "query two": [
            RetrievalResult(
                score=0.95,
                rank=1,
                chunk_id="chunk-a",
                text="Text A",
                metadata={"source": "doc-a"},
            ),
            RetrievalResult(
                score=0.85,
                rank=2,
                chunk_id="chunk-c",
                text="Text C",
                metadata={"source": "doc-c"},
            ),
            RetrievalResult(
                score=0.75,
                rank=3,
                chunk_id="chunk-d",
                text="Text D",
                metadata={"source": "doc-d"},
            ),
        ],
        "query three": [
            RetrievalResult(
                score=0.88,
                rank=1,
                chunk_id="chunk-a",
                text="Text A",
                metadata={"source": "doc-a"},
            ),
            RetrievalResult(
                score=0.78,
                rank=2,
                chunk_id="chunk-d",
                text="Text D",
                metadata={"source": "doc-d"},
            ),
            RetrievalResult(
                score=0.68,
                rank=3,
                chunk_id="chunk-e",
                text="Text E",
                metadata={"source": "doc-e"},
            ),
        ],
    }

    def search(self, query: str, top_k: int = 5):
        return self.RESULTS[query][:top_k]


def main() -> None:
    retriever = MultiQueryRetriever(
        retriever=FakeRetriever(),
        query_generator=FakeQueryGenerator(),
    )

    results = retriever.search(
        "original query",
        top_k=5,
    )

    assert len(results) == 5

    chunk_ids = [
        result.chunk_id
        for result in results
    ]

    assert len(chunk_ids) == len(set(chunk_ids))

    assert chunk_ids[0] == "chunk-a"

    chunk_a = results[0]

    assert len(chunk_a.source_queries) == 3

    assert "query one" in chunk_a.source_queries
    assert "query two" in chunk_a.source_queries
    assert "query three" in chunk_a.source_queries

    chunk_c = next(
        result
        for result in results
        if result.chunk_id == "chunk-c"
    )

    assert len(chunk_c.source_queries) == 2

    chunk_d = next(
        result
        for result in results
        if result.chunk_id == "chunk-d"
    )

    assert len(chunk_d.source_queries) == 2

    chunk_b = next(
        result
        for result in results
        if result.chunk_id == "chunk-b"
    )

    assert len(chunk_b.source_queries) == 1

    assert chunk_a.score == 0.90 + 0.95 + 0.88

    assert chunk_c.score == 0.70 + 0.85

    try:
        retriever.search("")
        raise AssertionError(
            "Expected ValueError for empty query"
        )
    except ValueError:
        pass

    try:
        retriever.search(
            "query",
            top_k=0,
        )
        raise AssertionError(
            "Expected ValueError for invalid top_k"
        )
    except ValueError:
        pass

    print("Final merged results:")

    for result in results:
        print(
            f"{result.rank}. "
            f"{result.chunk_id} "
            f"score={result.score:.2f} "
            f"found_by={len(result.source_queries)} queries"
        )

    print("\nSource queries for chunk-a:")

    for source_query in chunk_a.source_queries:
        print(f"- {source_query}")

    print(
        "\nAll multi-query retrieval "
        "assertions passed."
    )


if __name__ == "__main__":
    main()