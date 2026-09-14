from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
    EvaluationQuery,
    build_demo_evaluation_dataset,
)


def main() -> None:
    example = EvaluationQuery(
        query="enterprise product strategy",
        relevant_chunk_ids={
            "chunk-002",
            "chunk-004",
        },
        metadata={
            "topic": "product_strategy",
        },
    )

    assert (
        example.query
        == "enterprise product strategy"
    )

    assert example.relevant_chunk_ids == {
        "chunk-002",
        "chunk-004",
    }

    assert (
        example.metadata["topic"]
        == "product_strategy"
    )

    dataset = EvaluationDataset()

    assert len(dataset) == 0

    dataset.add(example)

    assert len(dataset) == 1

    assert dataset.get_queries() == [
        "enterprise product strategy"
    ]

    assert dataset.get_relevant_ids(
        "enterprise product strategy"
    ) == {
        "chunk-002",
        "chunk-004",
    }

    demo_dataset = (
        build_demo_evaluation_dataset()
    )

    assert len(demo_dataset) == 4

    assert (
        "pricing strategy"
        in demo_dataset.get_queries()
    )

    assert demo_dataset.get_relevant_ids(
        "pricing strategy"
    ) == {
        "chunk-003",
    }

    assert demo_dataset.get_relevant_ids(
        "AI workflow features"
    ) == {
        "chunk-004",
    }

    try:
        EvaluationQuery(
            query="",
            relevant_chunk_ids={"chunk-001"},
        )

        raise AssertionError(
            "Expected ValueError for empty query"
        )

    except ValueError:
        pass

    try:
        EvaluationQuery(
            query="test",
            relevant_chunk_ids=set(),
        )

        raise AssertionError(
            "Expected ValueError for empty relevance set"
        )

    except ValueError:
        pass

    try:
        dataset.add("not an evaluation query")

        raise AssertionError(
            "Expected TypeError for invalid example"
        )

    except TypeError:
        pass

    try:
        dataset.get_relevant_ids(
            "does not exist"
        )

        raise AssertionError(
            "Expected KeyError for unknown query"
        )

    except KeyError:
        pass

    print("Evaluation dataset:")

    for item in demo_dataset:
        print(f"\nQuery: {item.query}")
        print(
            "Relevant chunks: "
            f"{sorted(item.relevant_chunk_ids)}"
        )
        print(
            f"Metadata: {item.metadata}"
        )

    print(
        "\nAll evaluation dataset "
        "assertions passed."
    )


if __name__ == "__main__":
    main()