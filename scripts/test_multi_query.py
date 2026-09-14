from src.rag_engine.retrieval.multi_query import (
    MultiQueryGenerator,
    generate_queries,
)


def main() -> None:
    generator = MultiQueryGenerator()

    query = (
        "What changed in this company's "
        "product strategy over the last 12 months?"
    )

    result = generator.generate(query)

    assert result.original_query == query

    assert len(result.queries) == 5

    assert result.queries[0] == query

    assert (
        "product strategy changes"
        in result.queries[1].lower()
    )

    assert (
        "product roadmap"
        in result.queries[2].lower()
    )

    assert (
        "new products or features"
        in result.queries[3].lower()
    )

    assert (
        "product positioning"
        in result.queries[4].lower()
    )

    normalized_queries = [
        item.lower()
        for item in result.queries
    ]

    assert len(normalized_queries) == len(
        set(normalized_queries)
    )

    convenience_result = generate_queries(query)

    assert convenience_result == result.queries

    try:
        generator.generate("")
        raise AssertionError(
            "Expected ValueError for empty query"
        )
    except ValueError:
        pass

    pricing_query = "What changed in the pricing strategy?"

    pricing_result = generator.generate(pricing_query)

    assert len(pricing_result.queries) == 4

    assert any(
        "pricing model" in item.lower()
        for item in pricing_result.queries
    )

    generic_query = "customer retention"

    generic_result = generator.generate(generic_query)

    assert len(generic_result.queries) == 3

    assert generic_result.queries[0] == generic_query

    print("Original query:")
    print(result.original_query)

    print("\nGenerated queries:")

    for index, generated_query in enumerate(
        result.queries,
        start=1,
    ):
        print(f"{index}. {generated_query}")

    print("\nAll multi-query assertions passed.")


if __name__ == "__main__":
    main()