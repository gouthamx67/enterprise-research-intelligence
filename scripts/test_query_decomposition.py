from src.rag_engine.retrieval.query_decomposition import (
    QueryDecomposer,
    decompose_query,
)


def main() -> None:
    decomposer = QueryDecomposer()

    query = (
        "What changed in this company's "
        "product strategy over the last 12 months?"
    )

    result = decomposer.decompose(query)

    assert result.original_query == query

    assert len(result.subqueries) == 6

    # Previous-state subquestion.
    assert any(
        (
            "before the last 12 months"
            in subquery.lower()
        )
        for subquery in result.subqueries
    )

    # Current-state subquestion.
    assert any(
        (
            "current product strategy"
            in subquery.lower()
        )
        for subquery in result.subqueries
    )

    # Product changes.
    assert any(
        (
            "new products"
            in subquery.lower()
        )
        for subquery in result.subqueries
    )

    # Customer / positioning changes.
    assert any(
        (
            "target customers"
            in subquery.lower()
        )
        for subquery in result.subqueries
    )

    # Pricing / monetization changes.
    assert any(
        (
            "pricing"
            in subquery.lower()
        )
        for subquery in result.subqueries
    )

    # Temporal evidence.
    assert any(
        (
            "evidence establishes when"
            in subquery.lower()
        )
        for subquery in result.subqueries
    )

    # No duplicate subqueries.
    normalized_subqueries = [
        subquery.lower()
        for subquery in result.subqueries
    ]

    assert len(normalized_subqueries) == len(
        set(normalized_subqueries)
    )

    # Convenience function should produce
    # the same result.
    convenience_result = decompose_query(query)

    assert convenience_result == result.subqueries

    # Pricing decomposition.
    pricing_query = (
        "What changed in the pricing strategy?"
    )

    pricing_result = decomposer.decompose(
        pricing_query
    )

    assert len(pricing_result.subqueries) == 3

    assert any(
        (
            "pricing model"
            in subquery.lower()
        )
        for subquery in pricing_result.subqueries
    )

    # Generic decomposition.
    generic_query = "customer retention"

    generic_result = decomposer.decompose(
        generic_query
    )

    assert len(generic_result.subqueries) == 3

    assert any(
        (
            "main components"
            in subquery.lower()
        )
        for subquery in generic_result.subqueries
    )

    # Empty query validation.
    try:
        decomposer.decompose("")

        raise AssertionError(
            "Expected ValueError for empty query"
        )

    except ValueError:
        pass

    print("Original query:")
    print(result.original_query)

    print("\nDecomposed subqueries:")

    for index, subquery in enumerate(
        result.subqueries,
        start=1,
    ):
        print(f"{index}. {subquery}")

    print(
        "\nAll query decomposition "
        "assertions passed."
    )


if __name__ == "__main__":
    main()