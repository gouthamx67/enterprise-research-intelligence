from src.rag_engine.retrieval.query_expansion import (
    QueryExpander,
    expand_query,
)


def main() -> None:
    expander = QueryExpander()

    query = "What changed in the company's product strategy?"

    result = expander.expand(query)

    assert result.original_query == query

    assert "product roadmap" in result.added_terms
    assert "product portfolio" in result.added_terms
    assert "strategic direction" in result.added_terms
    assert "roadmap" in result.added_terms

    assert result.expanded_query.startswith(query)

    assert "product roadmap" in result.expanded_query
    assert "strategic direction" in result.expanded_query

    convenience_result = expand_query(query)

    assert convenience_result == result.expanded_query

    try:
        expander.expand("")
        raise AssertionError("Expected ValueError for empty query")
    except ValueError:
        pass

    query_without_matching_terms = "quarterly revenue"

    result_without_matching_terms = expander.expand(
        query_without_matching_terms
    )

    assert (
        result_without_matching_terms.expanded_query
        == query_without_matching_terms
    )

    assert result_without_matching_terms.added_terms == []

    print("Original query:")
    print(result.original_query)

    print("\nAdded terms:")
    for term in result.added_terms:
        print(f"- {term}")

    print("\nExpanded query:")
    print(result.expanded_query)

    print("\nAll query expansion assertions passed.")


if __name__ == "__main__":
    main()