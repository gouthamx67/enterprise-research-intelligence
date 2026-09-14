from src.rag_engine.retrieval.query_rewriting import (
    QueryRewriter,
    rewrite_query,
)


def main() -> None:
    rewriter = QueryRewriter()

    query = (
        "What changed in this company's "
        "product strategy over the last 12 months?"
    )

    result = rewriter.rewrite(query)

    assert result.original_query == query

    assert (
        result.rewritten_query
        == (
            "changes in this company's "
            "product strategy roadmap during the past 12 months?"
        )
    )

    assert "phrase_normalization" in result.transformations

    convenience_result = rewrite_query(query)

    assert convenience_result == result.rewritten_query

    try:
        rewriter.rewrite("")
        raise AssertionError("Expected ValueError for empty query")
    except ValueError:
        pass

    print("Original query:")
    print(result.original_query)

    print("\nRewritten query:")
    print(result.rewritten_query)

    print("\nTransformations:")
    for transformation in result.transformations:
        print(f"- {transformation}")

    print("\nAll query rewriting assertions passed.")


if __name__ == "__main__":
    main()