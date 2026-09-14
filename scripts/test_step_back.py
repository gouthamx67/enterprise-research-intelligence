from src.rag_engine.retrieval.step_back import (
    StepBackGenerator,
    generate_step_back_query,
)


def main() -> None:
    generator = StepBackGenerator()

    query = (
        "What changed in this company's "
        "product strategy over the last 12 months?"
    )

    result = generator.generate(query)

    assert result.original_query == query

    assert (
        "dimensions and factors"
        in result.step_back_query.lower()
    )

    assert (
        "product strategy"
        in result.step_back_query.lower()
    )

    assert (
        "positioning"
        in result.reasoning_focus.lower()
    )

    assert (
        "customers"
        in result.reasoning_focus.lower()
    )

    assert (
        "roadmap"
        in result.reasoning_focus.lower()
    )

    convenience_result = (
        generate_step_back_query(query)
    )

    assert (
        convenience_result
        == result.step_back_query
    )

    pricing_query = (
        "What changed in the pricing strategy?"
    )

    pricing_result = generator.generate(
        pricing_query
    )

    assert (
        "pricing strategy"
        in pricing_result.step_back_query.lower()
    )

    assert (
        "pricing models"
        in pricing_result.reasoning_focus.lower()
    )

    generic_query = "customer retention"

    generic_result = generator.generate(
        generic_query
    )

    assert (
        "broader concepts"
        in generic_result.step_back_query.lower()
    )

    assert (
        "general background"
        in generic_result.reasoning_focus.lower()
    )

    try:
        generator.generate("")

        raise AssertionError(
            "Expected ValueError for empty query"
        )

    except ValueError:
        pass

    print("Original query:")
    print(result.original_query)

    print("\nStep-back query:")
    print(result.step_back_query)

    print("\nReasoning focus:")
    print(result.reasoning_focus)

    print(
        "\nAll step-back assertions passed."
    )


if __name__ == "__main__":
    main()