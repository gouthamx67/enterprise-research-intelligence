from src.rag_engine.advanced.adaptive import AdaptiveRouter


def main():
    router = AdaptiveRouter()

    decision = router.route(
        "What changed in this company's product strategy?"
    )

    assert decision.strategy == "hybrid"

    decision = router.route(
        "Which companies did the company acquire?"
    )

    assert decision.strategy == "graph"

    decision = router.route(
        "What does this term mean exactly?"
    )

    assert decision.strategy == "sparse"

    decision = router.route(
        "Show me the product architecture diagram."
    )

    assert decision.strategy == "multimodal"

    decision = router.route(
        "Why is enterprise adoption increasing?"
    )

    assert decision.strategy == "dense"

    print("Adaptive RAG test passed.")


if __name__ == "__main__":
    main()