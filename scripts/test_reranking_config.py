from src.rag_engine.retrieval.reranking_config import (
    RerankingConfig,
    build_default_reranking_config,
    build_fast_reranking_config,
    build_high_recall_reranking_config,
)


def main():
    # ---------------------------------------------
    # Default configuration
    # ---------------------------------------------

    config = RerankingConfig()

    assert config.candidate_k == 20
    assert config.final_k == 5
    assert (
        config.model_name
        == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    assert config.score_threshold is None

    # ---------------------------------------------
    # Custom configuration
    # ---------------------------------------------

    custom = RerankingConfig(
        candidate_k=30,
        final_k=10,
        model_name="custom-reranker",
        score_threshold=0.25,
    )

    assert custom.candidate_k == 30
    assert custom.final_k == 10
    assert custom.model_name == "custom-reranker"
    assert custom.score_threshold == 0.25

    # ---------------------------------------------
    # Default builder
    # ---------------------------------------------

    default = build_default_reranking_config()

    assert default == RerankingConfig()

    # ---------------------------------------------
    # Fast configuration
    # ---------------------------------------------

    fast = build_fast_reranking_config()

    assert fast.candidate_k == 10
    assert fast.final_k == 5

    # ---------------------------------------------
    # High recall configuration
    # ---------------------------------------------

    high_recall = (
        build_high_recall_reranking_config()
    )

    assert high_recall.candidate_k == 50
    assert high_recall.final_k == 5

    # ---------------------------------------------
    # Invalid candidate K
    # ---------------------------------------------

    try:
        RerankingConfig(candidate_k=0)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid final K
    # ---------------------------------------------

    try:
        RerankingConfig(final_k=0)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # final_k > candidate_k
    # ---------------------------------------------

    try:
        RerankingConfig(
            candidate_k=5,
            final_k=10,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Empty model
    # ---------------------------------------------

    try:
        RerankingConfig(
            model_name="   "
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid threshold
    # ---------------------------------------------

    try:
        RerankingConfig(
            score_threshold=2.0
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print(
        "All reranking configuration "
        "assertions passed."
    )


if __name__ == "__main__":
    main()
