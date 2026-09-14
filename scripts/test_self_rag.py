from src.rag_engine.advanced.self_rag import SelfRAGCritic


def main():
    critic = SelfRAGCritic()

    evaluation = critic.evaluate(
        answer=(
            "The product strategy changed toward "
            "enterprise customers."
        ),
        evidence=[
            "The product strategy changed toward enterprise customers."
        ],
    )

    assert evaluation.supported is True
    assert evaluation.useful is True
    assert evaluation.score > 0

    unsupported = critic.evaluate(
        answer="The company became a space company.",
        evidence=[
            "The product strategy changed toward enterprise customers."
        ],
    )

    assert unsupported.supported is False

    no_evidence = critic.evaluate(
        answer="The company changed strategy.",
        evidence=[],
    )

    assert no_evidence.supported is False
    assert no_evidence.useful is False
    assert no_evidence.score == 0.0

    print("Self-RAG test passed.")


if __name__ == "__main__":
    main()