from src.rag_engine.production.reliability import (
    RetryConfig,
    retry_call,
)


def main():
    attempts = {
        "count": 0
    }

    def flaky():
        attempts["count"] += 1

        if attempts["count"] < 3:
            raise RuntimeError(
                "temporary failure"
            )

        return "success"

    result = retry_call(
        flaky,
        RetryConfig(
            attempts=2,
            delay_seconds=0,
        ),
    )

    assert result == "success"
    assert attempts["count"] == 3

    print(
        "Production reliability test passed."
    )


if __name__ == "__main__":
    main()