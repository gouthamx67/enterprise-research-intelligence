from dataclasses import dataclass
from time import sleep
from typing import Callable, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RetryConfig:
    attempts: int = 2
    delay_seconds: float = 0.1

    def __post_init__(self):
        if self.attempts < 0:
            raise ValueError(
                "attempts cannot be negative"
            )

        if self.delay_seconds < 0:
            raise ValueError(
                "delay_seconds cannot be negative"
            )


def retry_call(
    function: Callable[[], T],
    config: RetryConfig,
) -> T:
    """
    Retry a callable after transient failures.

    attempts=0 means one initial execution with no retry.
    """
    last_error: Exception | None = None

    total_attempts = config.attempts + 1

    for attempt in range(total_attempts):
        try:
            return function()
        except Exception as exc:
            last_error = exc

            if attempt == total_attempts - 1:
                break

            if config.delay_seconds > 0:
                sleep(
                    config.delay_seconds
                )

    assert last_error is not None
    raise last_error


class ReliablePipeline:
    """
    Wraps a pipeline with retry behavior.
    """

    def __init__(
        self,
        pipeline,
        retry_config: RetryConfig | None = None,
    ):
        if pipeline is None:
            raise ValueError(
                "pipeline cannot be None"
            )

        self.pipeline = pipeline
        self.retry_config = (
            retry_config or RetryConfig()
        )

    def run(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
    ):
        return retry_call(
            lambda: self.pipeline.run(
                query=query,
                candidate_k=candidate_k,
                final_k=final_k,
            ),
            self.retry_config,
        )