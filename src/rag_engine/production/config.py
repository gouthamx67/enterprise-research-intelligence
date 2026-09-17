from dataclasses import dataclass
import os


@dataclass(frozen=True)
class ProductionConfig:
    """
    Central configuration for the production RAG service.
    """

    environment: str = "development"

    candidate_k: int = 20
    final_k: int = 5

    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    request_timeout_seconds: float = 30.0
    retry_attempts: int = 2

    max_query_length: int = 2000

    require_company: bool = False

    def __post_init__(self):
        if self.candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        if self.final_k <= 0:
            raise ValueError(
                "final_k must be greater than 0"
            )

        if self.final_k > self.candidate_k:
            raise ValueError(
                "final_k cannot be greater than candidate_k"
            )

        if self.cache_ttl_seconds < 0:
            raise ValueError(
                "cache_ttl_seconds cannot be negative"
            )

        if self.cache_max_entries <= 0:
            raise ValueError(
                "cache_max_entries must be greater than 0"
            )

        if self.request_timeout_seconds <= 0:
            raise ValueError(
                "request_timeout_seconds must be greater than 0"
            )

        if self.retry_attempts < 0:
            raise ValueError(
                "retry_attempts cannot be negative"
            )

        if self.max_query_length <= 0:
            raise ValueError(
                "max_query_length must be greater than 0"
            )


def _get_bool(
    name: str,
    default: bool,
) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} must be a boolean value"
    )


def load_production_config() -> ProductionConfig:
    """
    Load configuration from environment variables.

    Environment variables:

        RAG_ENVIRONMENT
        RAG_CANDIDATE_K
        RAG_FINAL_K
        RAG_CACHE_ENABLED
        RAG_CACHE_TTL_SECONDS
        RAG_CACHE_MAX_ENTRIES
        RAG_REQUEST_TIMEOUT_SECONDS
        RAG_RETRY_ATTEMPTS
        RAG_MAX_QUERY_LENGTH
        RAG_REQUIRE_COMPANY
    """

    return ProductionConfig(
        environment=os.getenv(
            "RAG_ENVIRONMENT",
            "development",
        ),
        candidate_k=int(
            os.getenv(
                "RAG_CANDIDATE_K",
                "20",
            )
        ),
        final_k=int(
            os.getenv(
                "RAG_FINAL_K",
                "5",
            )
        ),
        cache_enabled=_get_bool(
            "RAG_CACHE_ENABLED",
            True,
        ),
        cache_ttl_seconds=int(
            os.getenv(
                "RAG_CACHE_TTL_SECONDS",
                "300",
            )
        ),
        cache_max_entries=int(
            os.getenv(
                "RAG_CACHE_MAX_ENTRIES",
                "1000",
            )
        ),
        request_timeout_seconds=float(
            os.getenv(
                "RAG_REQUEST_TIMEOUT_SECONDS",
                "30",
            )
        ),
        retry_attempts=int(
            os.getenv(
                "RAG_RETRY_ATTEMPTS",
                "2",
            )
        ),
        max_query_length=int(
            os.getenv(
                "RAG_MAX_QUERY_LENGTH",
                "2000",
            )
        ),
        require_company=_get_bool(
            "RAG_REQUIRE_COMPANY",
            False,
        ),
    )