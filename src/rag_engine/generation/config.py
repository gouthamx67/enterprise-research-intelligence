import os
from dataclasses import dataclass


class LLMConfigurationError(ValueError):
    """Raised when LLM configuration is invalid."""


@dataclass(frozen=True)
class LLMConfig:
    """
    Runtime configuration for LLM generation.

    Environment variables:

        RAG_LLM_ENABLED
        RAG_LLM_PROVIDER
        RAG_LLM_MODEL
        OPENAI_API_KEY
        RAG_LLM_BASE_URL
        RAG_LLM_TEMPERATURE
        RAG_LLM_MAX_TOKENS
    """

    enabled: bool = False
    provider: str = "openai-compatible"
    model: str = "gpt-5.6-luna"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int = 800

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        enabled = _read_bool(
            "RAG_LLM_ENABLED",
            default=False,
        )

        provider = os.getenv(
            "RAG_LLM_PROVIDER",
            "openai-compatible",
        ).strip().lower()

        model = os.getenv(
            "RAG_LLM_MODEL",
            "gpt-5.6-luna",
        ).strip()

        api_key = os.getenv(
            "OPENAI_API_KEY",
        )

        if api_key is not None:
            api_key = api_key.strip() or None

        base_url = os.getenv(
            "RAG_LLM_BASE_URL",
        )

        if base_url is not None:
            base_url = base_url.strip() or None

        temperature = _read_float(
            "RAG_LLM_TEMPERATURE",
            default=0.0,
        )

        max_tokens = _read_int(
            "RAG_LLM_MAX_TOKENS",
            default=800,
        )

        config = cls(
            enabled=enabled,
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        config.validate()

        return config

    def validate(self) -> None:
        if self.provider != "openai-compatible":
            raise LLMConfigurationError(
                "Unsupported LLM provider: "
                f"{self.provider}"
            )

        if not self.model:
            raise LLMConfigurationError(
                "RAG_LLM_MODEL cannot be empty."
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise LLMConfigurationError(
                "RAG_LLM_TEMPERATURE must be between 0 and 2."
            )

        if self.max_tokens <= 0:
            raise LLMConfigurationError(
                "RAG_LLM_MAX_TOKENS must be greater than 0."
            )

        if self.enabled and not self.api_key:
            raise LLMConfigurationError(
                "RAG_LLM_ENABLED is true, but "
                "OPENAI_API_KEY is not configured."
            )


def _read_bool(
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

    raise LLMConfigurationError(
        f"{name} must be a boolean value."
    )


def _read_float(
    name: str,
    default: float,
) -> float:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"{name} must be a number."
        ) from exc


def _read_int(
    name: str,
    default: int,
) -> int:
    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as exc:
        raise LLMConfigurationError(
            f"{name} must be an integer."
        ) from exc
