import os

from src.rag_engine.generation.config import (
    LLMConfig,
    LLMConfigurationError,
)


LLM_ENV_VARS = (
    "RAG_LLM_ENABLED",
    "RAG_LLM_PROVIDER",
    "RAG_LLM_MODEL",
    "OPENAI_API_KEY",
    "RAG_LLM_BASE_URL",
    "RAG_LLM_TEMPERATURE",
    "RAG_LLM_MAX_TOKENS",
)


def clear_llm_environment() -> None:
    for name in LLM_ENV_VARS:
        os.environ.pop(name, None)


def test_default_configuration(monkeypatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    config = LLMConfig.from_environment()

    assert config.enabled is False
    assert config.provider == "openai-compatible"
    assert config.model == "gpt-5.6-luna"
    assert config.api_key is None
    assert config.temperature == 0.0
    assert config.max_tokens == 800

    print("Default configuration: passed")


def test_enabled_configuration(monkeypatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv(
        "RAG_LLM_ENABLED",
        "true",
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY",
        "test-key",
    )
    monkeypatch.setenv(
        "RAG_LLM_MODEL",
        "test-model",
    )
    monkeypatch.setenv(
        "RAG_LLM_TEMPERATURE",
        "0.2",
    )
    monkeypatch.setenv(
        "RAG_LLM_MAX_TOKENS",
        "500",
    )

    config = LLMConfig.from_environment()

    assert config.enabled is True
    assert config.api_key == "test-key"
    assert config.model == "test-model"
    assert config.temperature == 0.2
    assert config.max_tokens == 500

    print("Enabled configuration: passed")


def test_enabled_without_api_key_fails(monkeypatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv(
        "RAG_LLM_ENABLED",
        "true",
    )

    try:
        LLMConfig.from_environment()
    except LLMConfigurationError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError(
            "Expected LLMConfigurationError"
        )

    print("Missing API key validation: passed")


def test_invalid_temperature_fails(monkeypatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv(
        "RAG_LLM_TEMPERATURE",
        "3.0",
    )

    try:
        LLMConfig.from_environment()
    except LLMConfigurationError:
        pass
    else:
        raise AssertionError(
            "Expected LLMConfigurationError"
        )

    print("Temperature validation: passed")


def test_invalid_provider_fails(monkeypatch) -> None:
    for name in LLM_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv(
        "RAG_LLM_PROVIDER",
        "unknown-provider",
    )

    try:
        LLMConfig.from_environment()
    except LLMConfigurationError:
        pass
    else:
        raise AssertionError(
            "Expected LLMConfigurationError"
        )

    print("Provider validation: passed")


if __name__ == "__main__":
    clear_llm_environment()

    try:
        # Standalone execution does not have pytest's monkeypatch fixture,
        # so use explicit environment cleanup between tests.
        test_default_configuration.__wrapped__()  # type: ignore[attr-defined]
    except AttributeError:
        # The focused standalone execution is intentionally delegated to
        # pytest because these tests use the monkeypatch fixture.
        print(
            "Run this test with: "
            "PYTHONPATH=. pytest -q scripts/test_llm_configuration.py"
        )
