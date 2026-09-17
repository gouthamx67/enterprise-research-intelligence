import re
from typing import Any

from openai import OpenAI

from src.rag_engine.generation.models import (
    GenerationRequest,
    GenerationResult,
)
from src.rag_engine.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.rag_engine.generation.provider import LLMProvider


_CITATION_PATTERN = re.compile(r"\[(C\d+)\]", re.IGNORECASE)


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        model: str = "gpt-5.6-luna",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 800,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")

        if not 0.0 <= temperature <= 2.0:
            raise ValueError(
                "temperature must be between 0 and 2"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be greater than 0"
            )

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        if client is not None:
            self._client = client
            return

        client_kwargs: dict[str, Any] = {}

        if api_key:
            client_kwargs["api_key"] = api_key

        if base_url:
            client_kwargs["base_url"] = base_url

        self._client = OpenAI(**client_kwargs)

    @property
    def provider_name(self) -> str:
        return "openai-compatible"

    @property
    def model_name(self) -> str:
        return self._model

    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(request),
                },
            ],
            temperature=self._temperature,
            max_completion_tokens=self._max_tokens,
        )

        answer = self._extract_answer(response)
        citation_ids = self.extract_citation_ids(answer)

        return GenerationResult(
            answer=answer,
            citation_ids=citation_ids,
            provider=self.provider_name,
            model=self.model_name,
            metadata={
                "temperature": self._temperature,
                "max_completion_tokens": self._max_tokens,
            },
        )

    @staticmethod
    def extract_citation_ids(
        answer: str,
    ) -> list[str]:
        citations: list[str] = []
        seen: set[str] = set()

        for match in _CITATION_PATTERN.finditer(answer):
            citation_id = match.group(1).upper()

            if citation_id in seen:
                continue

            seen.add(citation_id)
            citations.append(citation_id)

        return citations

    @staticmethod
    def _extract_answer(response: Any) -> str:
        choices = getattr(response, "choices", None)

        if not choices:
            raise RuntimeError(
                "LLM response did not contain any choices."
            )

        message = getattr(
            choices[0],
            "message",
            None,
        )

        if message is None:
            raise RuntimeError(
                "LLM response did not contain a message."
            )

        content = getattr(
            message,
            "content",
            None,
        )

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(
                "LLM response contained no text content."
            )

        return content.strip()
