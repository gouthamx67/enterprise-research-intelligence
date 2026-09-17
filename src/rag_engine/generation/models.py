from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationContext:
    """
    Evidence supplied to the language model.

    citation_id is the identifier the model must use when
    attributing factual claims, for example [C1].
    """

    citation_id: str
    text: str
    source: str = ""
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.citation_id.strip():
            raise ValueError("citation_id cannot be empty")

        if not self.text.strip():
            raise ValueError("text cannot be empty")


@dataclass(frozen=True)
class GenerationRequest:
    """
    Complete request sent to an LLM provider.
    """

    question: str
    contexts: list[GenerationContext]

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question cannot be empty")

        if not self.contexts:
            raise ValueError("contexts cannot be empty")


@dataclass(frozen=True)
class GenerationResult:
    """
    Result returned by an LLM provider.
    """

    answer: str
    citation_ids: list[str]
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.answer.strip():
            raise ValueError("answer cannot be empty")

        if not self.provider.strip():
            raise ValueError("provider cannot be empty")

        if not self.model.strip():
            raise ValueError("model cannot be empty")