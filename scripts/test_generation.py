from dataclasses import dataclass

from src.rag_engine.generation.models import (
    GenerationContext,
    GenerationRequest,
)
from src.rag_engine.generation.openai_provider import (
    OpenAIProvider,
)
from src.rag_engine.generation.prompts import (
    SYSTEM_PROMPT,
    build_user_prompt,
)
from src.rag_engine.generation.service import (
    GenerationService,
)


@dataclass
class FakeMessage:
    content: str


@dataclass
class FakeChoice:
    message: FakeMessage


@dataclass
class FakeResponse:
    choices: list[FakeChoice]


class FakeCompletions:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs

        return FakeResponse(
            choices=[
                FakeChoice(
                    message=FakeMessage(
                        content=(
                            "The company shifted its strategy "
                            "toward enterprise products [C1] "
                            "while reducing emphasis on the "
                            "legacy offering [C2]."
                        )
                    )
                )
            ]
        )


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletions()


class FakeClient:
    def __init__(self) -> None:
        self.chat = FakeChat()


def test_generation_models() -> None:
    context = GenerationContext(
        citation_id="C1",
        text="The company launched a new enterprise product.",
        source="earnings-transcript.pdf",
        page=4,
    )

    request = GenerationRequest(
        question="What changed in product strategy?",
        contexts=[context],
    )

    assert request.question == (
        "What changed in product strategy?"
    )
    assert len(request.contexts) == 1
    assert request.contexts[0].citation_id == "C1"


def test_prompt_contains_question_and_evidence() -> None:
    request = GenerationRequest(
        question="What changed?",
        contexts=[
            GenerationContext(
                citation_id="C1",
                text="The company launched an enterprise tier.",
                source="filing.pdf",
                page=10,
            ),
        ],
    )

    prompt = build_user_prompt(request)

    assert "What changed?" in prompt
    assert "[C1]" in prompt
    assert "The company launched an enterprise tier." in prompt
    assert "filing.pdf" in prompt
    assert "page 10" in prompt


def test_system_prompt_requires_grounding() -> None:
    assert "ONLY the evidence" in SYSTEM_PROMPT
    assert "[C1]" in SYSTEM_PROMPT
    assert "Do not invent facts" in SYSTEM_PROMPT


def test_openai_provider_uses_injected_client() -> None:
    client = FakeClient()

    provider = OpenAIProvider(
        model="test-model",
        client=client,
    )

    request = GenerationRequest(
        question="What changed?",
        contexts=[
            GenerationContext(
                citation_id="C1",
                text="The company launched an enterprise tier.",
                source="filing.pdf",
            ),
        ],
    )

    result = provider.generate(request)

    assert result.answer.startswith(
        "The company shifted its strategy"
    )
    assert result.provider == "openai-compatible"
    assert result.model == "test-model"
    assert result.citation_ids == ["C1", "C2"]

    kwargs = client.chat.completions.last_kwargs

    assert kwargs["model"] == "test-model"
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_completion_tokens"] == 800
    assert len(kwargs["messages"]) == 2
    assert kwargs["messages"][0]["role"] == "system"
    assert kwargs["messages"][1]["role"] == "user"


def test_citation_extraction_is_unique_and_ordered() -> None:
    answer = (
        "Claim one [c2]. "
        "Claim two [C1]. "
        "Claim three [C2]. "
        "Claim four [C3]."
    )

    citations = OpenAIProvider.extract_citation_ids(
        answer
    )

    assert citations == ["C2", "C1", "C3"]


def test_generation_service() -> None:
    client = FakeClient()

    provider = OpenAIProvider(
        model="test-model",
        client=client,
    )

    service = GenerationService(provider)

    result = service.generate(
        question="What changed?",
        contexts=[
            GenerationContext(
                citation_id="C1",
                text="Product strategy changed.",
            ),
        ],
    )

    assert result.provider == "openai-compatible"
    assert result.model == "test-model"
    assert result.citation_ids == ["C1", "C2"]


def main() -> None:
    test_generation_models()
    test_prompt_contains_question_and_evidence()
    test_system_prompt_requires_grounding()
    test_openai_provider_uses_injected_client()
    test_citation_extraction_is_unique_and_ordered()
    test_generation_service()

    print("Generation tests passed.")
    print("Provider: openai-compatible")
    print("Model: test-model")
    print("Citation extraction: passed")
    print("Fake LLM request: passed")


if __name__ == "__main__":
    main()