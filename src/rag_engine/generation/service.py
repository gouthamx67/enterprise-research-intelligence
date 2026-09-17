from src.rag_engine.generation.models import (
    GenerationContext,
    GenerationRequest,
    GenerationResult,
)
from src.rag_engine.generation.provider import LLMProvider


class GenerationService:
    """
    Application-level generation service.

    This class owns the provider-independent generation workflow.
    """

    def __init__(
        self,
        provider: LLMProvider,
    ) -> None:
        self.provider = provider

    def generate(
        self,
        question: str,
        contexts: list[GenerationContext],
    ) -> GenerationResult:
        request = GenerationRequest(
            question=question,
            contexts=contexts,
        )

        return self.provider.generate(request)