from abc import ABC, abstractmethod

from src.rag_engine.generation.models import (
    GenerationRequest,
    GenerationResult,
)


class LLMProvider(ABC):
    """
    Provider-independent interface for language-model generation.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        """Generate an evidence-grounded answer."""