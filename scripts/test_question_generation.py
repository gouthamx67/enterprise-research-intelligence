from dataclasses import dataclass
from types import SimpleNamespace

from src.rag_engine.app.question import (
    ContextResult,
    DocumentQuestionService,
)
from src.rag_engine.generation.models import GenerationResult
from src.rag_engine.generation.provider import LLMProvider


@dataclass
class FakeRecord:
    document_id: str = "doc-1"
    filename: str = "test.pdf"
    source_type: str = "pdf"
    size_bytes: int = 100
    stored_path: str = "/tmp/test.pdf"
    uploaded_at: str = "2026-01-01T00:00:00"
    status: str = "ready"


class FakeStorage:
    def load_record(self, document_id: str):
        return FakeRecord(document_id=document_id)


class FakeProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def generate(self, request):
        return GenerationResult(
            answer=(
                "The product strategy changed toward "
                "enterprise customers [C1]."
            ),
            citation_ids=["C1"],
            provider=self.provider_name,
            model=self.model_name,
        )


class FakeGenerationService:
    def __init__(self):
        self.provider = FakeProvider()
        self.called = False

    def generate(self, question, contexts):
        self.called = True

        return self.provider.generate(
            SimpleNamespace(
                question=question,
                contexts=contexts,
            )
        )


class QuestionServiceForTest(DocumentQuestionService):
    def _load_chunks(self, document_id):
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                text=(
                    "The company shifted its product strategy "
                    "toward enterprise customers."
                ),
            )
        ]

    def _load_dense_index(self, document_id):
        return SimpleNamespace()

    def _hybrid_retrieve(
        self,
        question,
        chunks,
        dense_index,
    ):
        return [
            SimpleNamespace(
                chunk_id="chunk-1",
                score=0.95,
                rank=1,
                text=(
                    "The company shifted its product strategy "
                    "toward enterprise customers."
                ),
                metadata={
                    "source": "test.pdf",
                    "start_page": 3,
                },
            )
        ]

    def _rerank(
        self,
        question,
        results,
        final_k,
    ):
        return results

    def _build_context(self, results):
        return [
            ContextResult(
                citation_id="C1",
                result=SimpleNamespace(
                    chunk_id="chunk-1",
                    score=0.95,
                    rank=1,
                    text=(
                        "The company shifted its product strategy "
                        "toward enterprise customers."
                    ),
                    metadata={
                        "source": "test.pdf",
                        "start_page": 3,
                    },
                ),
            )
        ]


def test_llm_generation_is_integrated() -> None:
    generation_service = FakeGenerationService()

    service = QuestionServiceForTest(
        storage=FakeStorage(),
        generation_service=generation_service,
    )

    result = service.ask(
        document_id="doc-1",
        question="What changed in the product strategy?",
    )

    assert generation_service.called is True

    assert result["answer"].startswith(
        "The product strategy changed toward enterprise customers"
    )

    assert result["metadata"]["generation_mode"] == "llm"
    assert result["metadata"]["generation_provider"] == "fake"
    assert result["metadata"]["generation_model"] == "fake-model"

    assert result["metadata"]["citation_valid"] is True

    print("Citation assignment before generation: passed")
    print("LLM provider integration: passed")
    print("Citation validation: passed")


def test_default_mode_remains_extractive() -> None:
    service = QuestionServiceForTest(
        storage=FakeStorage(),
    )

    result = service.ask(
        document_id="doc-1",
        question="What changed in the product strategy?",
    )

    assert result["metadata"]["generation_mode"] == "extractive"
    assert result["answer"]
    assert result["metadata"]["citation_valid"] is True

    print("Extractive fallback: passed")


def test_context_conversion_preserves_citation_metadata() -> None:
    service = QuestionServiceForTest(
        storage=FakeStorage(),
    )

    contexts = [
        ContextResult(
            citation_id="C7",
            result=SimpleNamespace(
                chunk_id="chunk-7",
                score=0.88,
                rank=2,
                text="Example evidence.",
                metadata={
                    "source": "filing.pdf",
                    "start_page": 12,
                    "section": "Strategy",
                },
            ),
        )
    ]

    generation_contexts = service._build_generation_contexts(
        contexts
    )

    assert len(generation_contexts) == 1

    context = generation_contexts[0]

    assert context.citation_id == "C7"
    assert context.text == "Example evidence."
    assert context.source == "filing.pdf"
    assert context.page == 12
    assert context.metadata["section"] == "Strategy"


if __name__ == "__main__":
    test_llm_generation_is_integrated()
    test_default_mode_remains_extractive()
    test_context_conversion_preserves_citation_metadata()

    print("Question-generation integration tests passed.")
