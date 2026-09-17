import os

from src.rag_engine.generation.config import LLMConfig
from src.rag_engine.generation.models import GenerationContext
from src.rag_engine.generation.openai_provider import OpenAIProvider
from src.rag_engine.generation.service import GenerationService
from src.rag_engine.pipeline.citation_validator import CitationValidator


def main() -> None:
    config = LLMConfig.from_environment()

    if not config.enabled:
        print(
            "Real LLM smoke test skipped: "
            "RAG_LLM_ENABLED is not true."
        )
        return

    if not config.api_key:
        raise RuntimeError(
            "RAG_LLM_ENABLED is true, but OPENAI_API_KEY is missing."
        )

    provider = OpenAIProvider(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    service = GenerationService(provider=provider)

    contexts = [
        GenerationContext(
            citation_id="C1",
            text=(
                "The company introduced an enterprise-focused "
                "product tier during the past year."
            ),
            source="example-product-report.pdf",
            page=4,
        ),
        GenerationContext(
            citation_id="C2",
            text=(
                "The company expanded administrative controls, "
                "security features, and organization-level billing."
            ),
            source="example-product-report.pdf",
            page=7,
        ),
    ]

    result = service.generate(
        question=(
            "What changed in the company's product strategy "
            "over the last 12 months?"
        ),
        contexts=contexts,
    )

    validator = CitationValidator()
    validation = validator.validate(
        result.citation_ids,
        {"C1", "C2"},
    )

    print()
    print("Real LLM smoke test")
    print("-------------------")
    print(f"Provider:          {result.provider}")
    print(f"Model:             {result.model}")
    print(f"Citations:         {result.citation_ids}")
    print(f"Citations valid:   {validation.valid}")
    print()
    print("Answer:")
    print(result.answer)
    print()

    if not validation.valid:
        raise RuntimeError(
            "LLM returned invalid or missing citations."
        )

    print("Real LLM smoke test passed.")


if __name__ == "__main__":
    main()