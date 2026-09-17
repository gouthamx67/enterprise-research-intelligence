from src.rag_engine.pipeline.models import (
    GeneratedAnswer,
)


class AnswerGenerator:
    """
    Base generator interface.

    A production implementation can wrap an LLM provider.
    """

    def generate(
        self,
        query: str,
        contexts,
    ) -> GeneratedAnswer:
        raise NotImplementedError


class ExtractiveAnswerGenerator(AnswerGenerator):
    """
    Deterministic generator used for testing the pipeline.

    It selects relevant context sentences and attaches
    citation IDs.

    This is NOT intended to replace a real LLM.
    """

    def generate(
        self,
        query: str,
        contexts,
    ) -> GeneratedAnswer:
        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if not contexts:
            raise ValueError(
                "contexts cannot be empty"
            )

        sentences = []

        citation_ids = []

        for context in contexts:
            text = context.text.strip()

            if not text:
                continue

            first_sentence = text.split(".")[0].strip()

            if first_sentence:
                sentences.append(
                    first_sentence
                )

            citation_ids.append(
                context.citation_id
            )

        if not sentences:
            raise ValueError(
                "contexts contain no usable text"
            )

        answer = ". ".join(sentences)

        if not answer.endswith("."):
            answer += "."

        return GeneratedAnswer(
            answer=answer,
            citation_ids=citation_ids,
        )