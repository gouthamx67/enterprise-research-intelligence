from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class CompressedContext:
    """
    A compressed version of a context item.

    The selected sentences are copied from the original
    context rather than newly generated.
    """

    chunk_id: str
    text: str
    score: float
    rank: int
    original_text: str
    original_character_count: int
    compressed_character_count: int
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextCompressionResult:
    """
    Result of compressing a collection of context items.
    """

    original_count: int
    compressed_count: int
    original_characters: int
    compressed_characters: int
    contexts: list[CompressedContext] = field(
        default_factory=list
    )

    @property
    def compression_ratio(self) -> float:
        """
        Fraction of the original text retained.

        Example:
            0.50 means 50% of the original characters
            remain after compression.
        """

        if self.original_characters == 0:
            return 0.0

        return (
            self.compressed_characters
            / self.original_characters
        )


def split_sentences(text: str) -> list[str]:
    """
    Split text into simple sentence units.

    This is intentionally deterministic and lightweight.
    More advanced sentence segmentation can be introduced
    later.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    text = text.strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def tokenize_for_relevance(text: str) -> set[str]:
    """
    Convert text into lowercase word tokens.

    Punctuation is ignored.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    return set(
        re.findall(
            r"\b\w+\b",
            text.lower(),
        )
    )


def sentence_relevance_score(
    query: str,
    sentence: str,
) -> float:
    """
    Calculate a simple lexical relevance score.

    Score = number of query terms appearing in the
    sentence divided by the number of unique query terms.
    """

    if not query or not query.strip():
        raise ValueError(
            "query cannot be empty"
        )

    if not sentence or not sentence.strip():
        return 0.0

    query_terms = tokenize_for_relevance(
        query
    )

    sentence_terms = tokenize_for_relevance(
        sentence
    )

    if not query_terms:
        return 0.0

    overlap = (
        query_terms
        & sentence_terms
    )

    return len(overlap) / len(query_terms)


class ContextCompressor:
    """
    Extractive context compressor.

    For each context item:
      1. Split into sentences.
      2. Score sentences against the query.
      3. Keep the most relevant sentences.
      4. Restore their original document order.
    """

    def __init__(
        self,
        max_sentences_per_chunk: int = 2,
        min_relevance_score: float = 0.0,
    ):
        if max_sentences_per_chunk <= 0:
            raise ValueError(
                "max_sentences_per_chunk must "
                "be greater than 0"
            )

        if not (
            0.0
            <= min_relevance_score
            <= 1.0
        ):
            raise ValueError(
                "min_relevance_score must be "
                "between 0 and 1"
            )

        self.max_sentences_per_chunk = (
            max_sentences_per_chunk
        )

        self.min_relevance_score = (
            min_relevance_score
        )

    def compress(
        self,
        query: str,
        contexts,
    ) -> ContextCompressionResult:

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if contexts is None:
            raise ValueError(
                "contexts cannot be None"
            )

        compressed_contexts = []

        original_characters = 0
        compressed_characters = 0

        for context in contexts:

            original_text = getattr(
                context,
                "text",
                None,
            )

            if not isinstance(
                original_text,
                str,
            ):
                continue

            original_text = (
                original_text.strip()
            )

            if not original_text:
                continue

            original_characters += len(
                original_text
            )

            sentences = split_sentences(
                original_text
            )

            if not sentences:
                continue

            scored_sentences = []

            for index, sentence in enumerate(
                sentences
            ):
                score = (
                    sentence_relevance_score(
                        query,
                        sentence,
                    )
                )

                if (
                    score
                    >= self.min_relevance_score
                ):
                    scored_sentences.append(
                        (
                            score,
                            index,
                            sentence,
                        )
                    )

            if not scored_sentences:
                continue

            # Select highest-scoring sentences.
            scored_sentences.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            selected = scored_sentences[
                : self.max_sentences_per_chunk
            ]

            # Restore original sentence order.
            selected.sort(
                key=lambda item: item[1]
            )

            compressed_text = " ".join(
                item[2]
                for item in selected
            )

            compressed_character_count = len(
                compressed_text
            )

            compressed_characters += (
                compressed_character_count
            )

            compressed_contexts.append(
                CompressedContext(
                    chunk_id=context.chunk_id,
                    text=compressed_text,
                    score=float(
                        context.score
                    ),
                    rank=int(
                        context.rank
                    ),
                    original_text=original_text,
                    original_character_count=len(
                        original_text
                    ),
                    compressed_character_count=(
                        compressed_character_count
                    ),
                    metadata=dict(
                        getattr(
                            context,
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        return ContextCompressionResult(
            original_count=len(contexts),
            compressed_count=len(
                compressed_contexts
            ),
            original_characters=(
                original_characters
            ),
            compressed_characters=(
                compressed_characters
            ),
            contexts=compressed_contexts,
        )


def compress_context(
    query: str,
    contexts,
    max_sentences_per_chunk: int = 2,
    min_relevance_score: float = 0.0,
) -> ContextCompressionResult:
    """
    Convenience function for one-off compression.
    """

    compressor = ContextCompressor(
        max_sentences_per_chunk=(
            max_sentences_per_chunk
        ),
        min_relevance_score=(
            min_relevance_score
        ),
    )

    return compressor.compress(
        query=query,
        contexts=contexts,
    )