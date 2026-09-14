from dataclasses import dataclass, field
import re
from typing import Any


@dataclass
class DeduplicatedContext:
    """
    A context item that survived deduplication.
    """

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextDeduplicationResult:
    """
    Result of context deduplication.
    """

    original_count: int
    deduplicated_count: int
    removed_count: int
    selected: list[DeduplicatedContext] = field(
        default_factory=list
    )


def normalize_for_deduplication(text: str) -> str:
    """
    Normalize text for duplicate comparison.

    This does not modify the actual context text.
    It only creates a comparison representation.
    """

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    normalized = text.strip().lower()

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    )

    return normalized


class ContextDeduplicator:
    """
    Remove exact and normalized duplicate context items.

    The first occurrence is retained because candidates
    are expected to already be ordered by relevance.
    """

    def deduplicate(
        self,
        contexts,
    ) -> ContextDeduplicationResult:

        if contexts is None:
            raise ValueError(
                "contexts cannot be None"
            )

        seen = set()
        deduplicated = []

        for context in contexts:
            text = getattr(
                context,
                "text",
                None,
            )

            if not isinstance(text, str):
                continue

            normalized_text = (
                normalize_for_deduplication(text)
            )

            if not normalized_text:
                continue

            if normalized_text in seen:
                continue

            seen.add(normalized_text)

            deduplicated.append(
                DeduplicatedContext(
                    chunk_id=context.chunk_id,
                    text=text.strip(),
                    score=float(context.score),
                    rank=int(context.rank),
                    metadata=dict(
                        getattr(
                            context,
                            "metadata",
                            {},
                        )
                    ),
                )
            )

        return ContextDeduplicationResult(
            original_count=len(contexts),
            deduplicated_count=len(
                deduplicated
            ),
            removed_count=(
                len(contexts)
                - len(deduplicated)
            ),
            selected=deduplicated,
        )


def deduplicate_context(
    contexts,
) -> ContextDeduplicationResult:
    """
    Convenience function for one-off deduplication.
    """

    deduplicator = ContextDeduplicator()

    return deduplicator.deduplicate(
        contexts
    )