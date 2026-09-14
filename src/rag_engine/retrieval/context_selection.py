
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SelectedContext:
    """
    A single piece of evidence selected for LLM context.
    """

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextSelectionResult:
    """
    Result of selecting evidence for the LLM context.
    """

    query: str
    selected: list[SelectedContext] = field(
        default_factory=list
    )
    candidate_count: int = 0
    selected_count: int = 0
    total_characters: int = 0


class ContextSelector:
    """
    Select the highest-ranked retrieval/reranking
    results that fit the configured context limits.

    This stage does NOT:
      - rerank
      - deduplicate
      - compress
      - summarize

    Those responsibilities belong to later stages.
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_characters: int | None = None,
    ):
        if max_chunks <= 0:
            raise ValueError(
                "max_chunks must be greater than 0"
            )

        if (
            max_characters is not None
            and max_characters <= 0
        ):
            raise ValueError(
                "max_characters must be greater than 0"
            )

        self.max_chunks = max_chunks
        self.max_characters = max_characters

    def select(
        self,
        query: str,
        candidates,
    ) -> ContextSelectionResult:
        """
        Select candidates in their existing ranking order.

        Candidates are expected to expose:
          - chunk_id
          - text
          - score
          - rank
          - metadata
        """

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if candidates is None:
            raise ValueError(
                "candidates cannot be None"
            )

        query = query.strip()

        selected = []
        total_characters = 0

        for candidate in candidates:

            if len(selected) >= self.max_chunks:
                break

            text = getattr(candidate, "text", "")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            if self.max_characters is not None:
                next_total = (
                    total_characters
                    + len(text)
                )

                if next_total > self.max_characters:
                    break

            selected.append(
                SelectedContext(
                    chunk_id=candidate.chunk_id,
                    text=text,
                    score=float(candidate.score),
                    rank=int(candidate.rank),
                    metadata=dict(
                        getattr(
                            candidate,
                            "metadata",
                            {},
                        )
                    ),
                )
            )

            total_characters += len(text)

        return ContextSelectionResult(
            query=query,
            selected=selected,
            candidate_count=len(candidates),
            selected_count=len(selected),
            total_characters=total_characters,
        )


def select_context(
    query: str,
    candidates,
    max_chunks: int = 5,
    max_characters: int | None = None,
) -> ContextSelectionResult:
    """
    Convenience function for one-off context selection.
    """

    selector = ContextSelector(
        max_chunks=max_chunks,
        max_characters=max_characters,
    )

    return selector.select(
        query=query,
        candidates=candidates,
    )