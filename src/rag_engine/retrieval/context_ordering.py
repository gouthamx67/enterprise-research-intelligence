from dataclasses import dataclass, field
from datetime import date
from typing import Any


@dataclass
class OrderedContext:
    """
    Context item after ordering.

    The actual evidence is unchanged.
    """

    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextOrderingResult:
    """
    Result of context ordering.
    """

    strategy: str
    original_count: int
    ordered_count: int
    contexts: list[OrderedContext] = field(
        default_factory=list
    )


def _get_source(context) -> str:
    metadata = getattr(
        context,
        "metadata",
        {},
    )

    source = metadata.get(
        "source",
        "",
    )

    return str(source)


def _get_document_date(context):
    metadata = getattr(
        context,
        "metadata",
        {},
    )

    value = metadata.get(
        "document_date"
    )

    if value is None:
        return date.min

    if isinstance(value, date):
        return value

    if isinstance(value, str):
        try:
            return date.fromisoformat(
                value
            )
        except ValueError:
            return date.min

    return date.min


class ContextOrderer:
    """
    Order context items according to an explicit strategy.

    Supported strategies:

        rank_order
        score_desc
        source_grouped
        chronological
    """

    SUPPORTED_STRATEGIES = {
        "rank_order",
        "score_desc",
        "source_grouped",
        "chronological",
    }

    def order(
        self,
        contexts,
        strategy: str = "rank_order",
    ) -> ContextOrderingResult:

        if contexts is None:
            raise ValueError(
                "contexts cannot be None"
            )

        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"unsupported ordering strategy: "
                f"{strategy!r}"
            )

        ordered = list(contexts)

        if strategy == "rank_order":
            ordered.sort(
                key=lambda item: item.rank
            )

        elif strategy == "score_desc":
            ordered.sort(
                key=lambda item: item.score,
                reverse=True,
            )

        elif strategy == "source_grouped":
            ordered.sort(
                key=lambda item: (
                    _get_source(item),
                    -item.score,
                )
            )

        elif strategy == "chronological":
            ordered.sort(
                key=lambda item: (
                    _get_document_date(item),
                    item.rank,
                )
            )

        result = []

        for context in ordered:
            result.append(
                OrderedContext(
                    chunk_id=context.chunk_id,
                    text=context.text,
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

        return ContextOrderingResult(
            strategy=strategy,
            original_count=len(contexts),
            ordered_count=len(result),
            contexts=result,
        )


def order_context(
    contexts,
    strategy: str = "rank_order",
) -> ContextOrderingResult:
    """
    Convenience function for one-off ordering.
    """

    return ContextOrderer().order(
        contexts,
        strategy=strategy,
    )