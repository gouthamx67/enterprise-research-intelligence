from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceAttribution:
    """
    Human-readable attribution for one evidence item.
    """

    citation_id: str
    chunk_id: str
    source: str
    source_type: str
    title: str | None = None
    company: str | None = None
    document_date: str | None = None
    start_page: int | None = None
    end_page: int | None = None
    section_title: str | None = None
    block_numbers: tuple[int, ...] = ()
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class SourceAttributionResult:
    """
    Attribution records for the complete context.
    """

    attributions: list[SourceAttribution] = field(
        default_factory=list
    )

    def get(
        self,
        citation_id: str,
    ) -> SourceAttribution | None:
        for attribution in self.attributions:
            if attribution.citation_id == citation_id:
                return attribution

        return None


class SourceAttributor:
    """
    Build source attribution records from context items.

    Metadata is read without destroying the original
    metadata dictionary.
    """

    def attribute(
        self,
        contexts,
    ) -> SourceAttributionResult:

        if contexts is None:
            raise ValueError(
                "contexts cannot be None"
            )

        attributions = []

        for index, context in enumerate(
            contexts,
            start=1,
        ):
            metadata = dict(
                getattr(
                    context,
                    "metadata",
                    {},
                )
            )

            block_numbers = metadata.get(
                "block_numbers",
                [],
            )

            if block_numbers is None:
                block_numbers = []

            attributions.append(
                SourceAttribution(
                    citation_id=f"C{index}",
                    chunk_id=context.chunk_id,
                    source=str(
                        metadata.get(
                            "source",
                            "",
                        )
                    ),
                    source_type=str(
                        metadata.get(
                            "source_type",
                            "",
                        )
                    ),
                    title=metadata.get(
                        "title"
                    ),
                    company=metadata.get(
                        "company"
                    ),
                    document_date=metadata.get(
                        "document_date"
                    ),
                    start_page=metadata.get(
                        "start_page"
                    ),
                    end_page=metadata.get(
                        "end_page"
                    ),
                    section_title=metadata.get(
                        "section_title"
                    ),
                    block_numbers=tuple(
                        block_numbers
                    ),
                    metadata=metadata,
                )
            )

        return SourceAttributionResult(
            attributions=attributions
        )


def attribute_sources(
    contexts,
) -> SourceAttributionResult:
    """
    Convenience function for one-off source attribution.
    """

    return SourceAttributor().attribute(
        contexts
    )