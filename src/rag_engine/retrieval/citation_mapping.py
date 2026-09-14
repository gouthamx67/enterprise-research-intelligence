from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CitationReference:
    """
    Stable reference to an evidence chunk.
    """

    citation_id: str
    chunk_id: str
    source: str
    source_type: str
    text: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class CitationMappingResult:
    """
    Mapping between context chunks and citation IDs.
    """

    citations: list[CitationReference] = field(
        default_factory=list
    )

    def get(
        self,
        citation_id: str,
    ) -> CitationReference | None:
        for citation in self.citations:
            if citation.citation_id == citation_id:
                return citation

        return None


class CitationMapper:
    """
    Create stable citation references for context items.

    Citation IDs are generated in context order:

        C1
        C2
        C3
        ...

    The original evidence and metadata are preserved.
    """

    def map(
        self,
        contexts,
    ) -> CitationMappingResult:

        if contexts is None:
            raise ValueError(
                "contexts cannot be None"
            )

        citations = []

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

            source = metadata.get(
                "source",
                "",
            )

            source_type = metadata.get(
                "source_type",
                "",
            )

            citations.append(
                CitationReference(
                    citation_id=f"C{index}",
                    chunk_id=context.chunk_id,
                    source=str(source),
                    source_type=str(
                        source_type
                    ),
                    text=context.text,
                    metadata=metadata,
                )
            )

        return CitationMappingResult(
            citations=citations
        )


def build_citation_mapping(
    contexts,
) -> CitationMappingResult:
    """
    Convenience function for one-off citation mapping.
    """

    return CitationMapper().map(
        contexts
    )