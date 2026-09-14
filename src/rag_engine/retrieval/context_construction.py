from dataclasses import dataclass, field
from typing import Any

from src.rag_engine.retrieval.context_selection import (
    ContextSelectionResult,
    ContextSelector,
)
from src.rag_engine.retrieval.context_deduplication import (
    ContextDeduplicationResult,
    ContextDeduplicator,
)
from src.rag_engine.retrieval.context_compression import (
    ContextCompressionResult,
    ContextCompressor,
)
from src.rag_engine.retrieval.context_ordering import (
    ContextOrderer,
    ContextOrderingResult,
)
from src.rag_engine.retrieval.citation_mapping import (
    CitationMappingResult,
    CitationMapper,
)
from src.rag_engine.retrieval.source_attribution import (
    SourceAttributionResult,
    SourceAttributor,
)


@dataclass
class ConstructedContext:
    """
    Final context package ready for downstream
    LLM synthesis.
    """

    chunk_id: str
    citation_id: str
    text: str
    score: float
    rank: int
    source: str
    source_type: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class ContextConstructionResult:
    """
    Complete output of the context construction stage.
    """

    query: str

    selection: ContextSelectionResult

    deduplication: ContextDeduplicationResult

    compression: ContextCompressionResult

    ordering: ContextOrderingResult

    citations: CitationMappingResult

    attribution: SourceAttributionResult

    contexts: list[ConstructedContext] = field(
        default_factory=list
    )

    @property
    def final_count(self) -> int:
        return len(self.contexts)

    @property
    def total_characters(self) -> int:
        return sum(
            len(context.text)
            for context in self.contexts
        )


class ContextConstructionPipeline:
    """
    Complete context construction pipeline.

    Stages:

        selection
        → deduplication
        → compression
        → ordering
        → citation mapping
        → source attribution
    """

    def __init__(
        self,
        max_chunks: int = 5,
        max_characters: int | None = None,
        max_sentences_per_chunk: int = 2,
        min_relevance_score: float = 0.0,
        ordering_strategy: str = "rank_order",
    ):
        self.selector = ContextSelector(
            max_chunks=max_chunks,
            max_characters=max_characters,
        )

        self.deduplicator = (
            ContextDeduplicator()
        )

        self.compressor = ContextCompressor(
            max_sentences_per_chunk=(
                max_sentences_per_chunk
            ),
            min_relevance_score=(
                min_relevance_score
            ),
        )

        self.orderer = ContextOrderer()

        self.citation_mapper = CitationMapper()

        self.source_attributor = (
            SourceAttributor()
        )

        self.ordering_strategy = (
            ordering_strategy
        )

        if ordering_strategy not in (
            self.orderer.SUPPORTED_STRATEGIES
        ):
            raise ValueError(
                f"unsupported ordering strategy: "
                f"{ordering_strategy!r}"
            )

    def build(
        self,
        query: str,
        candidates,
    ) -> ContextConstructionResult:

        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if candidates is None:
            raise ValueError(
                "candidates cannot be None"
            )

        query = query.strip()

        # -----------------------------------------
        # 1. Selection
        # -----------------------------------------

        selection = self.selector.select(
            query=query,
            candidates=candidates,
        )

        # -----------------------------------------
        # 2. Deduplication
        # -----------------------------------------

        deduplication = (
            self.deduplicator.deduplicate(
                selection.selected
            )
        )

        # -----------------------------------------
        # 3. Compression
        # -----------------------------------------

        compression = (
            self.compressor.compress(
                query=query,
                contexts=deduplication.selected,
            )
        )

        # -----------------------------------------
        # 4. Ordering
        # -----------------------------------------

        ordering = self.orderer.order(
            compression.contexts,
            strategy=self.ordering_strategy,
        )

        # -----------------------------------------
        # 5. Citation mapping
        # -----------------------------------------

        citations = self.citation_mapper.map(
            ordering.contexts
        )

        # -----------------------------------------
        # 6. Source attribution
        # -----------------------------------------

        attribution = (
            self.source_attributor.attribute(
                ordering.contexts
            )
        )

        # -----------------------------------------
        # 7. Build final context package
        # -----------------------------------------

        contexts = []

        for index, context in enumerate(
            ordering.contexts
        ):
            citation = citations.citations[
                index
            ]

            contexts.append(
                ConstructedContext(
                    chunk_id=context.chunk_id,
                    citation_id=citation.citation_id,
                    text=context.text,
                    score=context.score,
                    rank=context.rank,
                    source=citation.source,
                    source_type=(
                        citation.source_type
                    ),
                    metadata=dict(
                        context.metadata
                    ),
                )
            )

        return ContextConstructionResult(
            query=query,
            selection=selection,
            deduplication=deduplication,
            compression=compression,
            ordering=ordering,
            citations=citations,
            attribution=attribution,
            contexts=contexts,
        )


def build_context(
    query: str,
    candidates,
    max_chunks: int = 5,
    max_characters: int | None = None,
    max_sentences_per_chunk: int = 2,
    min_relevance_score: float = 0.0,
    ordering_strategy: str = "rank_order",
) -> ContextConstructionResult:
    """
    Convenience function for one complete context
    construction run.
    """

    pipeline = ContextConstructionPipeline(
        max_chunks=max_chunks,
        max_characters=max_characters,
        max_sentences_per_chunk=(
            max_sentences_per_chunk
        ),
        min_relevance_score=(
            min_relevance_score
        ),
        ordering_strategy=ordering_strategy,
    )

    return pipeline.build(
        query=query,
        candidates=candidates,
    )