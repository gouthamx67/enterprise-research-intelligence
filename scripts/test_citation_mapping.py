from dataclasses import dataclass

from src.rag_engine.retrieval.citation_mapping import (
    CitationMapper,
    CitationMappingResult,
    CitationReference,
    build_citation_mapping,
)


@dataclass
class FakeContext:
    chunk_id: str
    text: str
    score: float
    rank: int
    metadata: dict


def build_contexts():
    return [
        FakeContext(
            chunk_id="chunk-001",
            text=(
                "The company expanded its "
                "enterprise product strategy."
            ),
            score=0.95,
            rank=1,
            metadata={
                "source": "annual_report.pdf",
                "source_type": "pdf",
                "page": 42,
                "section_title": "Product Strategy",
            },
        ),
        FakeContext(
            chunk_id="chunk-002",
            text=(
                "The company introduced "
                "usage-based pricing."
            ),
            score=0.90,
            rank=2,
            metadata={
                "source": "earnings_transcript.txt",
                "source_type": "transcript",
                "page": 8,
                "section_title": "Pricing",
            },
        ),
        FakeContext(
            chunk_id="chunk-003",
            text=(
                "The product roadmap added "
                "new AI capabilities."
            ),
            score=0.85,
            rank=3,
            metadata={
                "source": "product_docs.md",
                "source_type": "markdown",
                "section_title": "Roadmap",
            },
        ),
    ]


def main():

    contexts = build_contexts()

    mapper = CitationMapper()

    result = mapper.map(
        contexts
    )

    # ---------------------------------------------
    # Result type
    # ---------------------------------------------

    assert isinstance(
        result,
        CitationMappingResult,
    )

    assert len(result.citations) == 3

    # ---------------------------------------------
    # Citation IDs
    # ---------------------------------------------

    assert [
        citation.citation_id
        for citation in result.citations
    ] == [
        "C1",
        "C2",
        "C3",
    ]

    # ---------------------------------------------
    # Chunk mapping
    # ---------------------------------------------

    assert [
        citation.chunk_id
        for citation in result.citations
    ] == [
        "chunk-001",
        "chunk-002",
        "chunk-003",
    ]

    # ---------------------------------------------
    # Source mapping
    # ---------------------------------------------

    assert (
        result.citations[0].source
        == "annual_report.pdf"
    )

    assert (
        result.citations[1].source
        == "earnings_transcript.txt"
    )

    assert (
        result.citations[2].source
        == "product_docs.md"
    )

    # ---------------------------------------------
    # Source type
    # ---------------------------------------------

    assert (
        result.citations[0].source_type
        == "pdf"
    )

    assert (
        result.citations[1].source_type
        == "transcript"
    )

    # ---------------------------------------------
    # Evidence text preserved
    # ---------------------------------------------

    assert (
        result.citations[0].text
        == contexts[0].text
    )

    assert (
        result.citations[1].text
        == contexts[1].text
    )

    # ---------------------------------------------
    # Metadata preserved
    # ---------------------------------------------

    assert (
        result.citations[0].metadata["page"]
        == 42
    )

    assert (
        result.citations[0].metadata[
            "section_title"
        ]
        == "Product Strategy"
    )

    # ---------------------------------------------
    # Lookup
    # ---------------------------------------------

    citation = result.get("C2")

    assert isinstance(
        citation,
        CitationReference,
    )

    assert (
        citation.chunk_id
        == "chunk-002"
    )

    # ---------------------------------------------
    # Missing citation
    # ---------------------------------------------

    assert (
        result.get("C99")
        is None
    )

    # ---------------------------------------------
    # Convenience function
    # ---------------------------------------------

    result = build_citation_mapping(
        contexts
    )

    assert len(
        result.citations
    ) == 3

    # ---------------------------------------------
    # Empty input
    # ---------------------------------------------

    result = mapper.map([])

    assert result.citations == []

    # ---------------------------------------------
    # None input
    # ---------------------------------------------

    try:
        mapper.map(None)

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    print(
        "All citation mapping "
        "assertions passed."
    )


if __name__ == "__main__":
    main()