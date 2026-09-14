from src.rag_engine.retrieval.context_construction import ContextConstructionPipeline
from src.rag_engine.retrieval.models import RetrievalResult


def make_candidates():
    return [
        RetrievalResult(
            score=0.95,
            rank=1,
            chunk_id="chunk-001",
            text=(
                "The company held its annual meeting. "
                "The product strategy changed toward enterprise customers. "
                "The company also opened a new office."
            ),
            metadata={
                "source": "annual_report.pdf",
                "source_type": "pdf",
                "title": "Annual Report",
                "company": "Example Corp",
                "document_date": "2026-01-15",
                "start_page": 42,
                "end_page": 43,
                "section_title": "Product Strategy",
                "block_numbers": [10, 11, 12],
            },
        ),
        RetrievalResult(
            score=0.90,
            rank=2,
            chunk_id="chunk-002",
            text=(
                "The company expanded its enterprise strategy. "
                "AI capabilities became an important part of the product roadmap."
            ),
            metadata={
                "source": "earnings_transcript.pdf",
                "source_type": "transcript",
                "title": "Q4 Earnings Transcript",
                "company": "Example Corp",
                "document_date": "2026-02-01",
                "start_page": 5,
                "end_page": 6,
                "section_title": "Strategy Update",
                "block_numbers": [20, 21],
            },
        ),
        RetrievalResult(
            score=0.85,
            rank=3,
            chunk_id="chunk-003",
            text=(
                "Pricing changed for enterprise customers. "
                "The company introduced a new monetization model."
            ),
            metadata={
                "source": "pricing.md",
                "source_type": "markdown",
                "title": "Pricing",
                "company": "Example Corp",
                "document_date": "2026-03-01",
                "section_title": "Pricing",
            },
        ),
        RetrievalResult(
            score=0.80,
            rank=4,
            chunk_id="chunk-004",
            text=(
                "The company held its annual meeting. "
                "The product strategy changed toward enterprise customers. "
                "The company announced an older strategy."
            ),
            metadata={
                "source": "old_report.pdf",
                "source_type": "pdf",
                "title": "Old Report",
                "company": "Example Corp",
                "document_date": "2025-01-15",
                "start_page": 10,
                "end_page": 11,
                "section_title": "Old Strategy",
            },
        ),
    ]


def main():
    candidates = make_candidates()

    pipeline = ContextConstructionPipeline(
        max_chunks=4,
        max_sentences_per_chunk=2,
        min_relevance_score=0.5,
        ordering_strategy="rank_order",
    )

    result = pipeline.build(
        query="product strategy enterprise",
        candidates=candidates,
    )

    # =========================================================
    # 1. Context selection
    # =========================================================

    assert result.selection.selected_count == 4, (
        f"expected 4 selected contexts, "
        f"got {result.selection.selected_count}"
    )

    # =========================================================
    # 2. Context deduplication
    # =========================================================

    assert result.deduplication.deduplicated_count == 4, (
        f"expected 4 contexts after deduplication, "
        f"got {result.deduplication.deduplicated_count}"
    )

    # =========================================================
    # 3. Context compression
    # =========================================================

    assert (
        0 < result.compression.compressed_count
        <= result.deduplication.deduplicated_count
    ), (
        "compression should produce a non-empty subset "
        "of the deduplicated contexts"
    )

    assert result.compression.compressed_count < 4, (
        "compression should remove at least one low-relevance context"
    )

    compressed_ids = {
        context.chunk_id
        for context in result.compression.contexts
    }

    assert "chunk-001" in compressed_ids
    assert "chunk-002" in compressed_ids

    # =========================================================
    # 4. Context ordering
    # =========================================================

    assert (
        result.ordering.ordered_count
        == result.compression.compressed_count
    )

    ordered_ids = [
        context.chunk_id
        for context in result.ordering.contexts
    ]

    assert len(ordered_ids) > 0

    original_rank_by_id = {
        candidate.chunk_id: candidate.rank
        for candidate in candidates
    }

    ordered_ranks = [
        original_rank_by_id[context.chunk_id]
        for context in result.ordering.contexts
    ]

    assert ordered_ranks == sorted(ordered_ranks)

    # =========================================================
    # 5. Citation mapping
    # =========================================================

    assert (
        len(result.citations.citations)
        == result.ordering.ordered_count
    )

    citation_ids = [
        citation.citation_id
        for citation in result.citations.citations
    ]

    expected_citation_ids = [
        f"C{i}"
        for i in range(1, result.ordering.ordered_count + 1)
    ]

    assert citation_ids == expected_citation_ids

    # =========================================================
    # 6. Source attribution
    # =========================================================

    assert (
        len(result.attribution.attributions)
        == result.ordering.ordered_count
    )

    for attribution in result.attribution.attributions:
        assert attribution.citation_id in citation_ids
        assert attribution.chunk_id
        assert attribution.source
        assert attribution.source_type

    # =========================================================
    # 7. Final constructed contexts
    # =========================================================

    assert result.final_count == result.ordering.ordered_count
    assert result.final_count > 0

    assert result.total_characters == sum(
        len(context.text)
        for context in result.contexts
    )

    for context in result.contexts:
        assert context.chunk_id
        assert context.citation_id
        assert context.text.strip()
        assert context.source
        assert context.source_type

    # =========================================================
    # 8. Citation/source consistency
    # =========================================================

    for context in result.contexts:
        citation = result.citations.get(context.citation_id)

        assert citation is not None
        assert citation.chunk_id == context.chunk_id
        assert citation.source == context.source
        assert citation.source_type == context.source_type

    # =========================================================
    # 9. Empty candidate set
    # =========================================================
    #
    # No retrieved evidence is a valid RAG outcome.
    # The pipeline should return an empty construction result.
    #

    empty_result = pipeline.build(
        "product strategy enterprise",
        [],
    )

    assert empty_result.final_count == 0
    assert empty_result.selection.selected_count == 0
    assert empty_result.deduplication.deduplicated_count == 0
    assert empty_result.compression.compressed_count == 0
    assert empty_result.ordering.ordered_count == 0
    assert len(empty_result.citations.citations) == 0
    assert len(empty_result.attribution.attributions) == 0

    # =========================================================
    # 10. Convenience configuration
    # =========================================================

    convenience_pipeline = ContextConstructionPipeline(
        max_chunks=2,
        max_sentences_per_chunk=1,
        min_relevance_score=0.5,
        ordering_strategy="rank_order",
    )

    convenience_result = convenience_pipeline.build(
        query="product strategy enterprise",
        candidates=candidates,
    )

    assert convenience_result.final_count > 0
    assert convenience_result.final_count <= 2

    # =========================================================
    # 11. Validation
    # =========================================================

    try:
        pipeline.build("", candidates)
        raise AssertionError("empty query should raise ValueError")
    except ValueError:
        pass

    try:
        ContextConstructionPipeline(
            ordering_strategy="invalid_strategy"
        )
        raise AssertionError(
            "invalid ordering strategy should raise ValueError"
        )
    except ValueError:
        pass

    # =========================================================
    # Success
    # =========================================================

    print("Context construction test passed.")
    print(f"Selected:       {result.selection.selected_count}")
    print(f"Deduplicated:   {result.deduplication.deduplicated_count}")
    print(f"Compressed:     {result.compression.compressed_count}")
    print(f"Ordered:        {result.ordering.ordered_count}")
    print(f"Citations:      {len(result.citations.citations)}")
    print(f"Attributions:   {len(result.attribution.attributions)}")
    print(f"Final contexts: {result.final_count}")
    print(f"Characters:     {result.total_characters}")


if __name__ == "__main__":
    main()