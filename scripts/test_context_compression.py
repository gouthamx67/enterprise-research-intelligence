from dataclasses import dataclass

from src.rag_engine.retrieval.context_compression import (
    CompressedContext,
    ContextCompressor,
    ContextCompressionResult,
    compress_context,
    sentence_relevance_score,
    split_sentences,
    tokenize_for_relevance,
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
                "The company held its annual meeting. "
                "The product strategy changed toward "
                "enterprise customers. "
                "The company also opened a new office."
            ),
            score=0.95,
            rank=1,
            metadata={
                "source": "filing.pdf",
                "page": 10,
            },
        ),
        FakeContext(
            chunk_id="chunk-002",
            text=(
                "The company reported revenue growth. "
                "The new pricing model uses usage-based "
                "billing. "
                "Several executives discussed hiring."
            ),
            score=0.90,
            rank=2,
            metadata={
                "source": "transcript.txt",
                "page": 4,
            },
        ),
    ]


def main():

    contexts = build_contexts()

    # ---------------------------------------------
    # Sentence splitting
    # ---------------------------------------------

    sentences = split_sentences(
        "First sentence. Second sentence! Third sentence?"
    )

    assert sentences == [
        "First sentence.",
        "Second sentence!",
        "Third sentence?",
    ]

    assert split_sentences("   ") == []

    # ---------------------------------------------
    # Tokenization
    # ---------------------------------------------

    tokens = tokenize_for_relevance(
        "Product Strategy!"
    )

    assert tokens == {
        "product",
        "strategy",
    }

    # ---------------------------------------------
    # Sentence relevance
    # ---------------------------------------------

    score = sentence_relevance_score(
        "product strategy",
        "The product strategy changed.",
    )

    assert score == 1.0

    score = sentence_relevance_score(
        "product strategy",
        "The company opened a new office.",
    )

    assert score == 0.0

    # ---------------------------------------------
    # Basic compression
    # ---------------------------------------------

    compressor = ContextCompressor(
        max_sentences_per_chunk=1
    )

    result = compressor.compress(
        query="product strategy",
        contexts=contexts,
    )

    assert isinstance(
        result,
        ContextCompressionResult,
    )

    assert result.original_count == 2

    assert result.compressed_count == 2

    # First chunk should keep the sentence
    # containing both query terms.
    assert (
        result.contexts[0].text
        == "The product strategy changed toward "
        "enterprise customers."
    )

    # Second chunk has no matching terms, but
    # threshold 0.0 permits its best sentence.
    assert (
        result.contexts[1].text
        == "The company reported revenue growth."
    )

    # ---------------------------------------------
    # Compression statistics
    # ---------------------------------------------

    assert (
        result.original_characters
        > result.compressed_characters
    )

    assert (
        0.0
        < result.compression_ratio
        < 1.0
    )

    # ---------------------------------------------
    # Multiple sentences
    # ---------------------------------------------

    compressor = ContextCompressor(
        max_sentences_per_chunk=2,
        min_relevance_score=0.5,
    )

    result = compressor.compress(
        query="product strategy",
        contexts=contexts,
    )

    assert result.compressed_count == 1

    assert (
        result.contexts[0].chunk_id
        == "chunk-001"
    )

    # ---------------------------------------------
    # Original order must be preserved
    # ---------------------------------------------

    text = (
        result.contexts[0].text
    )

    assert text.index(
        "product strategy"
    ) < text.index(
        "enterprise customers"
    )

    # ---------------------------------------------
    # Original text preserved
    # ---------------------------------------------

    compressed = result.contexts[0]

    assert isinstance(
        compressed,
        CompressedContext,
    )

    assert (
        compressed.original_text
        == contexts[0].text
    )

    assert (
        compressed.original_character_count
        == len(contexts[0].text)
    )

    assert (
        compressed.compressed_character_count
        == len(compressed.text)
    )

    # ---------------------------------------------
    # Metadata preserved
    # ---------------------------------------------

    assert (
        compressed.metadata["source"]
        == "filing.pdf"
    )

    assert (
        compressed.metadata["page"]
        == 10
    )

    # ---------------------------------------------
    # Rank and score preserved
    # ---------------------------------------------

    assert compressed.rank == 1
    assert compressed.score == 0.95

    # ---------------------------------------------
    # Convenience function
    # ---------------------------------------------

    result = compress_context(
        query="pricing model",
        contexts=contexts,
        max_sentences_per_chunk=1,
        min_relevance_score=0.5,
    )

    assert result.compressed_count == 1

    assert (
        result.contexts[0].chunk_id
        == "chunk-002"
    )

    assert (
        "pricing model"
        in result.contexts[0].text
    )

    # ---------------------------------------------
    # Invalid configuration
    # ---------------------------------------------

    try:
        ContextCompressor(
            max_sentences_per_chunk=0
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    try:
        ContextCompressor(
            min_relevance_score=1.5
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid query
    # ---------------------------------------------

    try:
        compressor.compress(
            query="   ",
            contexts=contexts,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Invalid contexts
    # ---------------------------------------------

    try:
        compressor.compress(
            query="test",
            contexts=None,
        )

        raise AssertionError(
            "Expected ValueError"
        )

    except ValueError:
        pass

    # ---------------------------------------------
    # Empty input
    # ---------------------------------------------

    result = compressor.compress(
        query="product strategy",
        contexts=[],
    )

    assert result.original_count == 0
    assert result.compressed_count == 0
    assert result.original_characters == 0
    assert result.compressed_characters == 0
    assert result.compression_ratio == 0.0

    print(
        "All context compression "
        "assertions passed."
    )


if __name__ == "__main__":
    main()