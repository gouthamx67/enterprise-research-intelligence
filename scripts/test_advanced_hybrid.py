from src.rag_engine.advanced.hybrid import (
    AdvancedHybridCombiner,
    HybridEvidence,
)


def main():
    combiner = AdvancedHybridCombiner()

    sparse = [
        HybridEvidence(
            source_type="sparse",
            evidence_id="a",
            text="Product strategy changed.",
            score=0.70,
        ),
        HybridEvidence(
            source_type="sparse",
            evidence_id="b",
            text="Enterprise customers increased.",
            score=0.60,
        ),
    ]

    dense = [
        HybridEvidence(
            source_type="dense",
            evidence_id="a",
            text="Product strategy changed.",
            score=0.90,
        ),
        HybridEvidence(
            source_type="dense",
            evidence_id="c",
            text="New AI capabilities launched.",
            score=0.80,
        ),
    ]

    result = combiner.combine(
        query="product strategy",
        evidence_sets=[sparse, dense],
        top_k=10,
    )

    assert result.count == 3

    ids = [
        evidence.evidence_id
        for evidence in result.evidence
    ]

    assert ids == ["a", "c", "b"]

    print("Advanced hybrid RAG test passed.")


if __name__ == "__main__":
    main()