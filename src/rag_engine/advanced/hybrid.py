from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HybridEvidence:
    source_type: str
    evidence_id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridResult:
    query: str
    evidence: list[HybridEvidence] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.evidence)


class AdvancedHybridCombiner:
    """
    Combines evidence from multiple retrieval systems.

    This layer does not replace the hybrid retriever already
    implemented in the retrieval package.

    Instead, it provides an advanced orchestration layer that
    can combine evidence from different sources such as:

        - sparse retrieval
        - dense retrieval
        - graph retrieval
        - multimodal retrieval

    Evidence with the same evidence_id is deduplicated and
    the highest score is retained.
    """

    def combine(
        self,
        query: str,
        evidence_sets: list[list[HybridEvidence]],
        top_k: int = 10,
    ) -> HybridResult:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        merged: dict[str, HybridEvidence] = {}

        for evidence_set in evidence_sets:
            for evidence in evidence_set:
                existing = merged.get(
                    evidence.evidence_id
                )

                if (
                    existing is None
                    or evidence.score > existing.score
                ):
                    merged[evidence.evidence_id] = evidence

        ordered = sorted(
            merged.values(),
            key=lambda item: item.score,
            reverse=True,
        )

        return HybridResult(
            query=query.strip(),
            evidence=ordered[:top_k],
        )