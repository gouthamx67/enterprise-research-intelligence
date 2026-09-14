from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MultimodalEvidence:
    evidence_id: str
    modality: str
    content: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MultimodalStore:
    """
    Simple in-memory evidence store for multimodal RAG.

    Supported evidence types include:
        - text
        - table
        - image
        - chart
        - diagram

    The content field currently stores a textual representation.
    Later this can be extended to actual image/table/chart
    representations and vision-model embeddings.
    """

    SUPPORTED_MODALITIES = {
        "text",
        "table",
        "image",
        "chart",
        "diagram",
    }

    def __init__(self):
        self.evidence: list[MultimodalEvidence] = []

    def add(
        self,
        evidence_id: str,
        modality: str,
        content: str,
        source: str,
        metadata: dict[str, Any] | None = None,
    ) -> MultimodalEvidence:
        if not evidence_id or not evidence_id.strip():
            raise ValueError(
                "evidence_id cannot be empty"
            )

        if modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(
                f"unsupported modality: {modality}"
            )

        if not content or not content.strip():
            raise ValueError(
                "content cannot be empty"
            )

        if not source or not source.strip():
            raise ValueError(
                "source cannot be empty"
            )

        evidence = MultimodalEvidence(
            evidence_id=evidence_id.strip(),
            modality=modality,
            content=content.strip(),
            source=source.strip(),
            metadata=dict(metadata or {}),
        )

        self.evidence.append(evidence)

        return evidence

    def search(
        self,
        modality: str | None = None,
    ) -> list[MultimodalEvidence]:
        if modality is not None:
            if modality not in self.SUPPORTED_MODALITIES:
                raise ValueError(
                    f"unsupported modality: {modality}"
                )

        if modality is None:
            return list(self.evidence)

        return [
            item
            for item in self.evidence
            if item.modality == modality
        ]