from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RetrievalResult:
    """
    A single retrieval result.

    score:
        Retrieval relevance score.

    rank:
        Position of this result in the ranked result list.

    chunk_id:
        Identifier of the retrieved chunk.

    text:
        Chunk text returned as evidence.

    metadata:
        Additional information needed for
        attribution and downstream processing.
    """

    score: float
    rank: int
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(
        default_factory=dict
    )


@dataclass
class EmbeddedChunk:
    """
    A chunk together with its dense vector representation.
    """

    chunk_id: str
    text: str
    embedding: np.ndarray
    metadata: dict[str, Any] = field(
        default_factory=dict
    )