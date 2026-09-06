import numpy as np
from sentence_transformers import SentenceTransformer

from src.rag_engine.chunking.strategies import (
    sentence_chunks,
)


class SemanticChunker:
    """
    Chunk text according to semantic similarity
    between neighboring sentences.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model = SentenceTransformer(model_name)

    def chunk(
        self,
        text: str,
        threshold: float = 0.35,
    ):
        """
        Create chunks by detecting semantic transitions.

        A new chunk starts when the semantic distance
        between neighboring sentences exceeds threshold.
        """

        if threshold < 0:
            raise ValueError(
                "threshold must be non-negative"
            )

        sentences = sentence_chunks(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        embeddings = self.model.encode(
            sentences,
            normalize_embeddings=True,
        )

        chunks = []
        current_chunk = [sentences[0]]

        for index in range(1, len(sentences)):
            previous = embeddings[index - 1]
            current = embeddings[index]

            similarity = float(
                np.dot(previous, current)
            )

            distance = 1.0 - similarity

            if distance > threshold:
                chunks.append(
                    " ".join(current_chunk).strip()
                )
                current_chunk = []

            current_chunk.append(sentences[index])

        if current_chunk:
            chunks.append(
                " ".join(current_chunk).strip()
            )

        return chunks