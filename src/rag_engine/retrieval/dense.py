from typing import Any

import numpy as np

from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.embeddings import EmbeddingModel
from src.rag_engine.retrieval.filters import filter_chunks
from src.rag_engine.retrieval.models import (
    EmbeddedChunk,
    RetrievalResult,
)


def cosine_similarity(
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query vector and
    a matrix of document vectors.
    """

    query_vector = np.asarray(
        query_vector,
        dtype=float,
    )

    document_matrix = np.asarray(
        document_matrix,
        dtype=float,
    )

    if query_vector.ndim != 1:
        raise ValueError(
            "query_vector must be a 1-dimensional array"
        )

    if document_matrix.ndim != 2:
        raise ValueError(
            "document_matrix must be a 2-dimensional array"
        )

    if query_vector.shape[0] != document_matrix.shape[1]:
        raise ValueError(
            "query and document dimensions must match"
        )

    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
    )

    if query_norm == 0:
        raise ValueError(
            "query_vector cannot have zero magnitude"
        )

    if np.any(document_norms == 0):
        raise ValueError(
            "document vectors cannot have zero magnitude"
        )

    return (
        document_matrix @ query_vector
    ) / (
        document_norms * query_norm
    )


class DenseIndex:
    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

        self.embedded_chunks: list[EmbeddedChunk] = []

        self.matrix = np.empty(
            (0, 0),
            dtype=float,
        )

    @property
    def size(self) -> int:
        return len(self.embedded_chunks)

    @property
    def dimensions(self) -> int:
        if self.matrix.ndim != 2:
            return 0

        if self.matrix.shape[0] == 0:
            return 0

        return self.matrix.shape[1]

    def add_chunks(
        self,
        chunks: list[Chunk],
    ) -> None:
        if not chunks:
            raise ValueError("chunks cannot be empty")

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts
        )

        new_embedded_chunks = []

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            metadata = {
                "document_id": chunk.document_id,
                "source": chunk.source,
                "source_type": chunk.source_type,
                "section_title": chunk.section_title,
                "start_page": chunk.start_page,
                "end_page": chunk.end_page,
                "block_numbers": chunk.block_numbers,
                **chunk.metadata,
            }

            new_embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=np.asarray(
                        embedding,
                        dtype=float,
                    ),
                    metadata=metadata,
                )
            )

        self.embedded_chunks.extend(
            new_embedded_chunks
        )

        self.matrix = np.vstack(
            [
                item.embedding
                for item in self.embedded_chunks
            ]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        query_embedding = (
            self.embedding_model.encode_one(query)
        )

        return self.search_by_vector(
            query_embedding,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )

    def search_by_vector(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        if self.size == 0:
            return []

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if score_threshold is not None:
            if not -1.0 <= score_threshold <= 1.0:
                raise ValueError(
                    "score_threshold must be between -1 and 1"
                )

        query_vector = np.asarray(
            query_vector,
            dtype=float,
        )

        if query_vector.ndim != 1:
            raise ValueError(
                "query_vector must be a 1-dimensional array"
            )

        if query_vector.shape[0] != self.dimensions:
            raise ValueError(
                "query vector dimension does not match index"
            )

        eligible_chunks = filter_chunks(
            [
                Chunk(
                    chunk_id=item.chunk_id,
                    text=item.text,
                    document_id=item.metadata.get(
                        "document_id",
                        "",
                    ),
                    source=item.metadata.get(
                        "source",
                        "",
                    ),
                    source_type=item.metadata.get(
                        "source_type",
                        "",
                    ),
                    section_title=item.metadata.get(
                        "section_title"
                    ),
                    start_page=item.metadata.get(
                        "start_page"
                    ),
                    end_page=item.metadata.get(
                        "end_page"
                    ),
                    block_numbers=item.metadata.get(
                        "block_numbers",
                        [],
                    ),
                    metadata=item.metadata,
                )
                for item in self.embedded_chunks
            ],
            filters,
        )

        eligible_ids = {
            chunk.chunk_id
            for chunk in eligible_chunks
        }

        eligible_indices = [
            index
            for index, item in enumerate(
                self.embedded_chunks
            )
            if item.chunk_id in eligible_ids
        ]

        if not eligible_indices:
            return []

        eligible_matrix = self.matrix[
            eligible_indices
        ]

        scores = cosine_similarity(
            query_vector,
            eligible_matrix,
        )

        scored_items = []

        for local_index, score in enumerate(
            scores
        ):
            global_index = eligible_indices[
                local_index
            ]

            if (
                score_threshold is not None
                and score < score_threshold
            ):
                continue

            scored_items.append(
                (
                    global_index,
                    float(score),
                )
            )

        scored_items.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        results = []

        for rank, (
            global_index,
            score,
        ) in enumerate(
            scored_items[:top_k],
            start=1,
        ):
            item = self.embedded_chunks[
                global_index
            ]

            results.append(
                RetrievalResult(
                    score=score,
                    rank=rank,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    metadata=item.metadata.copy(),
                )
            )

        return results