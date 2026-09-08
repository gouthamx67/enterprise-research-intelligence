import numpy as np

from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.embeddings import (
    EmbeddingModel,
)
from src.rag_engine.retrieval.models import (
    EmbeddedChunk,
    RetrievalResult,
)


def cosine_similarity(
    query_vector: np.ndarray,
    document_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculate cosine similarity between one query vector
    and multiple document vectors.

    Both the query vector and document vectors are expected
    to be normalized.

    Returns:
        One similarity score per document vector.
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
            "query_vector must be one-dimensional"
        )

    if document_matrix.ndim != 2:
        raise ValueError(
            "document_matrix must be two-dimensional"
        )

    if (
        document_matrix.shape[1]
        != query_vector.shape[0]
    ):
        raise ValueError(
            "query and document vectors must "
            "have the same dimensions"
        )

    query_norm = np.linalg.norm(
        query_vector
    )

    document_norms = np.linalg.norm(
        document_matrix,
        axis=1,
    )

    if query_norm == 0:
        raise ValueError(
            "query_vector cannot be a zero vector"
        )

    if np.any(document_norms == 0):
        raise ValueError(
            "document vectors cannot contain "
            "zero vectors"
        )

    return (
        document_matrix @ query_vector
    ) / (
        document_norms * query_norm
    )


class DenseIndex:
    """
    In-memory dense vector index.

    Embeddings are stored in a NumPy matrix and searched
    using cosine similarity.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel | None = None,
    ):
        self.embedding_model = (
            embedding_model
            or EmbeddingModel()
        )

        self.embedded_chunks: list[
            EmbeddedChunk
        ] = []

        self.matrix: np.ndarray | None = None

    def add_chunks(
        self,
        chunks: list[Chunk],
    ):
        """
        Generate embeddings for chunks and add them
        to the dense index.
        """

        if not chunks:
            raise ValueError(
                "chunks cannot be empty"
            )

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts
        )

        for chunk, embedding in zip(
            chunks,
            embeddings,
        ):
            self.embedded_chunks.append(
                EmbeddedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    embedding=np.asarray(
                        embedding
                    ),
                    metadata={
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        "section_title": chunk.section_title,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "block_numbers": chunk.block_numbers,
                        **chunk.metadata,
                    },
                )
            )

        self.matrix = np.vstack(
            [
                item.embedding
                for item in self.embedded_chunks
            ]
        )

    @property
    def size(self) -> int:
        """
        Number of embedded chunks currently indexed.
        """

        return len(
            self.embedded_chunks
        )

    @property
    def dimensions(self) -> int | None:
        """
        Number of dimensions in the embedding vectors.
        """

        if self.matrix is None:
            return None

        return self.matrix.shape[1]

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-k chunks using cosine similarity.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        if self.matrix is None:
            raise ValueError(
                "index is empty; add chunks first"
            )

        query_vector = (
            self.embedding_model.encode_one(
                query
            )
        )

        scores = cosine_similarity(
            query_vector,
            self.matrix,
        )

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):
            item = self.embedded_chunks[index]

            results.append(
                RetrievalResult(
                    score=float(scores[index]),
                    rank=rank,
                    chunk_id=item.chunk_id,
                    text=item.text,
                    metadata=item.metadata.copy(),
                )
            )

        return results