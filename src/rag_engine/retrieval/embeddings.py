import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Wrapper around a sentence-transformers embedding model.

    The wrapper gives the retrieval layer a simple interface
    for converting text into normalized dense vectors.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        self.model = SentenceTransformer(
            model_name
        )

    def encode(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Convert multiple texts into normalized embeddings.
        """

        if not texts:
            raise ValueError(
                "texts cannot be empty"
            )

        return self.model.encode(
            texts,
            normalize_embeddings=True,
        )

    def encode_one(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Convert a single text into one normalized embedding.
        """

        if not text.strip():
            raise ValueError(
                "text cannot be empty"
            )

        return self.model.encode(
            text,
            normalize_embeddings=True,
        )