from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.models import EmbeddedChunk


class ChunkStore:
    """
    Simple persistent JSON store for chunks.

    This deliberately uses a filesystem implementation so the
    persistence concept is visible before introducing a database.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        chunks: list[Chunk],
    ) -> None:
        payload = []

        for chunk in chunks:
            payload.append(
                asdict(chunk)
            )

        self.path.write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def load(self) -> list[Chunk]:
        if not self.path.exists():
            return []

        payload = json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )

        return [
            Chunk(**item)
            for item in payload
        ]


class EmbeddingStore:
    """
    Persistent embedding store.

    Embeddings are stored as a NumPy matrix while metadata is stored
    separately as JSON.
    """

    def __init__(
        self,
        embedding_path: str | Path,
        metadata_path: str | Path,
    ):
        self.embedding_path = Path(
            embedding_path
        )
        self.metadata_path = Path(
            metadata_path
        )

        self.embedding_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        embedded_chunks: list[EmbeddedChunk],
    ) -> None:
        if not embedded_chunks:
            raise ValueError(
                "embedded_chunks cannot be empty"
            )

        matrix = np.vstack(
            [
                item.embedding
                for item in embedded_chunks
            ]
        )

        metadata = [
            {
                "chunk_id": item.chunk_id,
                "text": item.text,
                "metadata": item.metadata,
            }
            for item in embedded_chunks
        ]

        np.save(
            self.embedding_path,
            matrix,
        )

        self.metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def load(self) -> list[EmbeddedChunk]:
        if (
            not self.embedding_path.exists()
            or not self.metadata_path.exists()
        ):
            return []

        matrix = np.load(
            self.embedding_path
        )

        metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if len(matrix) != len(metadata):
            raise ValueError(
                "embedding and metadata counts do not match"
            )

        return [
            EmbeddedChunk(
                chunk_id=item["chunk_id"],
                text=item["text"],
                embedding=matrix[index],
                metadata=item["metadata"],
            )
            for index, item in enumerate(metadata)
        ]


class IndexManifest:
    """
    Records which document versions are represented in an index.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        documents: dict[str, str],
    ) -> None:
        self.path.write_text(
            json.dumps(
                documents,
                indent=2,
            ),
            encoding="utf-8",
        )

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}

        return json.loads(
            self.path.read_text(
                encoding="utf-8"
            )
        )