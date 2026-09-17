from pathlib import Path
import tempfile

import numpy as np

from src.rag_engine.core.document import Chunk
from src.rag_engine.production.storage import (
    ChunkStore,
    EmbeddingStore,
    IndexManifest,
)
from src.rag_engine.retrieval.models import EmbeddedChunk


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)

        chunks = [
            Chunk(
                chunk_id="c1",
                text="Product strategy changed.",
                document_id="doc1",
                source="test.pdf",
                source_type="pdf",
                metadata={
                    "company": "Example",
                },
            )
        ]

        chunk_store = ChunkStore(
            root / "chunks.json"
        )

        chunk_store.save(chunks)

        loaded_chunks = chunk_store.load()

        assert len(loaded_chunks) == 1
        assert loaded_chunks[0].chunk_id == "c1"

        embedded = [
            EmbeddedChunk(
                chunk_id="c1",
                text="Product strategy changed.",
                embedding=np.array(
                    [1.0, 0.0, 0.0]
                ),
                metadata={
                    "company": "Example",
                },
            )
        ]

        embedding_store = EmbeddingStore(
            root / "embeddings.npy",
            root / "embedding_metadata.json",
        )

        embedding_store.save(embedded)

        loaded_embeddings = (
            embedding_store.load()
        )

        assert len(loaded_embeddings) == 1
        assert np.allclose(
            loaded_embeddings[0].embedding,
            embedded[0].embedding,
        )

        manifest = IndexManifest(
            root / "manifest.json"
        )

        manifest.save(
            {
                "doc1": "v1",
            }
        )

        assert manifest.load() == {
            "doc1": "v1"
        }

    print("Production storage test passed.")


if __name__ == "__main__":
    main()