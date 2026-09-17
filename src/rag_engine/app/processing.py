from datetime import datetime, timezone
from pathlib import Path

from src.rag_engine.app.models import ProcessingResult
from src.rag_engine.app.storage import DocumentStorage
from src.rag_engine.core.document import Chunk
from src.rag_engine.ingestion.pdf import extract_pdf_blocks
from src.rag_engine.production.storage import (
    ChunkStore,
    EmbeddingStore,
    IndexManifest,
)
from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.embeddings import EmbeddingModel
from src.rag_engine.retrieval.sparse import BM25Retriever


class DocumentProcessingError(RuntimeError):
    """Raised when document processing fails."""


class DocumentProcessor:
    """
    Convert an uploaded document into a searchable index.

    Current supported ingestion format:
        PDF

    Processing pipeline:

        PDF
          ↓
        blocks
          ↓
        chunks
          ↓
        BM25 persistence
          +
        dense embeddings persistence
          ↓
        READY
    """

    def __init__(
        self,
        storage: DocumentStorage | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self.storage = storage or DocumentStorage()

        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

    def _extract_pdf_chunks(
        self,
        document_id: str,
        pdf_path: Path,
        filename: str,
    ) -> list[Chunk]:
        import pymupdf

        pdf = pymupdf.open(pdf_path)

        chunks: list[Chunk] = []

        try:
            for page_number, page in enumerate(
                pdf,
                start=1,
            ):
                blocks = extract_pdf_blocks(
                    page,
                    page_number,
                )

                for block in blocks:
                    chunk_id = (
                        f"{document_id}"
                        f"-p{page_number}"
                        f"-b{block.block_number}"
                    )

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=block.text,
                            document_id=document_id,
                            source=filename,
                            source_type="pdf",
                            section_title=None,
                            start_page=page_number,
                            end_page=page_number,
                            block_numbers=[
                                block.block_number
                            ],
                            metadata={
                                "document_id": document_id,
                                "filename": filename,
                                "page_number": page_number,
                                "block_number": block.block_number,
                            },
                        )
                    )
        finally:
            pdf.close()

        return chunks

    def _persist_chunks(
        self,
        index_directory: Path,
        chunks: list[Chunk],
    ) -> None:
        ChunkStore(
            index_directory / "chunks.json"
        ).save(chunks)

    def _persist_embeddings(
        self,
        index_directory: Path,
        dense_index: DenseIndex,
    ) -> None:
        EmbeddingStore(
            embedding_path=(
                index_directory
                / "embeddings.npy"
            ),
            metadata_path=(
                index_directory
                / "embeddings.json"
            ),
        ).save(
            dense_index.embedded_chunks
        )

    def _persist_manifest(
        self,
        index_directory: Path,
        document_id: str,
    ) -> None:
        IndexManifest(
            index_directory / "manifest.json"
        ).save(
            {
                document_id: "1",
            }
        )

    def process(
        self,
        document_id: str,
    ) -> ProcessingResult:
        record = self.storage.load_record(
            document_id
        )

        if record.source_type != "pdf":
            raise DocumentProcessingError(
                "Only PDF processing is currently "
                "implemented."
            )

        self.storage.update_status(
            document_id,
            "processing",
        )

        try:
            pdf_path = Path(
                record.stored_path
            )

            if not pdf_path.exists():
                raise DocumentProcessingError(
                    f"Stored file does not exist: "
                    f"{pdf_path}"
                )

            chunks = self._extract_pdf_chunks(
                document_id=document_id,
                pdf_path=pdf_path,
                filename=record.filename,
            )

            if not chunks:
                raise DocumentProcessingError(
                    "No searchable text blocks were "
                    "extracted from the PDF."
                )

            index_directory = (
                self.storage.index_directory(
                    document_id
                )
            )

            self._persist_chunks(
                index_directory,
                chunks,
            )

            dense_index = DenseIndex(
                embedding_model=self.embedding_model
            )

            dense_index.add_chunks(chunks)

            self._persist_embeddings(
                index_directory,
                dense_index,
            )

            # Validate that the sparse index can
            # successfully be constructed from
            # the persisted chunk representation.
            sparse_index = BM25Retriever(chunks)

            if not sparse_index.chunks:
                raise DocumentProcessingError(
                    "BM25 index contains no chunks."
                )

            self._persist_manifest(
                index_directory,
                document_id,
            )

            self.storage.update_status(
                document_id,
                "ready",
            )

            processed_at = (
                datetime.now(timezone.utc)
                .isoformat()
            )

            return ProcessingResult(
                document_id=document_id,
                status="ready",
                chunk_count=len(chunks),
                embedding_count=dense_index.size,
                embedding_dimensions=(
                    dense_index.dimensions
                ),
                index_path=str(
                    index_directory
                ),
                processed_at=processed_at,
                metadata={
                    "source_type": record.source_type,
                    "filename": record.filename,
                    "retrieval": [
                        "bm25",
                        "dense",
                    ],
                },
            )

        except Exception as exc:
            self.storage.update_status(
                document_id,
                "failed",
            )

            if isinstance(
                exc,
                DocumentProcessingError,
            ):
                raise

            raise DocumentProcessingError(
                f"Document processing failed: {exc}"
            ) from exc