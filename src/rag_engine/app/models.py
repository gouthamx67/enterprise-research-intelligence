from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from pathlib import Path


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    filename: str
    source_type: str
    size_bytes: int
    stored_path: str
    uploaded_at: str
    status: str = "uploaded"

    @classmethod
    def create(
        cls,
        document_id: str,
        filename: str,
        source_type: str,
        size_bytes: int,
        stored_path: Path,
    ) -> "DocumentRecord":
        return cls(
            document_id=document_id,
            filename=filename,
            source_type=source_type,
            size_bytes=size_bytes,
            stored_path=str(stored_path),
            uploaded_at=datetime.now(timezone.utc).isoformat(),
            status="uploaded",
        )


@dataclass(frozen=True)
class ProcessingResult:
    document_id: str
    status: str
    chunk_count: int
    embedding_count: int
    embedding_dimensions: int
    index_path: str
    processed_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    text: str
    source: str
    page: int | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class QuestionResult:
    document_id: str
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    success: bool
    failure_count: int
    retrieved_count: int
    context_count: int
    metadata: dict[str, Any]