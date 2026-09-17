import hashlib
import re
import uuid
from pathlib import Path

from src.rag_engine.app.models import DocumentRecord
from src.rag_engine.app.storage import DocumentStorage


class UploadValidationError(ValueError):
    """Raised when an uploaded document is invalid."""


class DocumentUploadService:
    ALLOWED_EXTENSIONS = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".md": "markdown",
        ".markdown": "markdown",
        ".html": "html",
        ".htm": "html",
        ".csv": "csv",
        ".json": "json",
    }

    def __init__(
        self,
        storage: DocumentStorage | None = None,
        max_file_size: int = 25 * 1024 * 1024,
    ) -> None:
        self.storage = storage or DocumentStorage()
        self.max_file_size = max_file_size

    def validate_filename(self, filename: str) -> str:
        if not filename:
            raise UploadValidationError("Filename is required.")

        extension = Path(filename).suffix.lower()

        source_type = self.ALLOWED_EXTENSIONS.get(extension)

        if source_type is None:
            allowed = ", ".join(sorted(self.ALLOWED_EXTENSIONS))
            raise UploadValidationError(
                f"Unsupported file type '{extension}'. "
                f"Allowed extensions: {allowed}"
            )

        return source_type

    def validate_content(self, content: bytes) -> None:
        if not content:
            raise UploadValidationError("Uploaded file is empty.")

        if len(content) > self.max_file_size:
            raise UploadValidationError(
                f"File exceeds the maximum size of "
                f"{self.max_file_size} bytes."
            )

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        name = Path(filename).name

        name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)
        name = name.strip(" .")

        if not name:
            raise UploadValidationError(
                "Filename becomes empty after sanitization."
            )

        return name

    @staticmethod
    def generate_document_id(content: bytes) -> str:
        digest = hashlib.sha256(content).hexdigest()[:16]
        return f"doc_{digest}_{uuid.uuid4().hex[:8]}"

    def upload(
        self,
        filename: str,
        content: bytes,
    ) -> DocumentRecord:
        source_type = self.validate_filename(filename)
        self.validate_content(content)

        safe_filename = self.sanitize_filename(filename)
        document_id = self.generate_document_id(content)

        stored_path = self.storage.save_file(
            document_id=document_id,
            filename=safe_filename,
            content=content,
        )

        record = DocumentRecord.create(
            document_id=document_id,
            filename=safe_filename,
            source_type=source_type,
            size_bytes=len(content),
            stored_path=stored_path,
        )

        self.storage.save_record(record)

        return record