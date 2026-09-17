import json
from dataclasses import asdict
from pathlib import Path

from src.rag_engine.app.models import DocumentRecord


class DocumentStorage:
    def __init__(self, root: str | Path = "data/uploads") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def document_directory(self, document_id: str) -> Path:
        return self.root / document_id

    def save_file(
        self,
        document_id: str,
        filename: str,
        content: bytes,
    ) -> Path:
        directory = self.document_directory(document_id)
        directory.mkdir(parents=True, exist_ok=False)

        path = directory / filename
        path.write_bytes(content)

        return path

    def save_record(self, record: DocumentRecord) -> Path:
        directory = self.document_directory(record.document_id)
        directory.mkdir(parents=True, exist_ok=True)

        path = directory / "metadata.json"

        path.write_text(
            json.dumps(
                asdict(record),
                indent=2,
            ),
            encoding="utf-8",
        )

        return path

    def load_record(
        self,
        document_id: str,
    ) -> DocumentRecord:
        path = (
            self.document_directory(document_id)
            / "metadata.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Document record not found: {document_id}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        return DocumentRecord(**data)

    def update_status(
        self,
        document_id: str,
        status: str,
    ) -> DocumentRecord:
        record = self.load_record(document_id)

        updated = DocumentRecord(
            document_id=record.document_id,
            filename=record.filename,
            source_type=record.source_type,
            size_bytes=record.size_bytes,
            stored_path=record.stored_path,
            uploaded_at=record.uploaded_at,
            status=status,
        )

        self.save_record(updated)

        return updated

    def index_directory(
        self,
        document_id: str,
    ) -> Path:
        directory = (
            self.document_directory(document_id)
            / "index"
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return directory