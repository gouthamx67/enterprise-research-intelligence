import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from src.rag_engine.app.api import app, upload_service
from src.rag_engine.app.storage import DocumentStorage
from src.rag_engine.app.upload import DocumentUploadService


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        upload_service.storage = DocumentStorage(temp_dir)

        client = TestClient(app)

        health_response = client.get("/health")

        assert health_response.status_code == 200
        assert health_response.json()["status"] == "ok"

        content = b"This is a test research document."

        upload_response = client.post(
            "/documents/upload",
            files={
                "file": (
                    "research.pdf",
                    content,
                    "application/pdf",
                )
            },
        )

        assert upload_response.status_code == 201

        uploaded = upload_response.json()

        assert uploaded["filename"] == "research.pdf"
        assert uploaded["source_type"] == "pdf"
        assert uploaded["size_bytes"] == len(content)
        assert uploaded["status"] == "uploaded"

        document_id = uploaded["document_id"]

        assert document_id.startswith("doc_")

        directory = Path(temp_dir) / document_id
        stored_file = directory / "research.pdf"
        metadata_file = directory / "metadata.json"

        assert stored_file.exists()
        assert metadata_file.exists()
        assert stored_file.read_bytes() == content

        metadata = json.loads(
            metadata_file.read_text(encoding="utf-8")
        )

        assert metadata["document_id"] == document_id

        get_response = client.get(
            f"/documents/{document_id}"
        )

        assert get_response.status_code == 200
        assert get_response.json()["document_id"] == document_id

        bad_upload = client.post(
            "/documents/upload",
            files={
                "file": (
                    "malware.exe",
                    b"not allowed",
                    "application/octet-stream",
                )
            },
        )

        assert bad_upload.status_code == 400

        missing_response = client.get(
            "/documents/doc_does_not_exist"
        )

        assert missing_response.status_code == 404

    print("Document upload API test passed.")


if __name__ == "__main__":
    main()