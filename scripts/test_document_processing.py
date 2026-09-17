import tempfile
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from src.rag_engine.app.api import (
    app,
    document_processor,
    upload_service,
)
from src.rag_engine.app.storage import DocumentStorage
from src.rag_engine.retrieval.embeddings import EmbeddingModel


def create_test_pdf(path: Path) -> None:
    pdf = pymupdf.open()

    page = pdf.new_page()

    page.insert_text(
        (72, 72),
        "Product Strategy",
        fontsize=18,
    )

    page.insert_text(
        (72, 110),
        (
            "The company increased investment "
            "in artificial intelligence capabilities."
        ),
        fontsize=12,
    )

    page.insert_text(
        (72, 150),
        (
            "The company is focusing more heavily "
            "on enterprise customers."
        ),
        fontsize=12,
    )

    pdf.save(path)
    pdf.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        pdf_path = root / "research.pdf"

        create_test_pdf(pdf_path)

        upload_storage = DocumentStorage(
            root / "uploads"
        )

        upload_service.storage = upload_storage

        document_processor.storage = upload_storage

        # Reuse the existing embedding implementation.
        # The model is loaded once for this process.
        document_processor.embedding_model = (
            EmbeddingModel()
        )

        client = TestClient(app)

        with pdf_path.open("rb") as file:
            response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "research.pdf",
                        file,
                        "application/pdf",
                    )
                },
            )

        assert response.status_code == 201

        uploaded = response.json()

        document_id = uploaded[
            "document_id"
        ]

        assert uploaded["status"] == "uploaded"

        process_response = client.post(
            f"/documents/{document_id}/process"
        )

        assert process_response.status_code == 200

        processed = process_response.json()

        assert (
            processed["document_id"]
            == document_id
        )

        assert processed["status"] == "ready"

        assert processed["chunk_count"] > 0

        assert (
            processed["embedding_count"]
            == processed["chunk_count"]
        )

        assert (
            processed["embedding_dimensions"]
            == 384
        )

        index_directory = Path(
            processed["index_path"]
        )

        assert (
            index_directory
            / "chunks.json"
        ).exists()

        assert (
            index_directory
            / "embeddings.npy"
        ).exists()

        assert (
            index_directory
            / "embeddings.json"
        ).exists()

        assert (
            index_directory
            / "manifest.json"
        ).exists()

        record_response = client.get(
            f"/documents/{document_id}"
        )

        assert record_response.status_code == 200

        record = record_response.json()

        assert record["status"] == "ready"

        print(
            "Document processing test passed."
        )

        print()
        print(
            f"Document ID:          {document_id}"
        )
        print(
            f"Chunks:               {processed['chunk_count']}"
        )
        print(
            f"Embeddings:           {processed['embedding_count']}"
        )
        print(
            "Embedding dimensions: "
            f"{processed['embedding_dimensions']}"
        )
        print(
            f"Status:               {processed['status']}"
        )


if __name__ == "__main__":
    main()