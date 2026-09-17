import tempfile
from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient

from src.rag_engine.app.api import app
from src.rag_engine.app.processing import DocumentProcessor
from src.rag_engine.app.question import DocumentQuestionService
from src.rag_engine.app.storage import DocumentStorage
from src.rag_engine.app.upload import DocumentUploadService
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
            "The company launched an enterprise "
            "platform focused on analytics and automation."
        ),
        fontsize=12,
    )

    page.insert_text(
        (72, 150),
        (
            "The company increased investment in "
            "artificial intelligence capabilities."
        ),
        fontsize=12,
    )

    page.insert_text(
        (72, 190),
        (
            "The company is focusing more heavily "
            "on large enterprise customers."
        ),
        fontsize=12,
    )

    pdf.save(path)
    pdf.close()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        pdf_path = root / "strategy.pdf"

        create_test_pdf(pdf_path)

        storage = DocumentStorage(
            root / "uploads"
        )

        upload_service = DocumentUploadService(
            storage=storage
        )

        embedding_model = EmbeddingModel()

        processor = DocumentProcessor(
            storage=storage,
            embedding_model=embedding_model,
        )

        question_service = DocumentQuestionService(
            storage=storage,
            embedding_model=embedding_model,
        )

        client = TestClient(app)

        # Replace application services with
        # test-local storage.
        import src.rag_engine.app.api as api_module

        api_module.upload_service = upload_service
        api_module.document_processor = processor
        api_module.question_service = question_service

        # --------------------------------------------------
        # 1. Upload document
        # --------------------------------------------------

        with pdf_path.open("rb") as file:
            upload_response = client.post(
                "/documents/upload",
                files={
                    "file": (
                        "strategy.pdf",
                        file,
                        "application/pdf",
                    )
                },
            )

        assert upload_response.status_code == 201

        uploaded = upload_response.json()

        document_id = uploaded["document_id"]

        assert uploaded["status"] == "uploaded"

        # --------------------------------------------------
        # 2. Process document
        # --------------------------------------------------

        process_response = client.post(
            f"/documents/{document_id}/process"
        )

        assert process_response.status_code == 200

        processed = process_response.json()

        assert processed["status"] == "ready"

        assert processed["chunk_count"] == 4

        assert (
            processed["embedding_count"]
            == processed["chunk_count"]
        )

        assert (
            processed["embedding_dimensions"]
            == 384
        )

        # --------------------------------------------------
        # 3. Ask question
        # --------------------------------------------------

        ask_response = client.post(
            f"/documents/{document_id}/ask",
            json={
                "question": (
                    "What changed in the company's "
                    "product strategy?"
                )
            },
        )

        assert ask_response.status_code == 200

        result = ask_response.json()

        # --------------------------------------------------
        # 4. Validate response
        # --------------------------------------------------

        assert result["document_id"] == document_id

        assert (
            result["question"]
            == "What changed in the company's product strategy?"
        )

        assert result["answer"]

        assert result["retrieved_count"] > 0

        assert result["context_count"] > 0

        assert result["citations"]

        assert result["success"] is True

        assert result["metadata"][
            "retrieval_strategy"
        ] == "hybrid_rrf"

        assert result["metadata"][
            "citation_valid"
        ] is True

        # --------------------------------------------------
        # 5. Validate citation IDs
        # --------------------------------------------------

        citation_ids = [
            citation["citation_id"]
            for citation in result["citations"]
        ]

        assert citation_ids == [
            f"C{index}"
            for index in range(
                1,
                len(citation_ids) + 1,
            )
        ]

        # --------------------------------------------------
        # 6. Validate citation contents
        # --------------------------------------------------

        for citation in result["citations"]:
            assert citation["citation_id"]
            assert citation["chunk_id"]
            assert citation["text"]
            assert citation["source"]

        # --------------------------------------------------
        # 7. Final output
        # --------------------------------------------------

        print("Question API test passed.")
        print()
        print(f"Document ID:      {document_id}")
        print(f"Retrieved:        {result['retrieved_count']}")
        print(f"Context:          {result['context_count']}")
        print(f"Citations:        {len(result['citations'])}")
        print(f"Confidence:       {result['confidence']}")
        print(f"Success:          {result['success']}")


if __name__ == "__main__":
    main()