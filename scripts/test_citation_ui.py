from pathlib import Path

from fastapi.testclient import TestClient

from src.rag_engine.app.api import app


class FakeQuestionService:
    def ask(
        self,
        document_id: str,
        question: str,
    ) -> dict:
        return {
            "document_id": document_id,
            "question": question,
            "answer": (
                "The company expanded its enterprise "
                "product strategy [C1] and added stronger "
                "administrative controls [C2]."
            ),
            "citations": [
                {
                    "citation_id": "C1",
                    "chunk_id": "chunk-1",
                    "text": (
                        "The company expanded its enterprise "
                        "product strategy."
                    ),
                    "source": "strategy.pdf",
                    "page": 4,
                    "metadata": {
                        "section": "Product Strategy",
                    },
                },
                {
                    "citation_id": "C2",
                    "chunk_id": "chunk-2",
                    "text": (
                        "The company added stronger "
                        "administrative controls."
                    ),
                    "source": "strategy.pdf",
                    "page": 7,
                    "metadata": {
                        "section": "Enterprise Features",
                    },
                },
            ],
            "confidence": 0.92,
            "success": True,
            "failure_count": 0,
            "retrieved_count": 8,
            "context_count": 2,
            "grounding": {
                "valid": True,
                "claim_count": 2,
                "supported_claim_count": 2,
                "grounding_score": 0.94,
                "uncited_claims": [],
                "unsupported_claims": [],
                "invalid_citations": [],
            },
            "metadata": {
                "retrieval_strategy": "hybrid_rrf",
                "citation_valid": True,
                "generation_mode": "extractive",
                "generation_provider": "extractive",
                "generation_model": "extractive",
                "grounding_valid": True,
            },
        }


def test_homepage_and_static_assets() -> None:
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert (
        "Enterprise Research Intelligence"
        in response.text
    )
    assert "/static/app.js" in response.text
    assert "/static/styles.css" in response.text

    js_response = client.get(
        "/static/app.js"
    )

    assert js_response.status_code == 200

    js = js_response.text

    assert "focusCitation" in js
    assert "scrollIntoView" in js
    assert "citation-${citationId}" in js

    print("Homepage and static assets: passed")


def test_ask_endpoint_exposes_citation_contract() -> None:
    import src.rag_engine.app.api as api_module

    original_service = (
        api_module.question_service
    )

    api_module.question_service = (
        FakeQuestionService()
    )

    try:
        client = TestClient(app)

        response = client.post(
            "/documents/doc-1/ask",
            json={
                "question": (
                    "What changed in the product strategy?"
                )
            },
        )

        assert response.status_code == 200

        payload = response.json()

        assert payload["answer"]
        assert payload["citations"]

        first = payload["citations"][0]

        assert first["citation_id"] == "C1"
        assert first["chunk_id"] == "chunk-1"
        assert first["text"]
        assert first["source"] == "strategy.pdf"
        assert first["page"] == 4
        assert (
            first["metadata"]["section"]
            == "Product Strategy"
        )

        assert payload["grounding"]["valid"] is True
        assert (
            payload["metadata"]["citation_valid"]
            is True
        )

        print(
            "Citation API contract: passed"
        )
    finally:
        api_module.question_service = (
            original_service
        )


def test_ui_files_are_present_on_disk() -> None:
    static_dir = (
        Path(
            "src/rag_engine/app/static"
        )
    )

    assert (
        static_dir / "index.html"
    ).is_file()

    assert (
        static_dir / "app.js"
    ).is_file()

    assert (
        static_dir / "styles.css"
    ).is_file()

    print("UI files on disk: passed")


if __name__ == "__main__":
    test_homepage_and_static_assets()
    test_ask_endpoint_exposes_citation_contract()
    test_ui_files_are_present_on_disk()
    print("Citation UI tests passed.")
