from src.rag_engine.production.security import (
    AccessControlledRetriever,
)
from src.rag_engine.retrieval.models import (
    RetrievalResult,
)


class FakeRetriever:
    def search(
        self,
        query,
        top_k=5,
        filters=None,
    ):
        return [
            RetrievalResult(
                score=0.9,
                rank=1,
                chunk_id="allowed",
                text="Allowed evidence.",
                metadata={
                    "tenant_id": "tenant-a",
                    "company": "Example",
                    "source_type": "sec",
                },
            ),
            RetrievalResult(
                score=0.8,
                rank=2,
                chunk_id="blocked",
                text="Blocked evidence.",
                metadata={
                    "tenant_id": "tenant-b",
                    "company": "Other",
                    "source_type": "sec",
                },
            ),
        ]


def main():
    from src.rag_engine.production.security import (
        AccessPolicy,
    )

    policy = AccessPolicy(
        tenant_id="tenant-a",
        allowed_source_types={"sec"},
        allowed_companies={"Example"},
    )

    retriever = AccessControlledRetriever(
        FakeRetriever(),
        policy,
    )

    results = retriever.search(
        "product strategy",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "allowed"

    print(
        "Production security test passed."
    )


if __name__ == "__main__":
    main()