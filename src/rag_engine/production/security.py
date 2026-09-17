from dataclasses import dataclass, field

from src.rag_engine.retrieval.models import RetrievalResult


@dataclass(frozen=True)
class AccessPolicy:
    """
    Describes what a caller is allowed to retrieve.
    """

    tenant_id: str

    allowed_source_types: set[str] = field(
        default_factory=set
    )

    allowed_companies: set[str] = field(
        default_factory=set
    )


class AccessControlledRetriever:
    """
    Retrieval wrapper that applies metadata-level access control.
    """

    def __init__(
        self,
        retriever,
        policy: AccessPolicy,
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        if not policy.tenant_id.strip():
            raise ValueError(
                "tenant_id cannot be empty"
            )

        self.retriever = retriever
        self.policy = policy

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        results = self.retriever.search(
            query,
            top_k=top_k,
            filters=filters,
        )

        allowed = []

        for result in results:
            metadata = result.metadata

            tenant_id = metadata.get(
                "tenant_id"
            )

            if (
                tenant_id is not None
                and tenant_id != self.policy.tenant_id
            ):
                continue

            source_type = metadata.get(
                "source_type"
            )

            if (
                self.policy.allowed_source_types
                and source_type
                not in self.policy.allowed_source_types
            ):
                continue

            company = metadata.get(
                "company"
            )

            if (
                self.policy.allowed_companies
                and company
                not in self.policy.allowed_companies
            ):
                continue

            allowed.append(result)

        return allowed[:top_k]