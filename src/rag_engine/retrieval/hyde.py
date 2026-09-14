from dataclasses import dataclass
from typing import Any

from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.models import RetrievalResult


@dataclass
class HypotheticalDocument:
    query: str
    text: str


class HypotheticalDocumentGenerator:
    """
    Deterministic hypothetical-document generator.

    In a production HyDE pipeline, this component would normally
    use an LLM to generate a hypothetical answer/document.
    """

    def generate(self, query: str) -> HypotheticalDocument:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        normalized_query = " ".join(query.split())

        query_lower = normalized_query.lower()

        if "product strategy" in query_lower:
            text = (
                "The company's product strategy changed through "
                "new product capabilities, changes to the product "
                "roadmap, expansion into new customer segments, "
                "and changes in product positioning."
            )

        elif "pricing" in query_lower:
            text = (
                "The company's pricing strategy changed through "
                "new pricing models, monetization approaches, "
                "subscription structures, or usage-based pricing."
            )

        elif "revenue" in query_lower:
            text = (
                "The company's revenue changed because of changes "
                "in customer adoption, product performance, business "
                "segments, pricing, and overall demand."
            )

        else:
            text = (
                f"The following document describes developments "
                f"related to {normalized_query}. It discusses "
                f"important changes, decisions, and outcomes "
                f"associated with the topic."
            )

        return HypotheticalDocument(
            query=normalized_query,
            text=text,
        )


class HyDERetriever:
    """
    HyDE retriever.

    Generates a hypothetical document, embeds that document using
    the dense index's embedding model, and searches the real chunk
    embeddings using the hypothetical-document vector.
    """

    def __init__(
        self,
        dense_index: DenseIndex,
        generator: HypotheticalDocumentGenerator | None = None,
    ):
        if dense_index is None:
            raise ValueError("dense_index cannot be None")

        if dense_index.size == 0:
            raise ValueError("dense_index cannot be empty")

        self.dense_index = dense_index

        self.generator = (
            generator
            if generator is not None
            else HypotheticalDocumentGenerator()
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
        score_threshold: float | None = None,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        hypothetical_document = self.generator.generate(query)

        hypothetical_embedding = (
            self.dense_index.embedding_model.encode_one(
                hypothetical_document.text
            )
        )

        return self.dense_index.search_by_vector(
            hypothetical_embedding,
            top_k=top_k,
            filters=filters,
            score_threshold=score_threshold,
        )