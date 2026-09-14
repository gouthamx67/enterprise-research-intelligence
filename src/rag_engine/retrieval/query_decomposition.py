from dataclasses import dataclass


@dataclass
class DecomposedQuery:
    original_query: str
    subqueries: list[str]


class QueryDecomposer:
    """
    Deterministic query decomposer.

    This version identifies common complex research questions
    and breaks them into focused retrieval subqueries.

    A production implementation can later use an LLM to
    generate decomposition dynamically.
    """

    def decompose(self, query: str) -> DecomposedQuery:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        original_query = " ".join(query.split())

        subqueries = self._decompose_query(
            original_query
        )

        subqueries = self._deduplicate(
            subqueries
        )

        return DecomposedQuery(
            original_query=original_query,
            subqueries=subqueries,
        )

    def _decompose_query(
        self,
        query: str,
    ) -> list[str]:
        query_lower = query.lower()

        if (
            "product strategy" in query_lower
            and "last 12 months" in query_lower
        ):
            return [
                (
                    "What was the company's product "
                    "strategy before the last 12 months?"
                ),
                (
                    "What is the company's current "
                    "product strategy?"
                ),
                (
                    "What new products, features, or "
                    "capabilities were introduced?"
                ),
                (
                    "Did the company's target customers "
                    "or market positioning change?"
                ),
                (
                    "Did the company's pricing or "
                    "monetization strategy change?"
                ),
                (
                    "What evidence establishes when "
                    "these product strategy changes occurred?"
                ),
            ]

        if "product strategy" in query_lower:
            return [
                (
                    "What is the company's product strategy?"
                ),
                (
                    "What products, features, or capabilities "
                    "are central to the product strategy?"
                ),
                (
                    "Who are the company's target customers "
                    "for these products?"
                ),
            ]

        if "pricing strategy" in query_lower:
            return [
                (
                    "What is the company's current "
                    "pricing strategy?"
                ),
                (
                    "How has the company's pricing model "
                    "changed?"
                ),
                (
                    "What monetization approaches "
                    "does the company use?"
                ),
            ]

        if "revenue" in query_lower:
            return [
                (
                    "What are the company's main sources "
                    "of revenue?"
                ),
                (
                    "What caused revenue to increase "
                    "or decrease?"
                ),
                (
                    "Which products or business segments "
                    "contributed to the revenue change?"
                ),
            ]

        return [
            f"What are the main components of: {query}?",
            f"What changed regarding: {query}?",
            f"What evidence supports the answer to: {query}?",
        ]

    @staticmethod
    def _deduplicate(
        queries: list[str],
    ) -> list[str]:
        seen = set()
        unique_queries = []

        for query in queries:
            normalized = query.strip().lower()

            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query.strip())

        return unique_queries


def decompose_query(
    query: str,
) -> list[str]:
    """
    Convenience function returning only the subqueries.
    """

    decomposer = QueryDecomposer()

    result = decomposer.decompose(query)

    return result.subqueries