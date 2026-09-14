from dataclasses import dataclass


@dataclass
class MultiQueryResult:
    original_query: str
    queries: list[str]


class MultiQueryGenerator:
    """
    Deterministic multi-query generator.

    Generates several retrieval-oriented perspectives from
    the original user query.
    """

    def generate(self, query: str) -> MultiQueryResult:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        original_query = " ".join(query.split())

        queries = self._generate_queries(original_query)

        return MultiQueryResult(
            original_query=original_query,
            queries=queries,
        )

    def _generate_queries(self, query: str) -> list[str]:
        query_lower = query.lower()

        queries = [
            query,
        ]

        if "product strategy" in query_lower:
            queries.extend(
                [
                    "What major product strategy changes occurred?",
                    "How did the company's product roadmap change?",
                    "What new products or features were introduced?",
                    "How did product positioning or target customers change?",
                ]
            )

        elif "pricing" in query_lower:
            queries.extend(
                [
                    "What changes were made to the company's pricing?",
                    "How did the company's pricing model evolve?",
                    "What new monetization strategies were introduced?",
                ]
            )

        elif "revenue" in query_lower:
            queries.extend(
                [
                    "What factors drove revenue growth or decline?",
                    "How did the company's revenue performance change?",
                    "What business segments contributed to revenue changes?",
                ]
            )

        else:
            queries.extend(
                [
                    f"What are the main changes related to: {query}?",
                    f"What developments are described regarding: {query}?",
                ]
            )

        return self._deduplicate(queries)

    @staticmethod
    def _deduplicate(queries: list[str]) -> list[str]:
        seen = set()
        unique_queries = []

        for query in queries:
            normalized = query.strip().lower()

            if normalized not in seen:
                seen.add(normalized)
                unique_queries.append(query.strip())

        return unique_queries


def generate_queries(query: str) -> list[str]:
    """
    Convenience function returning only the generated queries.
    """

    generator = MultiQueryGenerator()

    result = generator.generate(query)

    return result.queries