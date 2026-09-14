from dataclasses import dataclass


@dataclass
class ExpandedQuery:
    original_query: str
    expanded_query: str
    added_terms: list[str]


class QueryExpander:
    """
    Deterministic query expander.

    The first implementation uses a small domain vocabulary to broaden
    the retrieval vocabulary while preserving the original query.
    """

    DOMAIN_TERMS = {
        "pricing": [
            "pricing",
            "pricing model",
            "monetization",
            "subscription",
            "usage-based pricing",
            "price changes",
        ],
        "product": [
            "product",
            "product strategy",
            "product roadmap",
            "product portfolio",
            "features",
            "platform",
        ],
        "strategy": [
            "strategy",
            "strategic direction",
            "strategic priorities",
            "roadmap",
            "business strategy",
        ],
        "enterprise": [
            "enterprise",
            "enterprise customers",
            "large organizations",
            "business customers",
            "B2B",
        ],
        "ai": [
            "AI",
            "artificial intelligence",
            "machine learning",
            "AI features",
            "AI-assisted",
        ],
    }

    def expand(self, query: str) -> ExpandedQuery:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        original_query = " ".join(query.split())

        query_lower = original_query.lower()

        added_terms = []

        for trigger, terms in self.DOMAIN_TERMS.items():
            if trigger in query_lower:
                for term in terms:
                    if term.lower() not in query_lower:
                        added_terms.append(term)

        expanded_query = self._build_expanded_query(
            original_query,
            added_terms,
        )

        return ExpandedQuery(
            original_query=original_query,
            expanded_query=expanded_query,
            added_terms=added_terms,
        )

    @staticmethod
    def _build_expanded_query(
        original_query: str,
        added_terms: list[str],
    ) -> str:
        if not added_terms:
            return original_query

        return f"{original_query} " + " ".join(added_terms)


def expand_query(query: str) -> str:
    """
    Convenience function returning only the expanded query.
    """

    expander = QueryExpander()

    result = expander.expand(query)

    return result.expanded_query