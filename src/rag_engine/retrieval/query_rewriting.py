from dataclasses import dataclass
import re


@dataclass
class RewrittenQuery:
    original_query: str
    rewritten_query: str
    transformations: list[str]


class QueryRewriter:
    """
    Deterministic query rewriter.

    This first version focuses on understanding the query-transformation
    interface without depending on an LLM.
    """

    def rewrite(self, query: str) -> RewrittenQuery:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        original_query = query.strip()

        rewritten_query = self._rewrite_phrases(original_query)

        transformations = []

        if rewritten_query != original_query:
            transformations.append("phrase_normalization")

        rewritten_query = self._normalize_whitespace(
            rewritten_query
        )

        if rewritten_query != original_query and (
            "phrase_normalization" not in transformations
        ):
            transformations.append("whitespace_normalization")

        return RewrittenQuery(
            original_query=original_query,
            rewritten_query=rewritten_query,
            transformations=transformations,
        )

    def _rewrite_phrases(self, query: str) -> str:
        replacements = {
            "what changed": "changes",
            "over the last 12 months": "during the past 12 months",
            "last 12 months": "past 12 months",
            "product strategy": "product strategy roadmap",
        }

        rewritten = query

        for source, target in replacements.items():
            rewritten = re.sub(
                re.escape(source),
                target,
                rewritten,
                flags=re.IGNORECASE,
            )

        return rewritten

    @staticmethod
    def _normalize_whitespace(query: str) -> str:
        return " ".join(query.split())


def rewrite_query(query: str) -> str:
    """
    Convenience function that returns only the rewritten query.
    """

    rewriter = QueryRewriter()

    result = rewriter.rewrite(query)

    return result.rewritten_query