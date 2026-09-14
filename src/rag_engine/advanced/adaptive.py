from dataclasses import dataclass


@dataclass(frozen=True)
class AdaptiveDecision:
    query: str
    strategy: str
    reason: str


class AdaptiveRouter:
    """
    Deterministic query router for adaptive RAG.

    The router chooses an appropriate retrieval strategy
    based on the type of information requested.
    """

    def route(self, query: str) -> AdaptiveDecision:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        normalized = query.strip().lower()

        # -----------------------------------------------------
        # Graph-oriented questions
        # -----------------------------------------------------
        #
        # These questions involve relationships between
        # entities: acquisitions, partnerships, competitors,
        # subsidiaries, etc.
        #

        graph_terms = (
            "acquire",
            "acquired",
            "acquires",
            "acquisition",
            "acquisitions",
            "subsidiary",
            "subsidiaries",
            "relationship",
            "relationships",
            "partner",
            "partners",
            "partnership",
            "partnerships",
            "competitor",
            "competitors",
            "compete",
            "competes",
            "ownership",
            "owned by",
        )

        if any(
            term in normalized
            for term in graph_terms
        ):
            return AdaptiveDecision(
                query=query.strip(),
                strategy="graph",
                reason=(
                    "The query asks about relationships "
                    "between entities."
                ),
            )

        # -----------------------------------------------------
        # Multimodal questions
        # -----------------------------------------------------

        multimodal_terms = (
            "image",
            "images",
            "diagram",
            "diagrams",
            "chart",
            "charts",
            "screenshot",
            "screenshots",
            "table",
            "tables",
            "visual",
            "visuals",
        )

        if any(
            term in normalized
            for term in multimodal_terms
        ):
            return AdaptiveDecision(
                query=query.strip(),
                strategy="multimodal",
                reason=(
                    "The query may require visual or "
                    "structured-document evidence."
                ),
            )

        # -----------------------------------------------------
        # Product strategy questions
        # -----------------------------------------------------

        if "product strategy" in normalized:
            return AdaptiveDecision(
                query=query.strip(),
                strategy="hybrid",
                reason=(
                    "Product strategy questions benefit "
                    "from semantic and lexical retrieval."
                ),
            )

        # -----------------------------------------------------
        # Exact / lexical questions
        # -----------------------------------------------------

        sparse_terms = (
            "exact",
            "defined as",
            "definition",
            "identifier",
        )

        if any(
            term in normalized
            for term in sparse_terms
        ):
            return AdaptiveDecision(
                query=query.strip(),
                strategy="sparse",
                reason=(
                    "Exact terminology benefits from "
                    "lexical retrieval."
                ),
            )

        # -----------------------------------------------------
        # Default semantic retrieval
        # -----------------------------------------------------

        return AdaptiveDecision(
            query=query.strip(),
            strategy="dense",
            reason=(
                "General semantic questions are suitable "
                "for dense retrieval."
            ),
        )