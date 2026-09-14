from dataclasses import dataclass, field


@dataclass
class RetrievalStep:
    query: str
    strategy: str
    reason: str


@dataclass
class AgentPlan:
    original_query: str
    steps: list[RetrievalStep] = field(default_factory=list)


class AgenticPlanner:
    """
    Deterministic planner for advanced RAG.

    The planner converts a complex research question into
    a sequence of retrieval actions.
    """

    def plan(self, query: str) -> AgentPlan:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        normalized = query.strip().lower()

        steps = []

        if "product strategy" in normalized:
            steps.append(
                RetrievalStep(
                    query=query.strip(),
                    strategy="hybrid",
                    reason=(
                        "Product strategy requires semantic and "
                        "lexical retrieval."
                    ),
                )
            )

            steps.append(
                RetrievalStep(
                    query=(
                        "previous product strategy before "
                        "the current period"
                    ),
                    strategy="hybrid",
                    reason=(
                        "Historical evidence is required to "
                        "identify strategic change."
                    ),
                )
            )

            steps.append(
                RetrievalStep(
                    query=(
                        "new products features capabilities "
                        "and roadmap changes"
                    ),
                    strategy="dense",
                    reason=(
                        "Product changes may be described "
                        "using different terminology."
                    ),
                )
            )

            steps.append(
                RetrievalStep(
                    query=(
                        "target customers market positioning "
                        "and enterprise strategy"
                    ),
                    strategy="hybrid",
                    reason=(
                        "Strategy changes can appear in customer "
                        "and market positioning evidence."
                    ),
                )
            )

        else:
            steps.append(
                RetrievalStep(
                    query=query.strip(),
                    strategy="hybrid",
                    reason="Default retrieval strategy.",
                )
            )

        return AgentPlan(
            original_query=query.strip(),
            steps=steps,
        )