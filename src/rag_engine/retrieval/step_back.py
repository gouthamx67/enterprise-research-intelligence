from dataclasses import dataclass


@dataclass
class StepBackQuery:
    original_query: str
    step_back_query: str
    reasoning_focus: str


class StepBackGenerator:
    """
    Deterministic step-back query generator.

    The purpose is to transform a specific question into a broader
    conceptual question that can retrieve useful background context.

    A production implementation can later use an LLM to generate
    step-back questions dynamically.
    """

    def generate(self, query: str) -> StepBackQuery:
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        original_query = " ".join(query.split())

        query_lower = original_query.lower()

        if "product strategy" in query_lower:
            step_back_query = (
                "What dimensions and factors define "
                "a company's product strategy?"
            )

            reasoning_focus = (
                "product strategy dimensions, "
                "positioning, customers, products, "
                "features, roadmap, and monetization"
            )

        elif "pricing strategy" in query_lower:
            step_back_query = (
                "What factors and approaches define "
                "a company's pricing strategy?"
            )

            reasoning_focus = (
                "pricing models, monetization, "
                "subscription structures, and pricing changes"
            )

        elif "revenue" in query_lower:
            step_back_query = (
                "What factors typically drive "
                "a company's revenue?"
            )

            reasoning_focus = (
                "customers, products, pricing, "
                "business segments, and demand"
            )

        else:
            step_back_query = (
                f"What broader concepts are important "
                f"for understanding {original_query}?"
            )

            reasoning_focus = (
                "general background and conceptual context"
            )

        return StepBackQuery(
            original_query=original_query,
            step_back_query=step_back_query,
            reasoning_focus=reasoning_focus,
        )


def generate_step_back_query(
    query: str,
) -> str:
    """
    Convenience function returning only the step-back query.
    """

    generator = StepBackGenerator()

    result = generator.generate(query)

    return result.step_back_query