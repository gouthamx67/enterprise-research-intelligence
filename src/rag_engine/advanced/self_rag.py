from dataclasses import dataclass


@dataclass(frozen=True)
class SelfRAGEvaluation:
    answer: str
    supported: bool
    useful: bool
    score: float
    feedback: str


class SelfRAGCritic:
    """
    Deterministic answer critic for Self-RAG.

    The critic evaluates whether a generated answer has
    sufficient lexical support from the supplied evidence.

    This is intentionally a simple implementation for
    learning the Self-RAG architecture. Later, this can
    be replaced with an LLM-based evaluator.
    """

    def evaluate(
        self,
        answer: str,
        evidence: list[str],
    ) -> SelfRAGEvaluation:
        if not answer or not answer.strip():
            raise ValueError(
                "answer cannot be empty"
            )

        if not evidence:
            return SelfRAGEvaluation(
                answer=answer,
                supported=False,
                useful=False,
                score=0.0,
                feedback="No evidence was provided.",
            )

        answer_tokens = set(
            answer.lower().split()
        )

        evidence_tokens = set(
            " ".join(evidence).lower().split()
        )

        overlap = answer_tokens & evidence_tokens

        score = len(overlap) / max(
            len(answer_tokens),
            1,
        )

        supported = score >= 0.20
        useful = len(answer.strip()) >= 20

        if supported and useful:
            feedback = (
                "Answer appears supported and sufficiently detailed."
            )
        elif not supported:
            feedback = (
                "Answer has insufficient lexical support "
                "from the provided evidence."
            )
        else:
            feedback = (
                "Answer has some support but may be too brief."
            )

        return SelfRAGEvaluation(
            answer=answer,
            supported=supported,
            useful=useful,
            score=score,
            feedback=feedback,
        )