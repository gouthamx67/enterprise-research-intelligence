from src.rag_engine.generation.models import GenerationRequest


SYSTEM_PROMPT = """\
You are an evidence-grounded research assistant.

Your task is to answer the user's question using ONLY the evidence
provided in the user message.

Rules:

1. Do not use outside knowledge.
2. Do not invent facts, dates, numbers, events, or explanations.
3. Every factual claim that depends on the supplied evidence must have
   one or more citations in the form [C1], [C2], etc.
4. Use only citation IDs that appear in the supplied evidence.
5. If the evidence is insufficient to answer the question, explicitly
   say that the available evidence is insufficient.
6. If supplied sources disagree, explicitly describe the disagreement
   and cite the relevant sources.
7. Distinguish clearly between what the evidence states and any
   synthesis that combines multiple pieces of evidence.
8. Prefer a concise, direct answer.
9. Do not create a references section unless requested.
10. Never fabricate a citation.

The citation identifiers are part of the evidence and must be preserved
exactly, such as [C1] or [C2].
"""


def build_user_prompt(request: GenerationRequest) -> str:
    """
    Build the user-facing prompt containing the question and evidence.
    """

    evidence_sections: list[str] = []

    for context in request.contexts:
        location = context.source

        if context.page is not None:
            if location:
                location = f"{location}, page {context.page}"
            else:
                location = f"page {context.page}"

        if not location:
            location = "source location unavailable"

        evidence_sections.append(
            "\n".join(
                [
                    f"[{context.citation_id}]",
                    f"Source: {location}",
                    "Evidence:",
                    context.text.strip(),
                ]
            )
        )

    evidence = "\n\n".join(evidence_sections)

    return (
        "Question:\n"
        f"{request.question.strip()}\n\n"
        "Evidence:\n"
        f"{evidence}\n\n"
        "Answer the question using only the evidence above. "
        "Cite factual claims with the supplied citation IDs."
    )