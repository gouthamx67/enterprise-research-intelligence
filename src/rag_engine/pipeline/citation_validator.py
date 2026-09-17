from dataclasses import dataclass


@dataclass
class CitationValidationResult:
    valid: bool
    valid_citations: list[str]
    invalid_citations: list[str]


class CitationValidator:
    """
    Validates generated citation IDs against the citations
    available in the constructed context.
    """

    def validate(
        self,
        generated_citations: list[str],
        valid_citation_ids: set[str],
    ) -> CitationValidationResult:
        invalid = [
            citation
            for citation in generated_citations
            if citation not in valid_citation_ids
        ]

        valid = [
            citation
            for citation in generated_citations
            if citation in valid_citation_ids
        ]

        return CitationValidationResult(
            valid=len(invalid) == 0,
            valid_citations=valid,
            invalid_citations=invalid,
        )