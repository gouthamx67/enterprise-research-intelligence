import re
from dataclasses import dataclass, field
from typing import Mapping


_CITATION_PATTERN = re.compile(
    r"\[(C\d+)\]",
    re.IGNORECASE,
)

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)?"
)


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "which",
    "who",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class GroundingValidationResult:
    valid: bool
    claim_count: int
    supported_claim_count: int
    grounding_score: float
    uncited_claims: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "claim_count": self.claim_count,
            "supported_claim_count": self.supported_claim_count,
            "grounding_score": round(
                self.grounding_score,
                3,
            ),
            "uncited_claims": list(
                self.uncited_claims
            ),
            "unsupported_claims": list(
                self.unsupported_claims
            ),
            "invalid_citations": list(
                self.invalid_citations
            ),
        }


class GroundingValidator:
    """
    Deterministic claim-to-evidence grounding validator.

    Each answer fragment is treated as a claim. A claim must:

    1. contain at least one citation;
    2. cite an existing evidence item;
    3. share enough non-stopword terms with the cited evidence.

    This is intentionally lexical rather than semantic. It catches
    obvious unsupported claims without pretending to prove semantic
    factuality.
    """

    def validate(
        self,
        answer: str,
        evidence_by_citation: Mapping[str, str],
    ) -> GroundingValidationResult:
        if not answer or not answer.strip():
            return GroundingValidationResult(
                valid=False,
                claim_count=0,
                supported_claim_count=0,
                grounding_score=0.0,
                uncited_claims=[],
                unsupported_claims=[],
                invalid_citations=[],
            )

        normalized_evidence = {
            citation_id.upper(): evidence
            for citation_id, evidence
            in evidence_by_citation.items()
        }

        claims = self._split_claims(answer)

        uncited_claims: list[str] = []
        unsupported_claims: list[str] = []
        invalid_citations: list[str] = []

        supported_claim_count = 0
        scores: list[float] = []

        for claim in claims:
            claim_text = _CITATION_PATTERN.sub(
                "",
                claim,
            ).strip()

            citations = [
                match.group(1).upper()
                for match in _CITATION_PATTERN.finditer(
                    claim
                )
            ]

            if not citations:
                uncited_claims.append(
                    claim_text
                )
                continue

            missing = [
                citation_id
                for citation_id in citations
                if citation_id not in normalized_evidence
            ]

            for citation_id in missing:
                if citation_id not in invalid_citations:
                    invalid_citations.append(
                        citation_id
                    )

            valid_citations = [
                citation_id
                for citation_id in citations
                if citation_id in normalized_evidence
            ]

            if not valid_citations:
                continue

            claim_terms = self._content_terms(
                claim_text
            )

            cited_evidence = " ".join(
                normalized_evidence[citation_id]
                for citation_id in valid_citations
            )

            evidence_terms = self._content_terms(
                cited_evidence
            )

            overlap = len(
                claim_terms.intersection(
                    evidence_terms
                )
            )

            coverage = (
                overlap / max(len(claim_terms), 1)
            )

            scores.append(
                min(1.0, coverage)
            )

            # A 50% lexical coverage threshold is intentionally
            # conservative enough to reject clearly unrelated
            # claims while still allowing straightforward paraphrase.
            supported = (
                bool(claim_terms)
                and overlap >= 1
                and coverage >= 0.5
            )

            if supported:
                supported_claim_count += 1
            else:
                unsupported_claims.append(
                    claim_text
                )

        claim_count = len(claims)

        grounding_score = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

        valid = (
            claim_count > 0
            and supported_claim_count == claim_count
            and not uncited_claims
            and not unsupported_claims
            and not invalid_citations
        )

        return GroundingValidationResult(
            valid=valid,
            claim_count=claim_count,
            supported_claim_count=supported_claim_count,
            grounding_score=grounding_score,
            uncited_claims=uncited_claims,
            unsupported_claims=unsupported_claims,
            invalid_citations=invalid_citations,
        )

    @staticmethod
    def _split_claims(
        answer: str,
    ) -> list[str]:
        fragments = re.split(
            r"\n+|(?<=[.!?])\s+",
            answer.strip(),
        )

        return [
            fragment.strip()
            for fragment in fragments
            if fragment.strip()
        ]

    @staticmethod
    def _content_terms(
        text: str,
    ) -> set[str]:
        terms: set[str] = set()

        for token in _TOKEN_PATTERN.findall(
            text.lower()
        ):
            if token in _STOPWORDS:
                continue

            if len(token) <= 2:
                continue

            terms.add(token)

        return terms
