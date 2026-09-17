from datetime import date
import re

from src.rag_engine.failures.models import (
    FailureFinding,
    FailureType,
)
from src.rag_engine.retrieval.models import RetrievalResult


def detect_retrieval_failure(
    results: list[RetrievalResult],
) -> FailureFinding | None:
    if results:
        return None

    return FailureFinding(
        failure_type=FailureType.RETRIEVAL_FAILURE,
        severity="high",
        message="The retriever returned no results.",
    )


def detect_insufficient_evidence(
    results: list[RetrievalResult],
    minimum_score: float = 0.50,
    minimum_results: int = 2,
) -> FailureFinding | None:
    relevant = [
        result
        for result in results
        if result.score >= minimum_score
    ]

    if len(relevant) >= minimum_results:
        return None

    return FailureFinding(
        failure_type=FailureType.INSUFFICIENT_EVIDENCE,
        severity="high",
        message=(
            "The retrieved evidence does not contain "
            "enough high-quality results."
        ),
        evidence=[
            result.chunk_id
            for result in results
        ],
        metadata={
            "relevant_count": len(relevant),
            "minimum_results": minimum_results,
            "minimum_score": minimum_score,
        },
    )


def detect_context_overload(
    results: list[RetrievalResult],
    max_chunks: int = 10,
    max_characters: int = 12000,
) -> FailureFinding | None:
    total_characters = sum(
        len(result.text)
        for result in results
    )

    if (
        len(results) <= max_chunks
        and total_characters <= max_characters
    ):
        return None

    return FailureFinding(
        failure_type=FailureType.CONTEXT_OVERLOAD,
        severity="medium",
        message=(
            "The candidate context exceeds the configured "
            "context budget."
        ),
        metadata={
            "chunk_count": len(results),
            "max_chunks": max_chunks,
            "characters": total_characters,
            "max_characters": max_characters,
        },
    )


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_duplicate_evidence(
    results: list[RetrievalResult],
) -> FailureFinding | None:
    seen: dict[str, str] = {}
    duplicates: list[str] = []

    for result in results:
        normalized = normalize_text(result.text)

        if not normalized:
            continue

        if normalized in seen:
            duplicates.append(result.chunk_id)
        else:
            seen[normalized] = result.chunk_id

    if not duplicates:
        return None

    return FailureFinding(
        failure_type=FailureType.DUPLICATE_EVIDENCE,
        severity="low",
        message="Duplicate evidence was detected.",
        evidence=duplicates,
        metadata={
            "duplicate_count": len(duplicates),
        },
    )


def detect_wrong_context(
    query: str,
    results: list[RetrievalResult],
) -> FailureFinding | None:
    if not query.strip() or not results:
        return None

    query_tokens = set(
        normalize_text(query).split()
    )

    matching_results = []

    for result in results:
        result_tokens = set(
            normalize_text(result.text).split()
        )

        if query_tokens & result_tokens:
            matching_results.append(result)

    if matching_results:
        return None

    return FailureFinding(
        failure_type=FailureType.WRONG_CONTEXT,
        severity="high",
        message=(
            "Retrieved evidence has no obvious lexical "
            "relationship to the query."
        ),
        evidence=[
            result.chunk_id
            for result in results
        ],
    )


def detect_missing_context(
    query: str,
    results: list[RetrievalResult],
) -> FailureFinding | None:
    if not query.strip():
        return None

    if not results:
        return FailureFinding(
            failure_type=FailureType.MISSING_CONTEXT,
            severity="high",
            message=(
                "No context was available for the query."
            ),
        )

    query_tokens = {
        token
        for token in normalize_text(query).split()
        if len(token) > 3
    }

    context_tokens = set(
        normalize_text(
            " ".join(result.text for result in results)
        ).split()
    )

    overlap = query_tokens & context_tokens

    if len(overlap) >= min(2, len(query_tokens)):
        return None

    return FailureFinding(
        failure_type=FailureType.MISSING_CONTEXT,
        severity="medium",
        message=(
            "The retrieved context does not appear to "
            "cover enough of the query."
        ),
        metadata={
            "query_tokens": sorted(query_tokens),
            "matched_tokens": sorted(overlap),
        },
    )


def detect_poor_chunking(
    results: list[RetrievalResult],
    max_chunk_characters: int = 5000,
) -> FailureFinding | None:
    oversized = [
        result.chunk_id
        for result in results
        if len(result.text) > max_chunk_characters
    ]

    if not oversized:
        return None

    return FailureFinding(
        failure_type=FailureType.POOR_CHUNKING,
        severity="medium",
        message=(
            "One or more retrieved chunks are unusually large."
        ),
        evidence=oversized,
        metadata={
            "max_chunk_characters": max_chunk_characters,
        },
    )


def detect_stale_information(
    results: list[RetrievalResult],
    current_date: date,
    max_age_days: int = 365,
) -> FailureFinding | None:
    stale_chunks = []

    for result in results:
        value = result.metadata.get(
            "document_date"
        )

        if not value:
            continue

        try:
            document_date = date.fromisoformat(value)
        except ValueError:
            continue

        age_days = (
            current_date - document_date
        ).days

        if age_days > max_age_days:
            stale_chunks.append(
                result.chunk_id
            )

    if not stale_chunks:
        return None

    return FailureFinding(
        failure_type=FailureType.STALE_INFORMATION,
        severity="medium",
        message=(
            "Retrieved evidence contains documents older "
            "than the configured freshness window."
        ),
        evidence=stale_chunks,
        metadata={
            "current_date": current_date.isoformat(),
            "max_age_days": max_age_days,
        },
    )


def detect_hallucination(
    answer: str,
    evidence: list[str],
    minimum_overlap: float = 0.20,
) -> FailureFinding | None:
    if not answer.strip():
        return None

    if not evidence:
        return FailureFinding(
            failure_type=FailureType.HALLUCINATION,
            severity="high",
            message=(
                "An answer was generated without supporting evidence."
            ),
        )

    answer_tokens = set(
        normalize_text(answer).split()
    )

    evidence_tokens = set(
        normalize_text(
            " ".join(evidence)
        ).split()
    )

    overlap = answer_tokens & evidence_tokens

    ratio = len(overlap) / max(
        len(answer_tokens),
        1,
    )

    if ratio >= minimum_overlap:
        return None

    return FailureFinding(
        failure_type=FailureType.HALLUCINATION,
        severity="high",
        message=(
            "The generated answer has insufficient "
            "lexical support from the evidence."
        ),
        metadata={
            "overlap_ratio": ratio,
            "minimum_overlap": minimum_overlap,
        },
    )


def detect_citation_error(
    citations: list[str],
    valid_citation_ids: set[str],
) -> FailureFinding | None:
    invalid = [
        citation
        for citation in citations
        if citation not in valid_citation_ids
    ]

    if not invalid:
        return None

    return FailureFinding(
        failure_type=FailureType.CITATION_ERROR,
        severity="high",
        message=(
            "The answer contains citations that do not "
            "map to known evidence."
        ),
        evidence=invalid,
    )