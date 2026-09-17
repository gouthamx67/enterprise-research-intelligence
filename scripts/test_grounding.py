from src.rag_engine.generation.grounding import (
    GroundingValidator,
)


def test_grounded_claim_with_valid_citation() -> None:
    validator = GroundingValidator()

    result = validator.validate(
        answer=(
            "The company shifted its product strategy "
            "toward enterprise customers [C1]."
        ),
        evidence_by_citation={
            "C1": (
                "The company shifted its product strategy "
                "toward enterprise customers."
            ),
        },
    )

    assert result.valid is True
    assert result.claim_count == 1
    assert result.supported_claim_count == 1
    assert result.uncited_claims == []
    assert result.unsupported_claims == []
    assert result.invalid_citations == []
    assert result.grounding_score >= 0.8

    print("Grounded claim: passed")


def test_uncited_claim_fails() -> None:
    validator = GroundingValidator()

    result = validator.validate(
        answer=(
            "The company shifted toward enterprise customers."
        ),
        evidence_by_citation={
            "C1": (
                "The company shifted toward enterprise customers."
            ),
        },
    )

    assert result.valid is False
    assert len(result.uncited_claims) == 1

    print("Uncited claim detection: passed")


def test_unrelated_cited_claim_fails() -> None:
    validator = GroundingValidator()

    result = validator.validate(
        answer=(
            "The company acquired a semiconductor manufacturer [C1]."
        ),
        evidence_by_citation={
            "C1": (
                "The company expanded administrative controls "
                "and security features."
            ),
        },
    )

    assert result.valid is False
    assert len(result.unsupported_claims) == 1

    print("Unsupported claim detection: passed")


def test_invalid_citation_fails() -> None:
    validator = GroundingValidator()

    result = validator.validate(
        answer=(
            "The company expanded its enterprise product offering [C9]."
        ),
        evidence_by_citation={
            "C1": (
                "The company expanded its enterprise product offering."
            ),
        },
    )

    assert result.valid is False
    assert result.invalid_citations == ["C9"]

    print("Invalid citation detection: passed")


def test_multiple_grounded_claims_pass() -> None:
    validator = GroundingValidator()

    result = validator.validate(
        answer=(
            "The company introduced an enterprise-focused "
            "product tier [C1]. "
            "It also expanded security and administrative "
            "controls [C2]."
        ),
        evidence_by_citation={
            "C1": (
                "The company introduced an enterprise-focused "
                "product tier."
            ),
            "C2": (
                "The company expanded security and "
                "administrative controls."
            ),
        },
    )

    assert result.valid is True
    assert result.claim_count == 2
    assert result.supported_claim_count == 2
    assert result.grounding_score >= 0.8

    print("Multiple grounded claims: passed")


if __name__ == "__main__":
    test_grounded_claim_with_valid_citation()
    test_uncited_claim_fails()
    test_unrelated_cited_claim_fails()
    test_invalid_citation_fails()
    test_multiple_grounded_claims_pass()
    print("Grounding tests passed.")
