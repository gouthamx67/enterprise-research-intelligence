from datetime import date

from src.rag_engine.production.freshness import (
    DocumentVersion,
    FreshnessChecker,
    VersionRegistry,
)


def main():
    version = DocumentVersion(
        document_id="doc1",
        version="v2",
        document_date=date(
            2026,
            8,
            1,
        ),
        indexed_at=date(
            2026,
            9,
            17,
        ),
    )

    checker = FreshnessChecker(
        max_age_days=365
    )

    result = checker.check(
        version,
        date(
            2026,
            9,
            17,
        ),
    )

    assert result.is_fresh is True
    assert result.age_days == 47

    registry = VersionRegistry()

    assert registry.needs_reindex(
        "doc1",
        "v1",
    )

    registry.register(version)

    assert not registry.needs_reindex(
        "doc1",
        "v2",
    )

    assert registry.needs_reindex(
        "doc1",
        "v3",
    )

    print(
        "Production freshness test passed."
    )


if __name__ == "__main__":
    main()