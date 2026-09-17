from datetime import date

from src.rag_engine.production.versioning import (
    DocumentVersionManager,
)


def main():
    manager = DocumentVersionManager()

    first = manager.create_version(
        document_id="doc-1",
        content="Version one",
        document_date=date(
            2026,
            1,
            1,
        ),
    )

    same = manager.create_version(
        document_id="doc-1",
        content="Version one",
        document_date=date(
            2026,
            1,
            1,
        ),
    )

    second = manager.create_version(
        document_id="doc-1",
        content="Version two",
    )

    assert (
        manager.is_same_version(
            first,
            same,
        )
        is True
    )

    assert (
        manager.is_same_version(
            first,
            second,
        )
        is False
    )

    assert (
        manager.is_stale(
            document_date=date(
                2024,
                1,
                1,
            ),
            current_date=date(
                2026,
                1,
                1,
            ),
            max_age_days=365,
        )
        is True
    )

    assert (
        manager.is_stale(
            document_date=date(
                2025,
                12,
                1,
            ),
            current_date=date(
                2026,
                1,
                1,
            ),
            max_age_days=365,
        )
        is False
    )

    print(
        "Production versioning test passed."
    )


if __name__ == "__main__":
    main()