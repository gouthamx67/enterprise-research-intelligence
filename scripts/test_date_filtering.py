from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.filters import (
    filter_chunks,
)


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="old-001",
            text="Old product strategy.",
            document_id="acme-old",
            source="old-report.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            metadata={
                "company": "Acme",
                "document_date": "2024-06-15",
            },
        ),
        build_chunk(
            chunk_id="recent-001",
            text="Recent product strategy change.",
            document_id="acme-recent",
            source="recent-report.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            metadata={
                "company": "Acme",
                "document_date": "2025-11-20",
            },
        ),
        build_chunk(
            chunk_id="recent-002",
            text="Recent AI strategy update.",
            document_id="acme-ai",
            source="ai-report.pdf",
            source_type="pdf",
            section_title="AI Strategy",
            metadata={
                "company": "Acme",
                "document_date": "2026-02-10",
            },
        ),
        build_chunk(
            chunk_id="future-001",
            text="Future strategy.",
            document_id="acme-future",
            source="future-report.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            metadata={
                "company": "Acme",
                "document_date": "2027-01-10",
            },
        ),
        build_chunk(
            chunk_id="globex-001",
            text="Globex recent strategy.",
            document_id="globex-report",
            source="globex-report.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            metadata={
                "company": "Globex",
                "document_date": "2026-01-15",
            },
        ),
    ]


def print_chunks(
    title,
    chunks,
):
    print(f"\n{title}")
    print("-" * len(title))

    for chunk in chunks:
        print(
            f"{chunk.chunk_id} | "
            f"company={chunk.metadata.get('company')} | "
            f"date={chunk.metadata.get('document_date')}"
        )


def main():
    chunks = create_demo_chunks()

    print("=" * 70)
    print("DATE-RANGE FILTERING")
    print("=" * 70)

    print_chunks(
        "All chunks",
        chunks,
    )

    # ---------------------------------------------------------
    # Test 1: Date range only
    # ---------------------------------------------------------

    recent_chunks = filter_chunks(
        chunks,
        {
            "document_date": {
                "gte": "2025-09-01",
                "lte": "2026-09-01",
            }
        },
    )

    print_chunks(
        "Date range: 2025-09-01 through 2026-09-01",
        recent_chunks,
    )

    # ---------------------------------------------------------
    # Test 2: Company + date range
    # ---------------------------------------------------------

    acme_recent = filter_chunks(
        chunks,
        {
            "company": "Acme",
            "document_date": {
                "gte": "2025-09-01",
                "lte": "2026-09-01",
            },
        },
    )

    print_chunks(
        "Acme + date range",
        acme_recent,
    )

    # ---------------------------------------------------------
    # Assertions
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("ASSERTIONS")
    print("=" * 70)

    recent_ids = {
        chunk.chunk_id
        for chunk in recent_chunks
    }

    # Date filter alone should include BOTH companies.
    assert recent_ids == {
        "recent-001",
        "recent-002",
        "globex-001",
    }

    acme_recent_ids = {
        chunk.chunk_id
        for chunk in acme_recent
    }

    # Company + date filter should include only Acme.
    assert acme_recent_ids == {
        "recent-001",
        "recent-002",
    }

    print(
        "\nAll date filtering assertions passed."
    )


if __name__ == "__main__":
    main()