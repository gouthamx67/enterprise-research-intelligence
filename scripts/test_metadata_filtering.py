from src.rag_engine.chunking.build import build_chunk
from src.rag_engine.retrieval.filters import (
    filter_chunks,
)


def create_demo_chunks():
    return [
        build_chunk(
            chunk_id="acme-sec-001",
            text=(
                "Acme increased investment in "
                "artificial intelligence."
            ),
            document_id="acme-report",
            source="acme-10k.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            start_page=25,
            end_page=25,
            metadata={
                "company": "Acme",
                "document_date": "2026-01-15",
            },
        ),
        build_chunk(
            chunk_id="acme-news-001",
            text=(
                "Acme announced a new AI product "
                "during its earnings call."
            ),
            document_id="acme-news",
            source="acme-news.html",
            source_type="html",
            section_title="Product News",
            start_page=None,
            end_page=None,
            metadata={
                "company": "Acme",
                "document_date": "2026-02-01",
            },
        ),
        build_chunk(
            chunk_id="globex-sec-001",
            text=(
                "Globex increased investment in "
                "artificial intelligence."
            ),
            document_id="globex-report",
            source="globex-10k.pdf",
            source_type="pdf",
            section_title="Product Strategy",
            start_page=30,
            end_page=30,
            metadata={
                "company": "Globex",
                "document_date": "2026-01-20",
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
            f"company="
            f"{chunk.metadata.get('company')} | "
            f"source_type="
            f"{chunk.source_type}"
        )


def main():
    chunks = create_demo_chunks()

    print("=" * 60)
    print("METADATA FILTERING")
    print("=" * 60)

    print_chunks(
        "All chunks",
        chunks,
    )

    acme_chunks = filter_chunks(
        chunks,
        {
            "company": "Acme",
        },
    )

    print_chunks(
        "Company = Acme",
        acme_chunks,
    )

    acme_pdf_chunks = filter_chunks(
        chunks,
        {
            "company": "Acme",
            "source_type": "pdf",
        },
    )

    print_chunks(
        "Company = Acme AND source_type = pdf",
        acme_pdf_chunks,
    )

    globex_chunks = filter_chunks(
        chunks,
        {
            "company": "Globex",
        },
    )

    print_chunks(
        "Company = Globex",
        globex_chunks,
    )


if __name__ == "__main__":
    main()