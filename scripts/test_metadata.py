from src.rag_engine.preprocessing.metadata import (
    build_document_metadata,
)


def main():
    metadata = build_document_metadata(
        document_id="acme_10k_2026",
        source="data/raw/pdf/acme_10k_2026.pdf",
        source_type="pdf",
        title="ACME Corporation Annual Report 2026",
        company="ACME Corporation",
        document_date="2026-02-15",
    )

    print("DOCUMENT METADATA")
    print("=================")

    for key, value in metadata.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()