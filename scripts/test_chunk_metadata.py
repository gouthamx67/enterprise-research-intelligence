from src.rag_engine.chunking.build import (
    build_chunk,
)


def main():
    chunk = build_chunk(
        chunk_id="acme-2026-chunk-0001",
        text=(
            "The company increased investment "
            "in AI capabilities."
        ),
        document_id="acme-2026",
        source="acme_annual_report_2026.pdf",
        source_type="pdf",
        section_title="Product Strategy",
        start_page=42,
        end_page=42,
        block_numbers=[7, 8],
        metadata={
            "company": "ACME Corporation",
            "document_date": "2026-02-15",
        },
    )

    print("CHUNK")
    print("=====")

    print("Chunk ID:", chunk.chunk_id)
    print("Text:", chunk.text)
    print("Document ID:", chunk.document_id)
    print("Source:", chunk.source)
    print("Source type:", chunk.source_type)
    print("Section:", chunk.section_title)
    print("Pages:", chunk.start_page, "-", chunk.end_page)
    print("Blocks:", chunk.block_numbers)
    print("Metadata:", chunk.metadata)


if __name__ == "__main__":
    main()