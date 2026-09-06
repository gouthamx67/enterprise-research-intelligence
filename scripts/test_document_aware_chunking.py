from src.rag_engine.chunking.document_aware import (
    document_aware_chunks,
)
from src.rag_engine.core.document import (
    Block,
    Section,
)


def create_test_sections():
    introduction = Section(
        title="Introduction",
        level=1,
        blocks=[
            Block(
                block_number=1,
                text="The company develops enterprise software.",
                bbox=(0, 0, 100, 20),
                page_number=1,
            ),
            Block(
                block_number=2,
                text="The company serves customers globally.",
                bbox=(0, 20, 100, 40),
                page_number=1,
            ),
        ],
        start_page=1,
        end_page=1,
    )

    financial_results = Section(
        title="Financial Results",
        level=1,
        blocks=[
            Block(
                block_number=3,
                text="Revenue increased by 18%.",
                bbox=(0, 0, 100, 20),
                page_number=2,
            ),
            Block(
                block_number=4,
                text="Operating income increased by 12%.",
                bbox=(0, 20, 100, 40),
                page_number=2,
            ),
        ],
        start_page=2,
        end_page=2,
    )

    return [
        introduction,
        financial_results,
    ]


def main():
    sections = create_test_sections()

    chunks = document_aware_chunks(
        sections=sections,
        document_id="doc-001",
        source="example.pdf",
        source_type="pdf",
        chunk_size=80,
        metadata={
            "company": "ACME Corporation",
        },
    )

    print("DOCUMENT-AWARE CHUNKS")
    print("=====================")

    for chunk in chunks:
        print()
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Section: {chunk.section_title}")
        print(f"Pages: {chunk.start_page}-{chunk.end_page}")
        print(f"Blocks: {chunk.block_numbers}")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")


if __name__ == "__main__":
    main()