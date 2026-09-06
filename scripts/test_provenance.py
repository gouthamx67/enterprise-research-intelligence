from src.rag_engine.core.provenance import Provenance


def main():
    provenance = Provenance(
        document_id="doc-001",
        source="your_document.pdf",
        source_type="pdf",
        page_number=2,
        block_number=3,
    )

    print("PROVENANCE")
    print("===========")
    print("Document ID:", provenance.document_id)
    print("Source:", provenance.source)
    print("Source type:", provenance.source_type)
    print("Page:", provenance.page_number)
    print("Block:", provenance.block_number)


if __name__ == "__main__":
    main()