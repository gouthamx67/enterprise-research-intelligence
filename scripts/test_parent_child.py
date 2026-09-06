from src.rag_engine.chunking.parent_child import (
    parent_child_chunks,
)


TEXT = (
    "The company changed its product strategy in 2026. "
    "It increased investment in AI capabilities and "
    "reduced investment in several legacy products. "
    "The company also changed its target customers. "
    "It is now focusing more heavily on large enterprise "
    "accounts. The strategy emphasizes automation, "
    "analytics, and enterprise-grade AI capabilities."
)


def main():
    chunks = parent_child_chunks(
        text=TEXT,
        document_id="doc-001",
        source="example.pdf",
        source_type="pdf",
        parent_size=220,
        child_size=80,
        section_title="Product Strategy",
        start_page=10,
        end_page=11,
        metadata={
            "company": "ACME Corporation",
        },
    )

    print("PARENT-CHILD CHUNKS")
    print("===================")

    current_parent = None

    for chunk in chunks:
        if chunk.parent_id != current_parent:
            current_parent = chunk.parent_id

            print()
            print(f"PARENT: {chunk.parent_id}")
            print("-" * 60)
            print(chunk.parent_text)

        print()
        print(f"  CHILD: {chunk.child_id}")
        print(f"  Text: {chunk.child_text}")


if __name__ == "__main__":
    main()