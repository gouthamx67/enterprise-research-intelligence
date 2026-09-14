from src.rag_engine.advanced.multimodal import MultimodalStore


def main():
    store = MultimodalStore()

    store.add(
        evidence_id="text-001",
        modality="text",
        content="Product strategy changed.",
        source="report.pdf",
    )

    store.add(
        evidence_id="table-001",
        modality="table",
        content="Enterprise revenue increased.",
        source="report.pdf",
    )

    store.add(
        evidence_id="chart-001",
        modality="chart",
        content="Enterprise ARR chart.",
        source="earnings.pdf",
    )

    assert len(store.search()) == 3
    assert len(store.search("table")) == 1
    assert len(store.search("chart")) == 1
    assert len(store.search("text")) == 1

    try:
        store.add(
            evidence_id="bad",
            modality="audio",
            content="unsupported",
            source="test",
        )
        raise AssertionError(
            "unsupported modality should raise ValueError"
        )
    except ValueError:
        pass

    print("Multimodal RAG test passed.")


if __name__ == "__main__":
    main()