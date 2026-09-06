from src.rag_engine.core.document import (
    ParentChildChunk,
)


def parent_child_chunks(
    text: str,
    document_id: str,
    source: str,
    source_type: str,
    parent_size: int = 500,
    child_size: int = 150,
    section_title: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    metadata: dict | None = None,
):
    """
    Create parent-child chunks.

    A parent contains broader context.
    Children are smaller retrieval units inside that parent.

    Both sizes are measured in characters.
    """

    if parent_size <= 0:
        raise ValueError(
            "parent_size must be greater than 0"
        )

    if child_size <= 0:
        raise ValueError(
            "child_size must be greater than 0"
        )

    if child_size > parent_size:
        raise ValueError(
            "child_size cannot be greater than parent_size"
        )

    text = text.strip()

    if not text:
        return []

    parents = [
        text[start:start + parent_size].strip()
        for start in range(0, len(text), parent_size)
        if text[start:start + parent_size].strip()
    ]

    results = []

    for parent_index, parent_text in enumerate(
        parents,
        start=1,
    ):
        parent_id = (
            f"{document_id}-parent-{parent_index:04d}"
        )

        children = [
            parent_text[start:start + child_size].strip()
            for start in range(
                0,
                len(parent_text),
                child_size,
            )
            if parent_text[
                start:start + child_size
            ].strip()
        ]

        for child_index, child_text in enumerate(
            children,
            start=1,
        ):
            child_id = (
                f"{parent_id}-child-{child_index:04d}"
            )

            results.append(
                ParentChildChunk(
                    parent_id=parent_id,
                    child_id=child_id,
                    parent_text=parent_text,
                    child_text=child_text,
                    document_id=document_id,
                    source=source,
                    source_type=source_type,
                    section_title=section_title,
                    start_page=start_page,
                    end_page=end_page,
                    metadata=dict(metadata or {}),
                )
            )

    return results