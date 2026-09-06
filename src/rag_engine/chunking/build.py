from src.rag_engine.core.document import Chunk


def build_chunk(
    chunk_id: str,
    text: str,
    document_id: str,
    source: str,
    source_type: str,
    section_title: str | None = None,
    start_page: int | None = None,
    end_page: int | None = None,
    block_numbers: list[int] | None = None,
    metadata: dict | None = None,
):
    """
    Construct a retrieval chunk while preserving
    document metadata and source information.
    """

    text = text.strip()

    if not text:
        raise ValueError("chunk text cannot be empty")

    return Chunk(
        chunk_id=chunk_id,
        text=text,
        document_id=document_id,
        source=source,
        source_type=source_type,
        section_title=section_title,
        start_page=start_page,
        end_page=end_page,
        block_numbers=list(block_numbers or []),
        metadata=dict(metadata or {}),
    )