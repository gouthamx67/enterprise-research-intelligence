from src.rag_engine.core.document import Chunk, Section


def document_aware_chunks(
    sections: list[Section],
    document_id: str,
    source: str,
    source_type: str,
    chunk_size: int = 500,
    metadata: dict | None = None,
):
    """
    Create chunks while preserving document section boundaries.

    Blocks from different sections are never combined
    into the same chunk.

    chunk_size is measured in characters.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    chunks = []

    for section in sections:
        current_text_parts = []
        current_blocks = []
        current_start_page = None
        current_end_page = None

        def flush_chunk():
            nonlocal current_text_parts
            nonlocal current_blocks
            nonlocal current_start_page
            nonlocal current_end_page

            if not current_text_parts:
                return

            chunk_number = len(chunks) + 1

            text = "\n\n".join(current_text_parts).strip()

            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-chunk-{chunk_number:04d}",
                    text=text,
                    document_id=document_id,
                    source=source,
                    source_type=source_type,
                    section_title=section.title,
                    start_page=current_start_page,
                    end_page=current_end_page,
                    block_numbers=current_blocks.copy(),
                    metadata=dict(metadata or {}),
                )
            )

            current_text_parts = []
            current_blocks = []
            current_start_page = None
            current_end_page = None

        for block in section.blocks:
            block_text = block.text.strip()

            if not block_text:
                continue

            if current_start_page is None:
                current_start_page = block.page_number

            candidate_parts = current_text_parts + [block_text]
            candidate_text = "\n\n".join(candidate_parts)

            if (
                current_text_parts
                and len(candidate_text) > chunk_size
            ):
                flush_chunk()

                current_start_page = block.page_number

            current_text_parts.append(block_text)
            current_blocks.append(block.block_number)
            current_end_page = block.page_number

        flush_chunk()

    return chunks