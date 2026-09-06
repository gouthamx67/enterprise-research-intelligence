from src.rag_engine.core.document import (
    Block,
    Section,
)


def build_sections(blocks):
    """
    Build logical sections from an ordered sequence of blocks.

    A heading starts a new section.

    Blocks before the first heading are placed into
    an introductory section.
    """

    sections = []

    current_section = None

    for block in blocks:

        if block.is_heading:

            current_section = Section(
                title=block.text,
                level=1,
                blocks=[],
                start_page=block.page_number,
                end_page=block.page_number,
            )

            sections.append(current_section)

            continue

        if current_section is None:

            current_section = Section(
                title="Introduction",
                level=1,
                blocks=[],
                start_page=block.page_number,
                end_page=block.page_number,
            )

            sections.append(current_section)

        current_section.blocks.append(block)

        current_section.end_page = block.page_number

    return sections