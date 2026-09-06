from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    """
    A layout-aware piece of extracted document content.
    """

    block_number: int
    text: str
    bbox: tuple
    page_number: int
    spans: list[dict] = field(default_factory=list)

    is_heading: bool = False


@dataclass
class Page:
    """
    A single page of a document.
    """

    page_number: int
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Section:
    """
    A logical section of a document.

    A section begins with a heading and contains
    the blocks belonging to that heading.
    """

    title: str
    level: int
    blocks: list[Block] = field(default_factory=list)

    start_page: int | None = None
    end_page: int | None = None


@dataclass
class Document:
    """
    Internal representation of an ingested document.
    """

    document_id: str
    source: str
    source_type: str

    title: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    pages: list[Page] = field(default_factory=list)

    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
    """
    A retrieval unit created from document content.
    """

    chunk_id: str
    text: str

    document_id: str
    source: str
    source_type: str

    section_title: str | None = None

    start_page: int | None = None
    end_page: int | None = None

    block_numbers: list[int] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParentChildChunk:
    """
    Represents a small child retrieval chunk and
    the larger parent context it belongs to.
    """

    parent_id: str
    child_id: str

    parent_text: str
    child_text: str

    document_id: str
    source: str
    source_type: str

    section_title: str | None = None

    start_page: int | None = None
    end_page: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)