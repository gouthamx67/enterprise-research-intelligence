from dataclasses import dataclass, field
from typing import Any


@dataclass
class Block:
    block_number: int
    text: str
    bbox: tuple
    page_number: int
    spans: list[dict] = field(default_factory=list)
    is_heading: bool = False


@dataclass
class Page:
    page_number: int
    blocks: list[Block] = field(default_factory=list)


@dataclass
class Section:
    title: str
    level: int
    blocks: list[Block] = field(default_factory=list)
    start_page: int | None = None
    end_page: int | None = None


@dataclass
class Document:
    document_id: str
    source: str | None = None
    source_type: str | None = None
    source_locator: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[Page] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)


@dataclass
class Chunk:
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