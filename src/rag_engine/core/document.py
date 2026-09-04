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