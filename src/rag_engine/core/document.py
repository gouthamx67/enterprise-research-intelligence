from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    document_id: str
    source_type: str
    source_locator: str
    metadata: dict[str, Any] = field(default_factory=dict)
    pages: list[dict[str, Any]] = field(default_factory=list)