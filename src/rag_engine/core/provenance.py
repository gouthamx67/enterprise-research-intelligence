from dataclasses import dataclass


@dataclass(frozen=True)
class Provenance:
    """
    Identifies where a piece of extracted content came from.
    """

    document_id: str
    source: str
    source_type: str
    page_number: int
    block_number: int