from pathlib import Path


def build_document_metadata(
    document_id: str,
    source: str,
    source_type: str,
    title: str | None = None,
    company: str | None = None,
    document_date: str | None = None,
) -> dict:
    """
    Build standardized metadata for an ingested document.

    Metadata describes the document itself.
    It is separate from the document's textual content
    and separate from block-level provenance.
    """

    metadata = {
        "document_id": document_id,
        "source": source,
        "source_type": source_type,
    }

    if title is not None:
        metadata["title"] = title

    if company is not None:
        metadata["company"] = company

    if document_date is not None:
        metadata["document_date"] = document_date

    metadata["file_name"] = Path(source).name

    return metadata