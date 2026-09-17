from dataclasses import dataclass
from datetime import date
import hashlib
from typing import Any


@dataclass(frozen=True)
class DocumentVersion:
    document_id: str
    version_id: str
    content_hash: str
    document_date: date | None
    metadata: dict[str, Any]


class DocumentVersionManager:
    """
    Creates deterministic document versions from content.
    """

    @staticmethod
    def content_hash(
        content: str,
    ) -> str:
        if not isinstance(content, str):
            raise TypeError(
                "content must be a string"
            )

        return hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    def create_version(
        self,
        document_id: str,
        content: str,
        document_date: date | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DocumentVersion:
        if not document_id.strip():
            raise ValueError(
                "document_id cannot be empty"
            )

        content_hash = self.content_hash(
            content
        )

        version_id = (
            f"{document_id}:{content_hash[:16]}"
        )

        return DocumentVersion(
            document_id=document_id,
            version_id=version_id,
            content_hash=content_hash,
            document_date=document_date,
            metadata=dict(metadata or {}),
        )

    @staticmethod
    def is_same_version(
        first: DocumentVersion,
        second: DocumentVersion,
    ) -> bool:
        return (
            first.document_id
            == second.document_id
            and first.content_hash
            == second.content_hash
        )

    @staticmethod
    def is_stale(
        document_date: date | None,
        current_date: date,
        max_age_days: int,
    ) -> bool:
        if max_age_days < 0:
            raise ValueError(
                "max_age_days cannot be negative"
            )

        if document_date is None:
            return False

        age_days = (
            current_date - document_date
        ).days

        return age_days > max_age_days