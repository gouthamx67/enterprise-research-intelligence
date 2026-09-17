from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DocumentVersion:
    document_id: str
    version: str
    document_date: date
    indexed_at: date


@dataclass(frozen=True)
class FreshnessResult:
    document_id: str
    is_fresh: bool
    age_days: int
    reason: str


class FreshnessChecker:
    def __init__(
        self,
        max_age_days: int = 365,
    ):
        if max_age_days < 0:
            raise ValueError(
                "max_age_days cannot be negative"
            )

        self.max_age_days = max_age_days

    def check(
        self,
        version: DocumentVersion,
        current_date: date,
    ) -> FreshnessResult:
        age_days = (
            current_date
            - version.document_date
        ).days

        is_fresh = (
            age_days <= self.max_age_days
        )

        if is_fresh:
            reason = "document is within freshness window"
        else:
            reason = "document exceeds freshness window"

        return FreshnessResult(
            document_id=version.document_id,
            is_fresh=is_fresh,
            age_days=age_days,
            reason=reason,
        )


class VersionRegistry:
    """
    Tracks the currently indexed version for each document.
    """

    def __init__(self):
        self._versions: dict[
            str,
            DocumentVersion,
        ] = {}

    def register(
        self,
        version: DocumentVersion,
    ) -> None:
        self._versions[
            version.document_id
        ] = version

    def get(
        self,
        document_id: str,
    ) -> DocumentVersion | None:
        return self._versions.get(
            document_id
        )

    def needs_reindex(
        self,
        document_id: str,
        version: str,
    ) -> bool:
        existing = self.get(document_id)

        if existing is None:
            return True

        return existing.version != version