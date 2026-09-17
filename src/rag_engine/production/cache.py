from dataclasses import dataclass
from time import time
from typing import Any


@dataclass
class CacheEntry:
    value: Any
    created_at: float


class QueryCache:
    """
    In-memory TTL cache for completed RAG responses.
    """

    def __init__(
        self,
        ttl_seconds: int = 300,
        max_entries: int = 1000,
    ):
        if ttl_seconds < 0:
            raise ValueError(
                "ttl_seconds cannot be negative"
            )

        if max_entries <= 0:
            raise ValueError(
                "max_entries must be greater than 0"
            )

        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: dict[str, CacheEntry] = {}

    def _expired(
        self,
        entry: CacheEntry,
    ) -> bool:
        return (
            time() - entry.created_at
            > self.ttl_seconds
        )

    def get(
        self,
        key: str,
    ) -> Any | None:
        entry = self._entries.get(key)

        if entry is None:
            return None

        if self._expired(entry):
            del self._entries[key]
            return None

        return entry.value

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        if len(self._entries) >= self.max_entries:
            oldest_key = min(
                self._entries,
                key=lambda item: (
                    self._entries[item].created_at
                ),
            )

            del self._entries[oldest_key]

        self._entries[key] = CacheEntry(
            value=value,
            created_at=time(),
        )

    def delete(
        self,
        key: str,
    ) -> None:
        self._entries.pop(
            key,
            None,
        )

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)