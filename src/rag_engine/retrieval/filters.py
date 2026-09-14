from datetime import date
from typing import Any

from src.rag_engine.core.document import Chunk


def parse_date(value: Any) -> date:
    """
    Convert a supported value into a Python date.

    Dates are expected to use ISO format:

        YYYY-MM-DD
    """

    if isinstance(value, date):
        return value

    if not isinstance(value, str):
        raise ValueError(
            "date value must be a YYYY-MM-DD string"
        )

    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid date: {value!r}; "
            "expected YYYY-MM-DD"
        ) from exc


def get_chunk_metadata_value(
    chunk: Chunk,
    key: str,
):
    """
    Retrieve a value from a Chunk.

    Direct Chunk attributes take precedence over
    values stored inside chunk.metadata.
    """

    if hasattr(chunk, key):
        return getattr(chunk, key)

    return chunk.metadata.get(key)


def matches_date_filter(
    chunk: Chunk,
    key: str,
    condition: dict[str, Any],
) -> bool:
    """
    Check a date field against one or more range predicates.

    Supported predicates:

        gte
        gt
        lte
        lt
        eq
    """

    value = get_chunk_metadata_value(
        chunk,
        key,
    )

    if value is None:
        return False

    chunk_date = parse_date(value)

    for operator, expected_value in condition.items():
        expected_date = parse_date(
            expected_value
        )

        if operator == "gte":
            if chunk_date < expected_date:
                return False

        elif operator == "gt":
            if chunk_date <= expected_date:
                return False

        elif operator == "lte":
            if chunk_date > expected_date:
                return False

        elif operator == "lt":
            if chunk_date >= expected_date:
                return False

        elif operator == "eq":
            if chunk_date != expected_date:
                return False

        else:
            raise ValueError(
                f"unsupported date operator: {operator}"
            )

    return True


def matches_metadata(
    chunk: Chunk,
    filters: dict[str, Any],
) -> bool:
    """
    Determine whether a chunk satisfies all metadata filters.

    Simple values use exact equality.

    Dictionary values are interpreted as range conditions.
    """

    for key, expected_value in filters.items():
        actual_value = get_chunk_metadata_value(
            chunk,
            key,
        )

        if isinstance(expected_value, dict):
            if not matches_date_filter(
                chunk,
                key,
                expected_value,
            ):
                return False

        elif actual_value != expected_value:
            return False

    return True


def filter_chunks(
    chunks: list[Chunk],
    filters: dict[str, Any] | None = None,
) -> list[Chunk]:
    """
    Return chunks matching all supplied metadata filters.

    If no filters are provided, all chunks are returned.
    """

    if not filters:
        return list(chunks)

    return [
        chunk
        for chunk in chunks
        if matches_metadata(
            chunk,
            filters,
        )
    ]