from src.rag_engine.production.cache import (
    QueryCache,
)


def main():
    cache = QueryCache(
        ttl_seconds=60,
        max_entries=2,
    )

    assert cache.get("missing") is None

    cache.set(
        "a",
        "value-a",
    )

    assert cache.get("a") == "value-a"

    cache.set(
        "b",
        "value-b",
    )

    cache.set(
        "c",
        "value-c",
    )

    assert cache.size == 2
    assert cache.get("c") == "value-c"

    cache.delete("c")

    assert cache.get("c") is None

    cache.clear()

    assert cache.size == 0

    print("Production cache test passed.")


if __name__ == "__main__":
    main()