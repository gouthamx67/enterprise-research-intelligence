from src.rag_engine.chunking.sliding_window import (
    sliding_window_chunks,
)


TEXT = (
    "The company launched a new enterprise platform in 2025. "
    "The platform combines analytics, automation, and collaboration. "
    "In 2026, the company changed its product strategy. "
    "It increased investment in AI capabilities and reduced "
    "investment in several legacy products."
)


def main():
    chunks = sliding_window_chunks(
        TEXT,
        chunk_size=120,
        overlap=30,
    )

    print("SLIDING-WINDOW CHUNKS")
    print("=====================")

    for index, chunk in enumerate(chunks, start=1):
        print()
        print(f"--- CHUNK {index} ---")
        print(chunk)


if __name__ == "__main__":
    main()