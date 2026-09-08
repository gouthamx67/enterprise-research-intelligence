from sentence_transformers import SentenceTransformer


def main():
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    texts = [
        "The company increased investment in artificial intelligence.",
        "Management expanded spending on machine learning capabilities.",
        "Revenue increased by 18 percent during the fiscal year.",
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
    )

    print("=" * 60)
    print("EMBEDDINGS")
    print("=" * 60)

    print(f"\nNumber of texts: {len(texts)}")

    print(
        f"Embedding dimensions: "
        f"{embeddings.shape[1]}"
    )

    for index, text in enumerate(texts):
        print("\n" + "-" * 60)
        print(f"Text {index + 1}:")
        print(text)

        print("\nFirst 10 vector values:")

        print(
            embeddings[index][:10]
        )


if __name__ == "__main__":
    main()