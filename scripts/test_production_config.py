import os

from src.rag_engine.production.config import (
    ProductionConfig,
    load_production_config,
)


def main():
    config = ProductionConfig(
        candidate_k=20,
        final_k=5,
    )

    assert config.candidate_k == 20
    assert config.final_k == 5

    os.environ["RAG_FINAL_K"] = "7"

    loaded = load_production_config()

    assert loaded.final_k == 7

    del os.environ["RAG_FINAL_K"]

    print("Production config test passed.")


if __name__ == "__main__":
    main()