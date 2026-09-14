from dataclasses import dataclass


@dataclass(frozen=True)
class RerankingConfig:
    """
    Configuration for the retrieval + reranking stage.
    """

    candidate_k: int = 20
    final_k: int = 5
    model_name: str = (
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    score_threshold: float | None = None

    def __post_init__(self):
        if self.candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        if self.final_k <= 0:
            raise ValueError(
                "final_k must be greater than 0"
            )

        if self.final_k > self.candidate_k:
            raise ValueError(
                "final_k cannot be greater than candidate_k"
            )

        if not self.model_name.strip():
            raise ValueError(
                "model_name cannot be empty"
            )

        if self.score_threshold is not None:
            if not -1.0 <= self.score_threshold <= 1.0:
                raise ValueError(
                    "score_threshold must be between "
                    "-1 and 1"
                )


def build_default_reranking_config() -> RerankingConfig:
    return RerankingConfig()


def build_fast_reranking_config() -> RerankingConfig:
    """
    Smaller candidate pool for lower latency.
    """

    return RerankingConfig(
        candidate_k=10,
        final_k=5,
    )


def build_high_recall_reranking_config() -> RerankingConfig:
    """
    Larger candidate pool to give the reranker
    more potentially relevant evidence.
    """

    return RerankingConfig(
        candidate_k=50,
        final_k=5,
    )
