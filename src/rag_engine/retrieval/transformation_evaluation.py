from dataclasses import dataclass
from typing import Callable

from src.rag_engine.retrieval.evaluation import (
    EvaluationDataset,
)
from src.rag_engine.retrieval.evaluation_runner import (
    RetrievalEvaluationResult,
    evaluate_retriever,
)
from src.rag_engine.retrieval.models import RetrievalResult


@dataclass
class TransformationEvaluation:
    name: str
    result: RetrievalEvaluationResult


class TransformedQueryRetriever:
    """
    Wrap a retriever with a query transformation.

    The transformation may return:

        str
            One transformed query.

        list[str]
            Multiple transformed queries.

    Results from multiple transformed queries are merged
    by chunk_id, keeping the highest retrieval score.
    """

    def __init__(
        self,
        retriever,
        transformer: Callable,
    ):
        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        if not callable(transformer):
            raise TypeError(
                "transformer must be callable"
            )

        self.retriever = retriever
        self.transformer = transformer

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if not query or not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        transformed = self.transformer(
            query.strip()
        )

        if isinstance(transformed, str):
            queries = [transformed]

        elif isinstance(transformed, list):
            queries = transformed

        else:
            raise TypeError(
                "transformer must return a string "
                "or list of strings"
            )

        queries = [
            q.strip()
            for q in queries
            if isinstance(q, str)
            and q.strip()
        ]

        if not queries:
            raise ValueError(
                "transformer produced no valid queries"
            )

        merged: dict[str, RetrievalResult] = {}

        for transformed_query in queries:
            results = self.retriever.search(
                transformed_query,
                top_k=top_k,
            )

            for result in results:
                existing = merged.get(
                    result.chunk_id
                )

                if (
                    existing is None
                    or result.score > existing.score
                ):
                    merged[result.chunk_id] = (
                        RetrievalResult(
                            score=result.score,
                            rank=1,
                            chunk_id=result.chunk_id,
                            text=result.text,
                            metadata=result.metadata.copy(),
                        )
                    )

        ranked = sorted(
            merged.values(),
            key=lambda result: result.score,
            reverse=True,
        )

        final_results = []

        for rank, result in enumerate(
            ranked[:top_k],
            start=1,
        ):
            final_results.append(
                RetrievalResult(
                    score=result.score,
                    rank=rank,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    metadata=result.metadata.copy(),
                )
            )

        return final_results


def evaluate_query_transformations(
    retriever,
    dataset: EvaluationDataset,
    transformations: dict[
        str,
        Callable,
    ],
    k: int = 5,
) -> list[TransformationEvaluation]:
    """
    Evaluate multiple query transformations.

    Each transformation is wrapped around the same
    base retriever and evaluated against the same
    ground-truth dataset.
    """

    if retriever is None:
        raise ValueError(
            "retriever cannot be None"
        )

    if not transformations:
        raise ValueError(
            "transformations cannot be empty"
        )

    results = []

    for name, transformer in transformations.items():
        if not name.strip():
            raise ValueError(
                "transformation names cannot be empty"
            )

        transformed_retriever = (
            TransformedQueryRetriever(
                retriever,
                transformer,
            )
        )

        evaluation = evaluate_retriever(
            transformed_retriever,
            dataset,
            k=k,
        )

        results.append(
            TransformationEvaluation(
                name=name,
                result=evaluation,
            )
        )

    return results


def print_transformation_table(
    evaluations: list[TransformationEvaluation],
) -> None:
    """
    Print transformation evaluation results.
    """

    if not evaluations:
        raise ValueError(
            "evaluations cannot be empty"
        )

    print(
        f"{'Transformation':<20}"
        f"{'Recall':>10}"
        f"{'Precision':>12}"
        f"{'MRR':>10}"
        f"{'nDCG':>10}"
    )

    print("-" * 62)

    for evaluation in evaluations:
        result = evaluation.result

        print(
            f"{evaluation.name:<20}"
            f"{result.recall_at_k:>10.4f}"
            f"{result.precision_at_k:>12.4f}"
            f"{result.mrr:>10.4f}"
            f"{result.ndcg_at_k:>10.4f}"
        )