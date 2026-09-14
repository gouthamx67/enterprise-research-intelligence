from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationQuery:
    """
    One labeled evaluation example.

    relevant_chunk_ids contains the chunk IDs that are considered
    relevant evidence for the query.
    """

    query: str
    relevant_chunk_ids: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query or not self.query.strip():
            raise ValueError("query cannot be empty")

        if not self.relevant_chunk_ids:
            raise ValueError(
                "relevant_chunk_ids cannot be empty"
            )

        self.query = " ".join(self.query.split())

        self.relevant_chunk_ids = set(
            self.relevant_chunk_ids
        )


class EvaluationDataset:
    """
    Collection of labeled retrieval queries.
    """

    def __init__(
        self,
        examples: list[EvaluationQuery] | None = None,
    ):
        self.examples = list(examples or [])

    def add(
        self,
        example: EvaluationQuery,
    ) -> None:
        if not isinstance(
            example,
            EvaluationQuery,
        ):
            raise TypeError(
                "example must be an EvaluationQuery"
            )

        self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __iter__(self):
        return iter(self.examples)

    def get_queries(self) -> list[str]:
        return [
            example.query
            for example in self.examples
        ]

    def get_relevant_ids(
        self,
        query: str,
    ) -> set[str]:
        for example in self.examples:
            if example.query == query:
                return set(
                    example.relevant_chunk_ids
                )

        raise KeyError(
            f"query not found in evaluation dataset: {query!r}"
        )


def build_demo_evaluation_dataset() -> EvaluationDataset:
    """
    Small deterministic dataset for development and testing.

    This is not a production benchmark.
    """

    examples = [
        EvaluationQuery(
            query="enterprise product strategy",
            relevant_chunk_ids={
                "chunk-002",
                "chunk-004",
            },
            metadata={
                "topic": "product_strategy",
            },
        ),
        EvaluationQuery(
            query="pricing strategy",
            relevant_chunk_ids={
                "chunk-003",
            },
            metadata={
                "topic": "pricing",
            },
        ),
        EvaluationQuery(
            query="new analytics platform",
            relevant_chunk_ids={
                "chunk-001",
            },
            metadata={
                "topic": "product",
            },
        ),
        EvaluationQuery(
            query="AI workflow features",
            relevant_chunk_ids={
                "chunk-004",
            },
            metadata={
                "topic": "ai",
            },
        ),
    ]

    return EvaluationDataset(examples)