import re
from typing import Any

from rank_bm25 import BM25Okapi

from src.rag_engine.core.document import Chunk
from src.rag_engine.retrieval.filters import (
    filter_chunks,
)
from src.rag_engine.retrieval.models import (
    RetrievalResult,
)


def tokenize(text: str) -> list[str]:
    """
    Convert text into normalized lexical tokens.
    """

    text = text.lower()

    return re.findall(
        r"\b\w+\b",
        text,
    )


class BM25Retriever:
    """
    Sparse lexical retriever based on BM25.

    Metadata filtering happens before BM25 ranking.
    """

    def __init__(
        self,
        chunks: list[Chunk],
    ):
        if not chunks:
            raise ValueError(
                "chunks cannot be empty"
            )

        self.chunks = chunks

        self.tokenized_chunks = [
            tokenize(chunk.text)
            for chunk in chunks
        ]

        self.index = BM25Okapi(
            self.tokenized_chunks
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the top-k chunks.

        If filters are provided, only matching chunks
        participate in BM25 ranking.
        """

        if not query.strip():
            raise ValueError(
                "query cannot be empty"
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0"
            )

        candidate_chunks = filter_chunks(
            self.chunks,
            filters,
        )

        if not candidate_chunks:
            return []

        candidate_lookup = {
            chunk.chunk_id: chunk
            for chunk in candidate_chunks
        }

        candidate_indices = [
            index
            for index, chunk in enumerate(
                self.chunks
            )
            if chunk.chunk_id in candidate_lookup
        ]

        query_tokens = tokenize(query)

        scores = self.index.get_scores(
            query_tokens
        )

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for rank, index in enumerate(
            ranked_indices[:top_k],
            start=1,
        ):
            chunk = self.chunks[index]

            results.append(
                RetrievalResult(
                    score=float(scores[index]),
                    rank=rank,
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata={
                        "document_id": chunk.document_id,
                        "source": chunk.source,
                        "source_type": chunk.source_type,
                        "section_title": chunk.section_title,
                        "start_page": chunk.start_page,
                        "end_page": chunk.end_page,
                        "block_numbers": chunk.block_numbers,
                        **chunk.metadata,
                    },
                )
            )

        return results