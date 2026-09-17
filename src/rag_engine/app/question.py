import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.rag_engine.core.document import Chunk
from src.rag_engine.failures.analyzer import RAGFailureAnalyzer
from src.rag_engine.generation.config import LLMConfig
from src.rag_engine.generation.models import GenerationContext
from src.rag_engine.generation.openai_provider import OpenAIProvider
from src.rag_engine.generation.service import GenerationService
from src.rag_engine.pipeline.citation_validator import CitationValidator
from src.rag_engine.pipeline.generator import ExtractiveAnswerGenerator
from src.rag_engine.production.storage import ChunkStore, EmbeddingStore
from src.rag_engine.retrieval.dense import DenseIndex
from src.rag_engine.retrieval.embeddings import EmbeddingModel
from src.rag_engine.retrieval.models import RetrievalResult
from src.rag_engine.retrieval.sparse import BM25Retriever


class QuestionServiceError(RuntimeError):
    """Raised when a document cannot answer a question."""


@dataclass(frozen=True)
class QuestionConfig:
    candidate_k: int = 20
    final_k: int = 5
    rrf_k: int = 60

    def __post_init__(self) -> None:
        if self.candidate_k <= 0:
            raise ValueError("candidate_k must be greater than 0")

        if self.final_k <= 0:
            raise ValueError("final_k must be greater than 0")

        if self.final_k > self.candidate_k:
            raise ValueError(
                "final_k cannot be greater than candidate_k"
            )

        if self.rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")


class ContextResult:
    """
    Citation-aware retrieval context.

    Citation IDs are assigned before answer generation so an LLM
    can reference the exact evidence supplied to it.
    """

    def __init__(
        self,
        citation_id: str,
        result: RetrievalResult,
    ) -> None:
        self.citation_id = citation_id
        self.score = result.score
        self.rank = result.rank
        self.chunk_id = result.chunk_id
        self.text = result.text
        self.metadata = dict(result.metadata)


class DocumentQuestionService:
    """
    Question-answering service for one persisted document.

    Flow:

        persisted chunks
            ↓
        sparse + dense retrieval
            ↓
        reciprocal-rank fusion
            ↓
        deterministic reranking
            ↓
        context construction
            ↓
        citation assignment
            ↓
        extractive or LLM generation
            ↓
        citation validation
            ↓
        failure analysis
    """

    def __init__(
        self,
        storage,
        embedding_model: EmbeddingModel | None = None,
        config: QuestionConfig | None = None,
        generation_service: GenerationService | None = None,
        llm_config: LLMConfig | None = None,
    ) -> None:
        self.storage = storage
        self.embedding_model = embedding_model
        self.config = config or QuestionConfig()

        self.answer_generator = ExtractiveAnswerGenerator()
        self.citation_validator = CitationValidator()
        self.failure_analyzer = RAGFailureAnalyzer()

        self.generation_service = generation_service

        self.llm_config = (
            llm_config
            if llm_config is not None
            else LLMConfig.from_environment()
        )

        self._configured_generation_service = (
            generation_service is not None
        )

    @property
    def llm_enabled(self) -> bool:
        if self._configured_generation_service:
            return True

        return self.llm_config.enabled

    def _get_embedding_model(self) -> EmbeddingModel:
        if self.embedding_model is None:
            self.embedding_model = EmbeddingModel()

        return self.embedding_model

    def _get_generation_service(self) -> GenerationService:
        if self.generation_service is not None:
            return self.generation_service

        if not self.llm_config.enabled:
            raise QuestionServiceError(
                "LLM generation is disabled."
            )

        if self.llm_config.provider != "openai-compatible":
            raise QuestionServiceError(
                "Unsupported LLM provider: "
                f"{self.llm_config.provider}"
            )

        provider = OpenAIProvider(
            model=self.llm_config.model,
            api_key=self.llm_config.api_key,
            base_url=self.llm_config.base_url,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens,
        )

        self.generation_service = GenerationService(
            provider=provider,
        )

        return self.generation_service

    @staticmethod
    def _build_generation_contexts(
        context: list[ContextResult],
    ) -> list[GenerationContext]:
        generation_contexts: list[GenerationContext] = []

        for item in context:
            generation_contexts.append(
                GenerationContext(
                    citation_id=item.citation_id,
                    text=item.text,
                    source=str(
                        item.metadata.get(
                            "source",
                            "",
                        )
                    ),
                    page=item.metadata.get(
                        "start_page",
                    ),
                    metadata=dict(item.metadata),
                )
            )

        return generation_contexts

    def _generate(
        self,
        question: str,
        context: list[ContextResult],
    ):
        if not self.llm_enabled:
            return self.answer_generator.generate(
                question,
                context,
            )

        generation_service = self._get_generation_service()

        generation_contexts = (
            self._build_generation_contexts(
                context
            )
        )

        return generation_service.generate(
            question=question,
            contexts=generation_contexts,
        )

    def _load_chunks(
        self,
        document_id: str,
    ) -> list[Chunk]:
        index_directory = self.storage.index_directory(
            document_id
        )

        chunk_path = index_directory / "chunks.json"

        chunks = ChunkStore(chunk_path).load()

        if not chunks:
            raise QuestionServiceError(
                "No indexed chunks were found for this document. "
                "Process the document before asking questions."
            )

        return chunks

    def _load_dense_index(
        self,
        document_id: str,
    ) -> DenseIndex:
        index_directory = self.storage.index_directory(
            document_id
        )

        embedded_chunks = EmbeddingStore(
            index_directory / "embeddings.npy",
            index_directory / "embeddings.json",
        ).load()

        if not embedded_chunks:
            raise QuestionServiceError(
                "No persisted embeddings were found for this document."
            )

        model = self._get_embedding_model()

        index = DenseIndex(
            embedding_model=model
        )

        index.embedded_chunks = embedded_chunks

        index.matrix = np.vstack(
            [
                item.embedding
                for item in embedded_chunks
            ]
        )

        return index

    @staticmethod
    def _reciprocal_rank_fusion(
        result_lists: list[list[RetrievalResult]],
        rrf_k: int,
    ) -> list[RetrievalResult]:
        scores: dict[str, float] = {}
        representatives: dict[
            str,
            RetrievalResult,
        ] = {}

        for results in result_lists:
            for rank, result in enumerate(
                results,
                start=1,
            ):
                scores[result.chunk_id] = (
                    scores.get(
                        result.chunk_id,
                        0.0,
                    )
                    + 1.0 / (rrf_k + rank)
                )

                if result.chunk_id not in representatives:
                    representatives[result.chunk_id] = result

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: scores[chunk_id],
            reverse=True,
        )

        fused: list[RetrievalResult] = []

        for rank, chunk_id in enumerate(
            ranked_ids,
            start=1,
        ):
            original = representatives[chunk_id]

            metadata = dict(original.metadata)
            metadata["rrf_score"] = scores[chunk_id]

            fused.append(
                RetrievalResult(
                    score=scores[chunk_id],
                    rank=rank,
                    chunk_id=original.chunk_id,
                    text=original.text,
                    metadata=metadata,
                )
            )

        return fused

    def _hybrid_retrieve(
        self,
        question: str,
        chunks: list[Chunk],
        dense_index: DenseIndex,
    ) -> list[RetrievalResult]:
        sparse_retriever = BM25Retriever(chunks)

        sparse_results = sparse_retriever.search(
            question,
            top_k=self.config.candidate_k,
        )

        dense_results = dense_index.search(
            question,
            top_k=self.config.candidate_k,
        )

        return self._reciprocal_rank_fusion(
            [
                sparse_results,
                dense_results,
            ],
            rrf_k=self.config.rrf_k,
        )

    @staticmethod
    def _rerank(
        question: str,
        results: list[RetrievalResult],
        final_k: int,
    ) -> list[RetrievalResult]:
        if not results:
            return []

        question_terms = {
            token.lower().strip(
                ".,!?;:()[]{}\"'"
            )
            for token in question.split()
            if len(
                token.lower().strip(
                    ".,!?;:()[]{}\"'"
                )
            ) > 2
        }

        scored: list[
            tuple[float, RetrievalResult]
        ] = []

        for result in results:
            text_terms = {
                token.lower().strip(
                    ".,!?;:()[]{}\"'"
                )
                for token in result.text.split()
                if len(
                    token.lower().strip(
                        ".,!?;:()[]{}\"'"
                    )
                ) > 2
            }

            overlap = len(
                question_terms.intersection(
                    text_terms
                )
            )

            lexical_bonus = (
                overlap
                / max(
                    len(question_terms),
                    1,
                )
            )

            combined_score = (
                0.8 * result.score
                + 0.2 * lexical_bonus
            )

            scored.append(
                (
                    combined_score,
                    result,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        reranked: list[RetrievalResult] = []

        for rank, (
            score,
            result,
        ) in enumerate(
            scored[:final_k],
            start=1,
        ):
            metadata = dict(result.metadata)

            metadata["rerank_score"] = score
            metadata["original_rrf_score"] = result.score

            reranked.append(
                RetrievalResult(
                    score=score,
                    rank=rank,
                    chunk_id=result.chunk_id,
                    text=result.text,
                    metadata=metadata,
                )
            )

        return reranked

    @staticmethod
    def _build_context(
        results: list[RetrievalResult],
    ) -> list[ContextResult]:
        seen: set[str] = set()
        context: list[ContextResult] = []

        for result in results:
            normalized = " ".join(
                result.text.lower().split()
            )

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)

            citation_id = (
                f"C{len(context) + 1}"
            )

            context.append(
                ContextResult(
                    citation_id=citation_id,
                    result=result,
                )
            )

        return context

    @staticmethod
    def _build_citations(
        context: list[ContextResult],
    ) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []

        for item in context:
            citations.append(
                {
                    "citation_id": item.citation_id,
                    "chunk_id": item.chunk_id,
                    "text": item.text,
                    "source": item.metadata.get(
                        "source",
                        "",
                    ),
                    "page": item.metadata.get(
                        "start_page"
                    ),
                    "metadata": dict(
                        item.metadata
                    ),
                }
            )

        return citations

    @staticmethod
    def _to_retrieval_results(
        context: list[ContextResult],
    ) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                score=item.score,
                rank=item.rank,
                chunk_id=item.chunk_id,
                text=item.text,
                metadata=dict(item.metadata),
            )
            for item in context
        ]

    def ask(
        self,
        document_id: str,
        question: str,
    ) -> dict[str, Any]:
        if not question or not question.strip():
            raise QuestionServiceError(
                "Question cannot be empty."
            )

        record = self.storage.load_record(
            document_id
        )

        if record.status != "ready":
            raise QuestionServiceError(
                "Document is not ready for questions. "
                "Process the document first."
            )

        chunks = self._load_chunks(
            document_id
        )

        dense_index = self._load_dense_index(
            document_id
        )

        hybrid_results = self._hybrid_retrieve(
            question=question,
            chunks=chunks,
            dense_index=dense_index,
        )

        if not hybrid_results:
            raise QuestionServiceError(
                "No relevant evidence was found for this question."
            )

        reranked_results = self._rerank(
            question=question,
            results=hybrid_results,
            final_k=self.config.final_k,
        )

        context_results = self._build_context(
            reranked_results
        )

        if not context_results:
            raise QuestionServiceError(
                "No relevant evidence was found for this question."
            )

        citation_data = self._build_citations(
            context_results
        )

        valid_citation_ids = {
            item["citation_id"]
            for item in citation_data
        }

        generated = self._generate(
            question=question,
            context=context_results,
        )

        citation_ids = list(
            generated.citation_ids
        )

        citation_validation = (
            self.citation_validator.validate(
                citation_ids,
                valid_citation_ids,
            )
        )

        retrieval_results = (
            self._to_retrieval_results(
                context_results
            )
        )

        failure_analysis = (
            self.failure_analyzer.analyze(
                query=question,
                results=retrieval_results,
                answer=generated.answer,
                citations=citation_ids,
                valid_citation_ids=valid_citation_ids,
            )
        )

        failure_count = (
            failure_analysis.failure_count
        )

        confidence = min(
            1.0,
            0.5
            + min(
                len(context_results),
                5,
            )
            * 0.08,
        )

        if not citation_validation.valid:
            confidence *= 0.5

        if failure_count:
            confidence *= 0.9

        generation_mode = (
            "llm"
            if self.llm_enabled
            else "extractive"
        )

        generation_provider = getattr(
            generated,
            "provider",
            "extractive",
        )

        generation_model = getattr(
            generated,
            "model",
            "extractive",
        )

        return {
            "document_id": document_id,
            "question": question,
            "answer": generated.answer,
            "citations": citation_data,
            "confidence": round(
                confidence,
                3,
            ),
            "success": (
                bool(generated.answer.strip())
                and bool(context_results)
                and citation_validation.valid
            ),
            "failure_count": failure_count,
            "retrieved_count": len(
                hybrid_results
            ),
            "context_count": len(
                context_results
            ),
            "metadata": {
                "retrieval_strategy": "hybrid_rrf",
                "rrf_k": self.config.rrf_k,
                "candidate_k": self.config.candidate_k,
                "final_k": self.config.final_k,
                "citation_valid": (
                    citation_validation.valid
                ),
                "document_filename": record.filename,
                "source_type": record.source_type,
                "generation_mode": generation_mode,
                "generation_provider": generation_provider,
                "generation_model": generation_model,
            },
        }
