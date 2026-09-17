from dataclasses import dataclass
from typing import Any

from src.rag_engine.advanced.adaptive import AdaptiveDecision, AdaptiveRouter
from src.rag_engine.advanced.agentic import AgentPlan, AgenticPlanner
from src.rag_engine.core.document import Chunk
from src.rag_engine.failures.analyzer import RAGFailureAnalyzer
from src.rag_engine.failures.models import (
    FailureAnalysisResult,
    FailureFinding,
    FailureType,
)
from src.rag_engine.pipeline.citation_validator import (
    CitationValidationResult,
    CitationValidator,
)
from src.rag_engine.pipeline.generator import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
)
from src.rag_engine.pipeline.models import GeneratedAnswer, RAGResponse
from src.rag_engine.retrieval.context_construction import (
    ContextConstructionPipeline,
)
from src.rag_engine.retrieval.models import RetrievalResult
from src.rag_engine.retrieval.reranker import RerankedResult


@dataclass
class PipelineComponents:
    retriever: Any
    reranker: Any
    context_pipeline: ContextConstructionPipeline
    generator: AnswerGenerator
    planner: AgenticPlanner
    router: AdaptiveRouter
    failure_analyzer: RAGFailureAnalyzer
    citation_validator: CitationValidator


class EndToEndRAGPipeline:
    """
    Complete end-to-end RAG pipeline.

    Flow:

        Query
          ↓
        Agentic planning
          ↓
        Adaptive routing
          ↓
        Multiple retrieval steps
          ↓
        Result merging
          ↓
        Failure analysis
          ↓
        Reranking
          ↓
        Context construction
          ↓
        Answer generation
          ↓
        Citation validation
          ↓
        Final failure analysis
          ↓
        Confidence calculation
          ↓
        RAG response
    """

    def __init__(
        self,
        retriever,
        reranker,
        context_pipeline: ContextConstructionPipeline,
        generator: AnswerGenerator | None = None,
        planner: AgenticPlanner | None = None,
        router: AdaptiveRouter | None = None,
        failure_analyzer: RAGFailureAnalyzer | None = None,
        citation_validator: CitationValidator | None = None,
    ):
        if retriever is None:
            raise ValueError("retriever cannot be None")

        if reranker is None:
            raise ValueError("reranker cannot be None")

        if context_pipeline is None:
            raise ValueError("context_pipeline cannot be None")

        self.retriever = retriever
        self.reranker = reranker
        self.context_pipeline = context_pipeline

        self.generator = generator or ExtractiveAnswerGenerator()
        self.planner = planner or AgenticPlanner()
        self.router = router or AdaptiveRouter()
        self.failure_analyzer = (
            failure_analyzer or RAGFailureAnalyzer()
        )
        self.citation_validator = (
            citation_validator or CitationValidator()
        )

    def _retrieve(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Run one retrieval operation."""
        results = self.retriever.search(
            query,
            top_k=top_k,
        )

        return list(results)

    def _merge_results(
        self,
        results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """
        Merge results from multiple retrieval strategies.

        If the same chunk appears multiple times, retain the
        highest-scoring occurrence.
        """
        merged: dict[str, RetrievalResult] = {}

        for result in results:
            existing = merged.get(result.chunk_id)

            if existing is None or result.score > existing.score:
                merged[result.chunk_id] = result

        merged_results = list(merged.values())

        merged_results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        for rank, result in enumerate(
            merged_results,
            start=1,
        ):
            result.rank = rank

        return merged_results

    def _to_chunks(
        self,
        results: list[RerankedResult],
    ) -> list[Chunk]:
        """
        Convert reranked retrieval results into Chunk objects.

        This method is retained as part of the pipeline's internal
        conversion layer.
        """
        chunks: list[Chunk] = []

        for result in results:
            metadata = dict(result.metadata)

            chunks.append(
                Chunk(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    document_id=metadata.get(
                        "document_id",
                        "unknown-document",
                    ),
                    source=metadata.get(
                        "source",
                        "unknown-source",
                    ),
                    source_type=metadata.get(
                        "source_type",
                        "unknown",
                    ),
                    section_title=metadata.get(
                        "section_title"
                    ),
                    start_page=metadata.get(
                        "start_page"
                    ),
                    end_page=metadata.get(
                        "end_page"
                    ),
                    block_numbers=list(
                        metadata.get(
                            "block_numbers",
                            [],
                        )
                    ),
                    metadata=metadata,
                )
            )

        return chunks

    def _reranked_to_retrieval_results(
        self,
        results: list[RerankedResult],
    ) -> list[RetrievalResult]:
        """Convert reranked results back to RetrievalResult."""
        return [
            RetrievalResult(
                score=result.score,
                rank=result.rank,
                chunk_id=result.chunk_id,
                text=result.text,
                metadata=dict(result.metadata),
            )
            for result in results
        ]

    def _calculate_confidence(
        self,
        reranked_results: list[RerankedResult],
        context_count: int,
        failure_count: int,
    ) -> float:
        """
        Calculate a transparent heuristic confidence score.

        This is not a calibrated probability.
        """
        if not reranked_results:
            return 0.0

        average_score = sum(
            result.score
            for result in reranked_results
        ) / len(reranked_results)

        evidence_component = min(
            context_count / 3.0,
            1.0,
        )

        failure_penalty = min(
            failure_count * 0.15,
            0.75,
        )

        confidence = (
            average_score * 0.6
            + evidence_component * 0.4
            - failure_penalty
        )

        return max(
            0.0,
            min(1.0, confidence),
        )

    def _add_citation_failure(
        self,
        failure_analysis: FailureAnalysisResult,
        validation: CitationValidationResult,
    ) -> FailureAnalysisResult:
        """Add a citation failure when citations are invalid."""
        if validation.valid:
            return failure_analysis

        failure_analysis.findings.append(
            FailureFinding(
                failure_type=FailureType.CITATION_ERROR,
                severity="high",
                message=(
                    "The generated answer contains "
                    "invalid citation IDs."
                ),
                evidence=list(
                    validation.invalid_citations
                ),
            )
        )

        return failure_analysis

    def run(
        self,
        query: str,
        candidate_k: int = 20,
        final_k: int = 5,
    ) -> RAGResponse:
        """Execute the complete RAG pipeline."""
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if candidate_k <= 0:
            raise ValueError(
                "candidate_k must be greater than 0"
            )

        if final_k <= 0:
            raise ValueError(
                "final_k must be greater than 0"
            )

        if final_k > candidate_k:
            raise ValueError(
                "final_k cannot be greater than candidate_k"
            )

        query = query.strip()

        # --------------------------------------------------------
        # 1. Agentic planning
        # --------------------------------------------------------
        plan: AgentPlan = self.planner.plan(query)

        # --------------------------------------------------------
        # 2. Adaptive routing
        # --------------------------------------------------------
        routing: AdaptiveDecision = self.router.route(query)

        # --------------------------------------------------------
        # 3. Multiple retrieval steps
        # --------------------------------------------------------
        all_results: list[RetrievalResult] = []

        for step in plan.steps:
            step_results = self._retrieve(
                step.query,
                top_k=candidate_k,
            )

            all_results.extend(step_results)

        # --------------------------------------------------------
        # 4. Merge retrieval results
        # --------------------------------------------------------
        merged_results = self._merge_results(
            all_results
        )

        # --------------------------------------------------------
        # 5. Initial failure analysis
        #
        # IMPORTANT:
        # RAGFailureAnalyzer.analyze() expects `results=`,
        # not `retrieval_results=`.
        # --------------------------------------------------------
        initial_failures = self.failure_analyzer.analyze(
            query=query,
            results=merged_results,
        )

        # --------------------------------------------------------
        # 6. Reranking
        # --------------------------------------------------------
        reranked_results = self.reranker.rerank(
            query=query,
            candidates=merged_results,
            top_k=final_k,
        )

        # --------------------------------------------------------
        # 7. Context construction
        # --------------------------------------------------------
        context_candidates = (
            self._reranked_to_retrieval_results(
                reranked_results
            )
        )

        context_result = self.context_pipeline.build(
            query=query,
            candidates=context_candidates,
        )

        # --------------------------------------------------------
        # 8. No-context guard
        # --------------------------------------------------------
        if context_result.final_count == 0:
            return RAGResponse(
                query=query,
                plan=plan,
                routing=routing,
                retrieved_results=reranked_results,
                generated_answer=None,
                failure_analysis=initial_failures,
                confidence=0.0,
                metadata={
                    "candidate_count": len(
                        merged_results
                    ),
                    "retrieved_count": len(
                        all_results
                    ),
                    "reranked_count": len(
                        reranked_results
                    ),
                    "context_count": 0,
                    "routing_strategy": routing.strategy,
                    "routing_reason": routing.reason,
                    "initial_failure_count": (
                        initial_failures.failure_count
                    ),
                },
            )

        # --------------------------------------------------------
        # 9. Answer generation
        # --------------------------------------------------------
        generated_answer: GeneratedAnswer = (
            self.generator.generate(
                query=query,
                contexts=context_result.contexts,
            )
        )

        # --------------------------------------------------------
        # 10. Citation validation
        # --------------------------------------------------------
        valid_citation_ids = {
            context.citation_id
            for context in context_result.contexts
        }

        citation_validation = (
            self.citation_validator.validate(
                generated_citations=(
                    generated_answer.citation_ids
                ),
                valid_citation_ids=valid_citation_ids,
            )
        )

        # --------------------------------------------------------
        # 11. Final failure analysis
        # --------------------------------------------------------
        final_failures = self.failure_analyzer.analyze(
            query=query,
            results=self._reranked_to_retrieval_results(
                reranked_results
            ),
            answer=generated_answer.answer,
            citations=generated_answer.citation_ids,
            valid_citation_ids=valid_citation_ids,
        )

        final_failures = self._add_citation_failure(
            final_failures,
            citation_validation,
        )

        # --------------------------------------------------------
        # 12. Confidence
        # --------------------------------------------------------
        confidence = self._calculate_confidence(
            reranked_results=reranked_results,
            context_count=context_result.final_count,
            failure_count=final_failures.failure_count,
        )

        # --------------------------------------------------------
        # 13. Final response
        # --------------------------------------------------------
        return RAGResponse(
            query=query,
            plan=plan,
            routing=routing,
            retrieved_results=reranked_results,
            generated_answer=generated_answer,
            failure_analysis=final_failures,
            confidence=confidence,
            metadata={
                "candidate_count": len(
                    merged_results
                ),
                "retrieved_count": len(
                    all_results
                ),
                "reranked_count": len(
                    reranked_results
                ),
                "context_count": context_result.final_count,
                "context_characters": (
                    context_result.total_characters
                ),
                "routing_strategy": routing.strategy,
                "routing_reason": routing.reason,
                "citation_valid": citation_validation.valid,
                "initial_failure_count": (
                    initial_failures.failure_count
                ),
                "final_failure_count": (
                    final_failures.failure_count
                ),
            },
        )


def build_end_to_end_pipeline(
    retriever,
    reranker,
    context_pipeline: ContextConstructionPipeline,
    generator: AnswerGenerator | None = None,
    planner: AgenticPlanner | None = None,
    router: AdaptiveRouter | None = None,
    failure_analyzer: RAGFailureAnalyzer | None = None,
    citation_validator: CitationValidator | None = None,
) -> EndToEndRAGPipeline:
    """Convenience factory for the complete pipeline."""
    return EndToEndRAGPipeline(
        retriever=retriever,
        reranker=reranker,
        context_pipeline=context_pipeline,
        generator=generator,
        planner=planner,
        router=router,
        failure_analyzer=failure_analyzer,
        citation_validator=citation_validator,
    )