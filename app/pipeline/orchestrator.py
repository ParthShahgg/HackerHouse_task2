"""The pipeline orchestrator: a state machine over typed stages.

Live path (nothing here touches the corpus or does any chunking - that is all
offline, which is what makes the latency budget achievable)::

    audio -> STT -> input guard -> query embedding -> hybrid retrieval
          -> rerank -> confidence gate -> generation -> output validation

Cross-cutting rules applied at every external boundary:

* explicit timeout
* bounded retries with exponential backoff + jitter (deterministic failures are
  not retried at all)
* structured error, never a bare exception to the caller
* a fallback where one is technically meaningful, and it is always *recorded*

CPU-bound model work (embedding, reranking, NLI) is dispatched to a thread via
``asyncio.to_thread`` so a single slow inference cannot stall the event loop and
delay every other in-flight request.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any

from app.config import get_settings, get_thresholds
from app.guardrails.grounding import OutputGuardrail
from app.guardrails.input import InputGuardrail, blocked_message
from app.observability.metrics import METRICS
from app.observability.tracing import Trace, get_logger
from app.retrieval.confidence import ConfidenceGate, abstain_message
from app.retrieval.hybrid import decide_languages
from app.schemas.common import (
    AbstainReason,
    GateDecision,
    GroundingStatus,
    LatencyBreakdown,
    PipelineStage,
    RetrievalMode,
    Stopwatch,
    ValidationAction,
    now_ns,
    ns_to_ms,
)
from app.schemas.generation import GenerationResult, ValidationResult
from app.schemas.query import GuardrailResult, STTResult, TranscriptSegment
from app.schemas.response import CandidateDebug, Citation, DebugInfo, FinalResponse
from app.schemas.retrieval import GroundingDecision, ParentContext, RerankResult, RetrievalResult
from app.pipeline.states import PipelineStage as _Stage  # noqa: F401  (re-export symmetry)
from app.pipeline.states import validate_transition

logger = get_logger(__name__)

__all__ = ["RAGOrchestrator", "get_orchestrator", "reset_orchestrator"]


class _Ctx:
    """Mutable per-request state carried between stages."""

    def __init__(self, trace: Trace, *, voice: bool) -> None:
        self.trace = trace
        self.voice = voice
        self.latency = LatencyBreakdown()
        self.stage = PipelineStage.START
        self.t0 = now_ns()
        self.first_token_ns: int | None = None

        self.stt: STTResult | None = None
        self.guard: GuardrailResult | None = None
        self.retrieval: RetrievalResult | None = None
        self.rerank: RerankResult | None = None
        self.gate: GroundingDecision | None = None
        self.generation: GenerationResult | None = None
        self.validation: ValidationResult | None = None
        self.contexts: list[ParentContext] = []
        self.language: str | None = None
        self.retrieval_mode: RetrievalMode | None = None
        self.languages_searched: list[str] = []

    def goto(self, stage: PipelineStage) -> None:
        validate_transition(self.stage, stage)
        self.stage = stage
        self.trace.enter(stage)


class RAGOrchestrator:
    """Executes the pipeline. One instance per process; stateless per request."""

    def __init__(self, *, retrieval_service=None, generator=None, stt=None, settings=None) -> None:
        self.settings = settings or get_settings()
        self._service = retrieval_service
        self._generator = generator
        self._stt = stt
        self.input_guardrail = InputGuardrail()
        self.output_guardrail = OutputGuardrail()
        self.gate = ConfidenceGate()

    # ------------------------------------------------------------ lazy services
    @property
    def service(self):
        if self._service is None:
            from app.retrieval.service import RetrievalService

            self._service = RetrievalService()
        return self._service

    @property
    def generator(self):
        if self._generator is None:
            from app.generation.mock import build_generator

            self._generator = build_generator()
        return self._generator

    @property
    def stt(self):
        if self._stt is None:
            from app.stt.sarvam import get_stt

            self._stt = get_stt()
        return self._stt

    def warmup(self) -> None:
        self.service.warmup()

    # ------------------------------------------------------------------ entry: text
    async def run_text(
        self,
        query: str,
        *,
        language: str | None = None,
        top_k: int | None = None,
        include_debug: bool = False,
        trace: Trace | None = None,
    ) -> FinalResponse:
        trace = trace or Trace()
        ctx = _Ctx(trace, voice=False)
        with trace:
            # A caller-supplied language hint is an explicit assertion, so it is
            # trusted at full confidence. Absent a hint, retrieval stays
            # cross-lingual rather than guessing from script.
            ctx.stt = STTResult(
                transcript=query,
                detected_language=language,
                raw_language_tag=language,
                language_confidence=1.0 if language else None,
                transport="injected",
            )
            return await self._run_pipeline(ctx, top_k=top_k, include_debug=include_debug)

    # ----------------------------------------------------------------- entry: voice
    async def run_voice(
        self,
        audio: bytes | None = None,
        *,
        audio_stream: AsyncIterator[bytes] | None = None,
        language: str | None = None,
        sarvam_language: str | None = None,
        is_wav: bool = True,
        transcript_override: str | None = None,
        top_k: int | None = None,
        include_debug: bool = False,
        on_partial: Callable[[TranscriptSegment], Any] | None = None,
        trace: Trace | None = None,
    ) -> FinalResponse:
        trace = trace or Trace()
        ctx = _Ctx(trace, voice=True)
        with trace:
            ctx.goto(PipelineStage.STT)

            if transcript_override is not None:
                # Test/debug path: exercise the RAG stages without Sarvam.
                ctx.stt = STTResult(
                    transcript=transcript_override,
                    detected_language=language,
                    language_confidence=1.0 if language else None,
                    transport="injected",
                )
                ctx.latency.stt_latency = None
            else:
                # sarvam_language is the BCP-47 pin (e.g. "hi-IN") sent from the
                # frontend when the user selects a language explicitly. Passing it
                # to Sarvam skips the auto-detection model (~200-400ms saving).
                # Falls back to language (ISO-639-1) which the STT client maps to
                # the correct Sarvam tag, or None for full auto-detect.
                stt_lang = sarvam_language or language
                try:
                    with Stopwatch(ctx.latency, "stt_latency"):
                        if audio_stream is not None:
                            ctx.stt = await self.stt.transcribe_stream(
                                audio_stream,
                                language_code=stt_lang,
                                is_wav=is_wav,
                                on_partial=on_partial,
                            )
                        else:
                            ctx.stt = await self.stt.transcribe_bytes(
                                audio or b"",
                                language_code=stt_lang,
                                is_wav=is_wav,
                                on_partial=on_partial,
                            )
                    trace.record(
                        PipelineStage.STT,
                        duration_ms=ctx.latency.stt_latency,
                        transport=ctx.stt.transport,
                        used_fallback=ctx.stt.used_fallback,
                        detected_language=ctx.stt.detected_language,
                        confidence=ctx.stt.language_confidence,
                        chars=len(ctx.stt.transcript),
                    )
                    if ctx.stt.used_fallback:
                        trace.warn("STT used REST fallback (streaming socket failed)")
                except Exception as exc:  # noqa: BLE001
                    logger.error("STT failed: %s", exc)
                    trace.record(PipelineStage.STT, ok=False, error=str(exc))
                    return self._fail(
                        ctx,
                        AbstainReason.INPUT_BLOCKED,
                        f"Speech recognition failed: {exc}",
                        include_debug=include_debug,
                    )

            if ctx.stt is None or ctx.stt.is_empty:
                return self._abstain(
                    ctx,
                    AbstainReason.INPUT_BLOCKED,
                    "I didn't catch any speech. Could you try again?",
                    include_debug=include_debug,
                )

            return await self._run_pipeline(ctx, top_k=top_k, include_debug=include_debug)

    # -------------------------------------------------------------- core pipeline
    async def _run_pipeline(
        self, ctx: _Ctx, *, top_k: int | None, include_debug: bool
    ) -> FinalResponse:
        settings = self.settings
        stt = ctx.stt
        assert stt is not None

        # ---------------------------------------------------------- INPUT_GUARD
        ctx.goto(PipelineStage.INPUT_GUARD)
        with Stopwatch(ctx.latency, "guardrail_latency"):
            ctx.guard = self.input_guardrail.apply(stt.transcript)
        ctx.trace.record(
            PipelineStage.INPUT_GUARD,
            duration_ms=ctx.latency.guardrail_latency,
            allowed=ctx.guard.allowed,
            category=ctx.guard.category.value,
            artifacts_removed=ctx.guard.artifacts_removed,
        )
        if not ctx.guard.allowed:
            return self._abstain(
                ctx,
                AbstainReason.INPUT_BLOCKED,
                blocked_message(ctx.guard),
                include_debug=include_debug,
            )

        query = ctx.guard.normalized_query
        ctx.language = stt.detected_language

        # ---------------------------------------------------------------- EMBED
        ctx.goto(PipelineStage.EMBED)
        try:
            embedding = await asyncio.to_thread(self.service.embed_query, query, ctx.latency)
        except Exception as exc:  # noqa: BLE001
            logger.error("query embedding failed: %s", exc)
            ctx.trace.record(PipelineStage.EMBED, ok=False, error=str(exc))
            return self._fail(
                ctx, AbstainReason.INTERNAL_ERROR,
                f"Query embedding failed: {exc}", include_debug=include_debug,
            )
        ctx.trace.record(
            PipelineStage.EMBED,
            duration_ms=ctx.latency.query_embedding_latency,
            dim=embedding.dim,
            sparse_terms=len(embedding.sparse_indices),
        )

        # ------------------------------------------------------------- RETRIEVE
        ctx.goto(PipelineStage.RETRIEVE)
        languages, mode = decide_languages(
            detected_language=stt.detected_language,
            confidence=stt.language_confidence,
            is_code_mixed=stt.is_code_mixed,
            configured=settings.language_list,
            min_confidence=settings.language_filter_confidence,
        )
        ctx.retrieval_mode = mode
        ctx.languages_searched = list(languages)

        ctx.retrieval = await self._retrieve_with_retry(ctx, embedding, languages, mode)
        ctx.trace.record(
            PipelineStage.RETRIEVE,
            duration_ms=(ctx.latency.dense_latency or 0) + (ctx.latency.sparse_latency or 0),
            mode=mode.value,
            languages=languages,
            dense=ctx.retrieval.dense_count,
            sparse=ctx.retrieval.sparse_count,
            fused=ctx.retrieval.fused_count,
            degraded=ctx.retrieval.degraded,
        )

        if ctx.retrieval.degraded and not ctx.retrieval.candidates:
            return self._abstain(
                ctx,
                AbstainReason.RETRIEVAL_ERROR,
                abstain_message(ctx.language),
                include_debug=include_debug,
            )

        # ---------------------------------------------------------------- RERANK
        ctx.goto(PipelineStage.RERANK)
        try:
            ctx.rerank = await asyncio.to_thread(
                self.service.rerank,
                query,
                ctx.retrieval,
                latency=ctx.latency,
                final_top_k=top_k or settings.final_top_k,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("rerank stage failed: %s", exc)
            ctx.trace.record(PipelineStage.RERANK, ok=False, error=str(exc))
            return self._abstain(
                ctx, AbstainReason.INTERNAL_ERROR,
                abstain_message(ctx.language), include_debug=include_debug,
            )

        if ctx.rerank.fallback_used:
            ctx.trace.warn(
                "reranker unavailable; ranking fell back to RRF order "
                "(confidence gate cannot be calibrated in this mode)"
            )
        ctx.contexts = list(ctx.rerank.contexts)
        ctx.trace.record(
            PipelineStage.RERANK,
            duration_ms=ctx.latency.rerank_latency,
            considered=ctx.rerank.considered,
            contexts=len(ctx.contexts),
            fallback=ctx.rerank.fallback_used,
            top_score=ctx.rerank.top_score,
        )

        # ------------------------------------------------------- CONFIDENCE_GATE
        ctx.goto(PipelineStage.CONFIDENCE_GATE)
        with Stopwatch(ctx.latency, "grounding_gate_latency"):
            ctx.gate = self.gate.evaluate(ctx.rerank)
        ctx.trace.record(
            PipelineStage.CONFIDENCE_GATE,
            duration_ms=ctx.latency.grounding_gate_latency,
            decision=ctx.gate.decision.value,
            reason=ctx.gate.reason.value,
            top_score=ctx.gate.top_score,
            margin=ctx.gate.margin,
            calibrated=ctx.gate.thresholds_calibrated,
        )
        if not ctx.gate.thresholds_calibrated:
            ctx.trace.warn(
                "abstention thresholds are uncalibrated; run "
                "scripts/calibrate_thresholds.py"
            )

        if ctx.gate.decision == GateDecision.ABSTAIN:
            # The LLM is deliberately never called here.
            return self._abstain(
                ctx, ctx.gate.reason, abstain_message(ctx.language), include_debug=include_debug
            )

        # -------------------------------------------------- GENERATE / VALIDATE
        return await self._generate_and_validate(ctx, query, include_debug=include_debug)

    async def _retrieve_with_retry(
        self, ctx: _Ctx, embedding, languages: Sequence[str], mode: RetrievalMode
    ) -> RetrievalResult:
        """Retrieve, retrying once on a transient Qdrant failure."""
        for attempt in (1, 2):
            result = await asyncio.to_thread(
                self.service.retrieve,
                embedding,
                languages=languages,
                mode=mode,
                latency=ctx.latency,
            )
            if not result.degraded:
                return result
            if attempt == 1:
                logger.warning(
                    "Qdrant retrieval degraded (%s); retrying once", result.degraded_reason
                )
                ctx.trace.warn(f"retrieval retry after: {result.degraded_reason}")
                await asyncio.sleep(0.1)
                continue
            logger.error("Qdrant retrieval failed twice: %s", result.degraded_reason)
        return result

    async def _generate_and_validate(
        self, ctx: _Ctx, query: str, *, include_debug: bool
    ) -> FinalResponse:
        supported_only = False
        for round_index in (0, 1):
            if round_index == 1:
                ctx.goto(PipelineStage.REGENERATE)
                ctx.goto(PipelineStage.GENERATE)
            else:
                ctx.goto(PipelineStage.GENERATE)

            generation = await self.generator.generate(
                query,
                ctx.contexts,
                language=ctx.language,
                latency=ctx.latency,
                supported_only=supported_only,
                is_regeneration=round_index == 1,
            )
            ctx.generation = generation

            if ctx.first_token_ns is None and ctx.latency.generation_ttft is not None:
                ctx.first_token_ns = now_ns()

            ctx.trace.record(
                PipelineStage.GENERATE,
                ok=generation.ok,
                duration_ms=ctx.latency.generation_e2e,
                ttft_ms=ctx.latency.generation_ttft,
                model=generation.model,
                attempts=generation.attempts,
                strict_retry=generation.used_strict_json_retry,
                refused=generation.model_refused,
                error=generation.error,
            )

            if not generation.ok:
                # No hallucinated fallback: the generator being unavailable is
                # reported as an abstention.
                return self._abstain(
                    ctx, generation.abstain_reason,
                    abstain_message(ctx.language), include_debug=include_debug,
                )
            if generation.model_refused:
                return self._abstain(
                    ctx, AbstainReason.MODEL_REFUSED,
                    abstain_message(ctx.language), include_debug=include_debug,
                )

            # ------------------------------------------------- OUTPUT_VALIDATE
            ctx.goto(PipelineStage.OUTPUT_VALIDATE)
            validation = await asyncio.to_thread(
                self.output_guardrail.validate,
                generation,
                ctx.contexts,
                latency=ctx.latency,
                allow_regeneration=round_index == 0,
            )
            ctx.validation = validation
            ctx.trace.record(
                PipelineStage.OUTPUT_VALIDATE,
                duration_ms=ctx.latency.output_guardrail_latency,
                action=validation.action.value,
                citations_valid=validation.citations_valid,
                grounding=validation.grounding_status.value,
                unsupported=len(validation.unsupported_sentences),
                nli_ran=validation.nli_ran,
            )

            if validation.action == ValidationAction.PASS:
                return self._success(ctx, include_debug=include_debug)

            if validation.action == ValidationAction.REGENERATE and round_index == 0:
                logger.info("regenerating once with supported context only")
                ctx.trace.warn("regenerated once after grounding failure")
                # Narrow the evidence to the contexts that actually supported
                # something, so the retry cannot lean on the same weak passage.
                supported = {
                    s.best_context_id
                    for s in validation.sentence_results
                    if s.status == GroundingStatus.ENTAILED and s.best_context_id
                }
                if supported:
                    narrowed = [c for c in ctx.contexts if c.citation_id in supported]
                    if narrowed:
                        ctx.contexts = narrowed
                supported_only = True
                continue

            return self._abstain(
                ctx, validation.reason, abstain_message(ctx.language), include_debug=include_debug
            )

        return self._abstain(
            ctx, AbstainReason.NOT_GROUNDED,
            abstain_message(ctx.language), include_debug=include_debug,
        )

    # -------------------------------------------------------------- terminations
    def _finalize_latency(self, ctx: _Ctx) -> None:
        end = now_ns()
        total = ns_to_ms(end - ctx.t0)
        ctx.latency.total_completion_latency = total

        # total_rag_latency deliberately EXCLUDES STT: it measures transcript-in
        # to validated-answer-out, which is the retrieval/RAG number.
        stt_ms = ctx.latency.stt_latency or 0.0
        ctx.latency.total_rag_latency = round(max(0.0, total - stt_ms), 3)

        # total_voice_latency: audio submitted -> FIRST answer token. Only
        # meaningful when we actually ran STT on audio.
        if ctx.voice and ctx.latency.stt_latency is not None:
            if ctx.first_token_ns is not None:
                ctx.latency.total_voice_latency = ns_to_ms(ctx.first_token_ns - ctx.t0)
            else:
                # Abstained before any token was generated; first "token" is the
                # abstention itself.
                ctx.latency.total_voice_latency = total

    def _debug(self, ctx: _Ctx) -> DebugInfo:
        candidates: list[CandidateDebug] = []
        source = ctx.rerank.candidates if ctx.rerank and ctx.rerank.candidates else (
            ctx.retrieval.candidates if ctx.retrieval else []
        )
        for candidate in source[:20]:
            candidates.append(
                CandidateDebug(
                    chunk_id=candidate.chunk_id,
                    parent_id=candidate.parent_id,
                    language=candidate.language,
                    strategy=candidate.strategy,
                    dense_rank=candidate.dense_rank,
                    sparse_rank=candidate.sparse_rank,
                    fused_score=candidate.fused_score,
                    rerank_score=getattr(candidate, "rerank_score", None),
                    retrieved_by=candidate.retrieved_by,
                    text_preview=candidate.text[:220],
                )
            )
        return DebugInfo(
            trace_id=ctx.trace.trace_id,
            detected_language=ctx.stt.detected_language if ctx.stt else None,
            raw_language_tag=ctx.stt.raw_language_tag if ctx.stt else None,
            language_confidence=ctx.stt.language_confidence if ctx.stt else None,
            is_code_mixed=ctx.stt.is_code_mixed if ctx.stt else False,
            normalized_query=ctx.guard.normalized_query if ctx.guard else None,
            retrieval_mode=ctx.retrieval_mode,
            languages_searched=ctx.languages_searched,
            candidates=candidates,
            selected_chunk_ids=[c.citation_id for c in ctx.contexts],
            gate_top_score=ctx.gate.top_score if ctx.gate else None,
            gate_margin=ctx.gate.margin if ctx.gate else None,
            gate_threshold=ctx.gate.threshold_used if ctx.gate else None,
            thresholds_calibrated=bool(get_thresholds().get("calibrated")),
            grounding_status=ctx.validation.grounding_status if ctx.validation else None,
            unsupported_sentences=ctx.validation.unsupported_sentences if ctx.validation else [],
            invalid_citations=ctx.validation.invalid_citations if ctx.validation else [],
            stage_path=list(ctx.trace.stage_path),
            latency=ctx.latency,
            warnings=list(ctx.trace.warnings),
            reranker_fallback=bool(ctx.rerank and ctx.rerank.fallback_used),
            generation_model=ctx.generation.model if ctx.generation else None,
            corpus_mode=self.settings.ingest_mode,
        )

    def _citations(self, ctx: _Ctx) -> list[Citation]:
        if not ctx.validation:
            return []
        by_id = {c.citation_id: c for c in ctx.contexts}
        out: list[Citation] = []
        for cid in ctx.validation.valid_citations:
            context = by_id.get(cid)
            if context is None:
                continue
            out.append(
                Citation(
                    chunk_id=cid,
                    score=round(context.best_score, 4),
                    language=context.language,
                    strategy=",".join(context.strategies),
                    text=context.text,
                    doc_id=context.doc_id,
                )
            )
        return out

    def _success(self, ctx: _Ctx, *, include_debug: bool) -> FinalResponse:
        ctx.goto(PipelineStage.DONE)
        self._finalize_latency(ctx)
        assert ctx.generation is not None and ctx.validation is not None

        METRICS.record_latency(ctx.latency)
        METRICS.record_request(
            abstained=False, grounded=ctx.validation.grounded, voice=ctx.voice
        )

        return FinalResponse(
            answer=ctx.generation.answer,
            language=ctx.language,
            citations=self._citations(ctx),
            grounded=ctx.validation.grounded,
            abstained=False,
            abstain_reason=AbstainReason.NONE,
            latency_ms=ctx.latency.api_view(),
            trace_id=ctx.trace.trace_id,
            transcript=ctx.stt.transcript if ctx.stt else None,
            detected_language=ctx.stt.detected_language if ctx.stt else None,
            latency_detail=ctx.latency,
            debug=self._debug(ctx) if include_debug else None,
        )

    def _abstain(
        self, ctx: _Ctx, reason: AbstainReason, message: str, *, include_debug: bool
    ) -> FinalResponse:
        if ctx.stage != PipelineStage.ABSTAIN:
            ctx.goto(PipelineStage.ABSTAIN)
        ctx.goto(PipelineStage.DONE)
        self._finalize_latency(ctx)

        METRICS.record_latency(ctx.latency)
        METRICS.record_request(
            abstained=True, grounded=False, abstain_reason=reason.value, voice=ctx.voice
        )
        logger.info("ABSTAIN reason=%s", reason.value)

        return FinalResponse(
            answer=message,
            language=ctx.language,
            citations=[],
            grounded=False,
            abstained=True,
            abstain_reason=reason,
            latency_ms=ctx.latency.api_view(),
            trace_id=ctx.trace.trace_id,
            transcript=ctx.stt.transcript if ctx.stt else None,
            detected_language=ctx.stt.detected_language if ctx.stt else None,
            latency_detail=ctx.latency,
            debug=self._debug(ctx) if include_debug else None,
        )

    def _fail(
        self, ctx: _Ctx, reason: AbstainReason, message: str, *, include_debug: bool
    ) -> FinalResponse:
        try:
            ctx.goto(PipelineStage.ERROR)
            ctx.goto(PipelineStage.DONE)
        except Exception:  # noqa: BLE001
            pass
        self._finalize_latency(ctx)
        METRICS.record_request(
            abstained=True, grounded=False, error=True, abstain_reason=reason.value, voice=ctx.voice
        )
        return FinalResponse(
            answer=message,
            language=ctx.language,
            citations=[],
            grounded=False,
            abstained=True,
            abstain_reason=reason,
            latency_ms=ctx.latency.api_view(),
            trace_id=ctx.trace.trace_id,
            transcript=ctx.stt.transcript if ctx.stt else None,
            latency_detail=ctx.latency,
            debug=self._debug(ctx) if include_debug else None,
        )


_ORCHESTRATOR: RAGOrchestrator | None = None


def get_orchestrator() -> RAGOrchestrator:
    global _ORCHESTRATOR
    if _ORCHESTRATOR is None:
        _ORCHESTRATOR = RAGOrchestrator()
    return _ORCHESTRATOR


def reset_orchestrator() -> None:
    global _ORCHESTRATOR
    _ORCHESTRATOR = None
