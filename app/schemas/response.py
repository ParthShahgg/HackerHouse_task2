"""Public API response models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import (
    AbstainReason,
    GroundingStatus,
    LatencyBreakdown,
    PipelineStage,
    RetrievalMode,
)

__all__ = [
    "Citation",
    "DebugInfo",
    "FinalResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "MetricsResponse",
]


class Citation(BaseModel):
    chunk_id: str
    score: float
    language: str | None = None
    strategy: str | None = None
    text: str | None = Field(
        default=None, description="Cited passage text, for source display in the UI."
    )
    doc_id: str | None = None


class CandidateDebug(BaseModel):
    chunk_id: str
    parent_id: str
    language: str
    strategy: str
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = None
    rerank_score: float | None = None
    retrieved_by: list[str] = Field(default_factory=list)
    text_preview: str = ""


class DebugInfo(BaseModel):
    """Everything the developer drawer renders. Never includes secrets."""

    trace_id: str
    detected_language: str | None = None
    raw_language_tag: str | None = None
    language_confidence: float | None = None
    is_code_mixed: bool = False
    normalized_query: str | None = None
    retrieval_mode: RetrievalMode | None = None
    languages_searched: list[str] = Field(default_factory=list)

    candidates: list[CandidateDebug] = Field(default_factory=list)
    selected_chunk_ids: list[str] = Field(default_factory=list)

    gate_top_score: float | None = None
    gate_margin: float | None = None
    gate_threshold: float | None = None
    thresholds_calibrated: bool = False

    grounding_status: GroundingStatus | None = None
    unsupported_sentences: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)

    stage_path: list[PipelineStage] = Field(default_factory=list)
    latency: LatencyBreakdown | None = None
    warnings: list[str] = Field(default_factory=list)
    reranker_fallback: bool = False
    generation_model: str | None = None
    corpus_mode: str | None = None


class FinalResponse(BaseModel):
    """Response contract for ``/api/query`` and ``/api/voice``."""

    answer: str
    language: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = False
    abstained: bool = False
    abstain_reason: AbstainReason = AbstainReason.NONE
    latency_ms: dict[str, float | None] = Field(default_factory=dict)
    trace_id: str

    # Extras beyond the minimum contract - all additive, so a client coded
    # against the base schema keeps working.
    transcript: str | None = None
    detected_language: str | None = None
    latency_detail: LatencyBreakdown | None = None
    debug: DebugInfo | None = None


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal["up", "down"]
    reason: str | None = Field(default=None, max_length=1000)
    expected_answer: str | None = Field(default=None, max_length=2000)


class FeedbackResponse(BaseModel):
    ok: bool = True
    stored: bool = False
    trace_id: str


class ComponentHealth(BaseModel):
    name: str
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    version: str
    corpus_mode: str
    collection: str
    languages: list[str] = Field(default_factory=list)
    vectors: int | None = None
    device: str
    components: list[ComponentHealth] = Field(default_factory=list)
    # Explicit so a demo is never mistaken for a fully-configured deployment.
    missing_secrets: list[str] = Field(default_factory=list)
    thresholds_calibrated: bool = False


class MetricsResponse(BaseModel):
    uptime_s: float
    requests_total: int
    abstentions_total: int
    abstention_rate: float
    errors_total: int
    by_stage_latency_ms: dict[str, dict[str, float]] = Field(default_factory=dict)
    counters: dict[str, int] = Field(default_factory=dict)
    abstain_reasons: dict[str, int] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
