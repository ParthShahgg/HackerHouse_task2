"""Retrieval-side stage models: candidates -> RRF -> rerank -> abstention gate."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.common import AbstainReason, GateDecision, RetrievalMode

__all__ = [
    "RetrievalCandidate",
    "RetrievalResult",
    "RerankedCandidate",
    "RerankResult",
    "ParentContext",
    "GroundingDecision",
]


class RetrievalCandidate(BaseModel):
    """One chunk returned by hybrid retrieval, before reranking.

    Note the absence of ``is_selected``. Relevance labels live in the offline
    evaluation store only; if they were on this object they could leak into the
    live ranking path.
    """

    chunk_id: str
    parent_id: str
    doc_id: str
    language: str
    strategy: str
    text: str
    source_split: str | None = None
    content_hash: str | None = None
    sentence_start: int | None = None
    sentence_end: int | None = None

    dense_score: float | None = None
    sparse_score: float | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None
    fused_score: float | None = Field(default=None, description="RRF score.")
    retrieved_by: list[str] = Field(
        default_factory=list, description="Which branches surfaced it: dense / sparse."
    )


class RetrievalResult(BaseModel):
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    mode: RetrievalMode = RetrievalMode.CROSS_LINGUAL
    languages_searched: list[str] = Field(default_factory=list)
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    used_server_side_fusion: bool = True
    degraded: bool = False
    degraded_reason: str | None = None

    @property
    def is_empty(self) -> bool:
        return not self.candidates


class RerankedCandidate(RetrievalCandidate):
    """Candidate carrying a cross-encoder score.

    ``rerank_score`` is the raw bge-reranker-v2-m3 logit (unbounded, centred
    near 0). ``rerank_prob`` is its sigmoid. Thresholds are calibrated on the
    logit because sigmoid saturation destroys margin resolution at the tails.
    """

    rerank_score: float
    rerank_prob: float | None = None
    rerank_rank: int | None = None


class ParentContext(BaseModel):
    """A parent passage assembled for generation.

    Produced by collapsing every retrieved child that shares a ``parent_id``,
    so the LLM sees one coherent passage instead of overlapping fragments.
    """

    parent_id: str
    doc_id: str
    language: str
    text: str
    best_score: float
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    # Stable label the LLM must cite; also what citation validation checks.
    citation_id: str = ""


class RerankResult(BaseModel):
    candidates: list[RerankedCandidate] = Field(default_factory=list)
    contexts: list[ParentContext] = Field(default_factory=list)
    considered: int = 0
    fallback_used: bool = False
    fallback_reason: str | None = None
    model: str | None = None

    @property
    def top_score(self) -> float | None:
        return self.candidates[0].rerank_score if self.candidates else None

    @property
    def margin(self) -> float | None:
        """Gap between the best and second-best candidate.

        A high top score with a negligible margin means the reranker cannot
        actually discriminate - treated as ambiguity, not confidence.
        """
        if len(self.candidates) < 2:
            return None
        return self.candidates[0].rerank_score - self.candidates[1].rerank_score


class GroundingDecision(BaseModel):
    """Pre-generation abstention gate verdict."""

    decision: GateDecision
    reason: AbstainReason = AbstainReason.NONE
    explanation: str | None = None
    top_score: float | None = None
    margin: float | None = None
    threshold_used: float | None = None
    margin_threshold_used: float | None = None
    # Surfaced everywhere so an uncalibrated guess is never mistaken for an
    # empirically chosen threshold.
    thresholds_calibrated: bool = False
    threshold_source: str | None = None
