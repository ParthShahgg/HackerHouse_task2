"""Shared enums plus the latency instrumentation primitives.

All timing uses :func:`time.perf_counter_ns` - a monotonic clock. ``time.time``
is wall-clock and can step backwards (NTP, DST), which silently corrupts
percentile reports.
"""

from __future__ import annotations

import time
from types import TracebackType
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "AbstainReason",
    "GateDecision",
    "GroundingStatus",
    "LatencyBreakdown",
    "PipelineStage",
    "RetrievalMode",
    "SafetyCategory",
    "Stopwatch",
    "StrEnum",
    "ValidationAction",
    "now_ns",
    "ns_to_ms",
]

try:  # py>=3.11
    from enum import StrEnum
except ImportError:  # pragma: no cover
    from enum import Enum

    class StrEnum(str, Enum):  # type: ignore[no-redef]
        pass


def now_ns() -> int:
    return time.perf_counter_ns()


def ns_to_ms(ns: int) -> float:
    return round(ns / 1_000_000, 3)


class PipelineStage(StrEnum):
    START = "START"
    STT = "STT"
    INPUT_GUARD = "INPUT_GUARD"
    EMBED = "EMBED"
    RETRIEVE = "RETRIEVE"
    RERANK = "RERANK"
    CONFIDENCE_GATE = "CONFIDENCE_GATE"
    GENERATE = "GENERATE"
    OUTPUT_VALIDATE = "OUTPUT_VALIDATE"
    REGENERATE = "REGENERATE"
    ABSTAIN = "ABSTAIN"
    DONE = "DONE"
    ERROR = "ERROR"


class GateDecision(StrEnum):
    GENERATE = "GENERATE"
    ABSTAIN = "ABSTAIN"


class ValidationAction(StrEnum):
    PASS = "PASS"
    REGENERATE = "REGENERATE"
    ABSTAIN = "ABSTAIN"


class GroundingStatus(StrEnum):
    ENTAILED = "ENTAILED"
    NOT_ENTAILED = "NOT_ENTAILED"
    UNKNOWN = "UNKNOWN"
    SKIPPED = "SKIPPED"


class RetrievalMode(StrEnum):
    """Whether the language filter was applied, and why."""

    LANGUAGE_FILTERED = "language_filtered"
    CROSS_LINGUAL = "cross_lingual"
    CODE_MIXED_CROSS_LINGUAL = "code_mixed_cross_lingual"


class SafetyCategory(StrEnum):
    SAFE = "safe"
    SELF_HARM = "self_harm"
    WEAPONS = "weapons"
    ILLICIT = "illicit"
    HATE = "hate"
    PROMPT_INJECTION = "prompt_injection"
    EMPTY = "empty"
    TOO_LONG = "too_long"


class AbstainReason(StrEnum):
    NONE = "none"
    INPUT_BLOCKED = "input_blocked"
    NO_CANDIDATES = "no_candidates"
    LOW_CONFIDENCE = "low_confidence"
    WEAK_MARGIN = "weak_margin"
    INVALID_CITATION = "invalid_citation"
    NOT_GROUNDED = "not_grounded"
    GENERATION_UNAVAILABLE = "generation_unavailable"
    GENERATION_MALFORMED = "generation_malformed"
    MODEL_REFUSED = "model_refused"
    RETRIEVAL_ERROR = "retrieval_error"
    INTERNAL_ERROR = "internal_error"


class LatencyBreakdown(BaseModel):
    """Per-stage latencies in milliseconds.

    ``None`` means *not measured* and is deliberately distinct from ``0.0``,
    which means *measured as sub-microsecond*. Reports must never coerce an
    unmeasured stage to zero - that is how fake benchmark numbers appear.
    """

    stt_latency: float | None = None
    guardrail_latency: float | None = None
    query_embedding_latency: float | None = None
    dense_latency: float | None = None
    sparse_latency: float | None = None
    rrf_latency: float | None = None
    rerank_latency: float | None = None
    grounding_gate_latency: float | None = None
    generation_ttft: float | None = None
    generation_e2e: float | None = None
    output_guardrail_latency: float | None = None
    nli_latency: float | None = None

    # Aggregates.
    total_rag_latency: float | None = None
    """Transcript received -> final validated textual answer."""

    total_voice_latency: float | None = None
    """Audio submitted -> first answer token (includes STT). None for text-only."""

    total_completion_latency: float | None = None
    """Audio/transcript in -> final answer token."""

    @property
    def retrieval_total(self) -> float | None:
        """Dense + sparse + fusion, or ``None`` if none of them was measured.

        Returning ``None`` rather than ``0.0`` for the all-unmeasured case is
        deliberate: a ``0`` here reads as "instantaneous retrieval", which is
        exactly the kind of number that turns an unmeasured stage into a
        performance claim.

        Component stages may run concurrently, so this is a sum of measured work,
        not necessarily wall-clock.
        """
        measured = [
            v
            for v in (self.dense_latency, self.sparse_latency, self.rrf_latency)
            if v is not None
        ]
        return sum(measured) if measured else None

    def api_view(self) -> dict[str, float | None]:
        """The compact ``latency_ms`` object in the public API contract."""
        retrieval = self.retrieval_total
        return {
            "retrieval": round(retrieval, 3) if retrieval is not None else None,
            "rerank": self.rerank_latency,
            "generation_ttft": self.generation_ttft,
            "total": self.total_rag_latency,
        }


class Stopwatch:
    """Context manager recording a stage duration into a LatencyBreakdown field.

    >>> lat = LatencyBreakdown()
    >>> with Stopwatch(lat, "rerank_latency") as sw:
    ...     ...
    >>> lat.rerank_latency is not None
    True

    Records even when the body raises, so a timeout still reports how long it
    waited before giving up.
    """

    __slots__ = ("_breakdown", "_field", "_start", "elapsed_ms")

    def __init__(self, breakdown: LatencyBreakdown | None = None, field: str | None = None) -> None:
        self._breakdown = breakdown
        self._field = field
        self._start = 0
        self.elapsed_ms: float = 0.0

    def __enter__(self) -> Stopwatch:
        self._start = now_ns()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.elapsed_ms = ns_to_ms(now_ns() - self._start)
        if self._breakdown is not None and self._field is not None:
            setattr(self._breakdown, self._field, self.elapsed_ms)

    def lap(self) -> float:
        return ns_to_ms(now_ns() - self._start)


class StageEvent(BaseModel):
    """One entry in the orchestrator's execution trace."""

    stage: PipelineStage
    ok: bool = True
    duration_ms: float | None = None
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
