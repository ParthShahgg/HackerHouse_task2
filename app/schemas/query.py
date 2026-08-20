"""Input-side stage models: request -> STT -> guardrail -> query embedding."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import SafetyCategory

__all__ = [
    "QueryRequest",
    "STTResult",
    "TranscriptSegment",
    "GuardrailResult",
    "QueryEmbeddingResult",
]


class QueryRequest(BaseModel):
    """Typed JSON input for ``POST /api/query`` (the text debugging path)."""

    query: str = Field(min_length=1, max_length=4000)
    language: str | None = Field(
        default=None,
        description=(
            "Optional ISO-639-1 hint (hi, mr, ta, te...). When omitted the "
            "retriever runs cross-lingual instead of guessing."
        ),
    )
    top_k: int | None = Field(default=None, ge=1, le=50)
    include_debug: bool = False

    @field_validator("query")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v


class TranscriptSegment(BaseModel):
    text: str
    language: str | None = None
    start_s: float | None = None
    end_s: float | None = None
    is_final: bool = False


class STTResult(BaseModel):
    """Output of the Sarvam Saaras v3 stage.

    Carries detected-language metadata (an explicit task requirement) plus
    enough provenance to tell a streaming WebSocket transcript apart from a
    REST-fallback one - which matters when reading latency reports.
    """

    transcript: str
    detected_language: str | None = Field(
        default=None, description="Canonical ISO-639-1 code, e.g. 'hi'."
    )
    raw_language_tag: str | None = Field(
        default=None, description="Exactly what Sarvam returned, e.g. 'hi-IN'."
    )
    language_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_code_mixed: bool = False
    segments: list[TranscriptSegment] = Field(default_factory=list)
    transport: str = Field(default="websocket", description="websocket | rest | injected")
    used_fallback: bool = False
    attempts: int = 1
    model: str | None = None
    audio_duration_s: float | None = None
    partial_count: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.transcript.strip()


class GuardrailResult(BaseModel):
    """Input guardrail verdict.

    Distinguishes *unsafe* (block outright) from *unsupported / out-of-corpus*
    (decided later, by retrieval, because this corpus is broad web text and no
    static rule can predict answerability).
    """

    allowed: bool = True
    category: SafetyCategory = SafetyCategory.SAFE
    reason: str | None = None
    normalized_query: str = ""
    original_query: str = ""
    # True when cheap rules fired and a deeper safety model was consulted.
    deep_check_ran: bool = False
    artifacts_removed: list[str] = Field(default_factory=list)


class QueryEmbeddingResult(BaseModel):
    """BGE-M3 query representation: dense vector + learned sparse weights."""

    dense: list[float]
    sparse_indices: list[int] = Field(default_factory=list)
    sparse_values: list[float] = Field(default_factory=list)
    dim: int
    model: str
    truncated: bool = False

    @field_validator("dense")
    @classmethod
    def _non_empty(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("dense vector must not be empty")
        return v

    def sparse_dict(self) -> dict[int, float]:
        return dict(zip(self.sparse_indices, self.sparse_values, strict=True))
