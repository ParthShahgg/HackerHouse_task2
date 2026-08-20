"""Generation and output-validation stage models."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import AbstainReason, GroundingStatus, ValidationAction

__all__ = [
    "GeneratedAnswer",
    "GenerationResult",
    "SentenceGrounding",
    "ValidationResult",
]


class GeneratedAnswer(BaseModel):
    """The exact structured output contract required of the LLM.

    Kept intentionally minimal: this schema is embedded in the prompt and used
    as the ``response_format`` JSON schema, so every extra field is another way
    for a small model to produce something unparseable.
    """

    answer: str
    citations: list[str] = Field(default_factory=list)

    @field_validator("citations", mode="before")
    @classmethod
    def _coerce_citations(cls, v: object) -> object:
        # Small models sometimes emit a single string or a list of objects.
        # Normalising here is cheaper than a second round-trip to the model,
        # and it does not weaken validation: IDs are still checked for
        # membership in the retrieved set by the citation guardrail.
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    out.append(item)
                elif isinstance(item, dict):
                    for key in ("chunk_id", "id", "citation_id", "citation"):
                        if isinstance(item.get(key), str):
                            out.append(item[key])
                            break
            return out
        return v


class GenerationResult(BaseModel):
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    model: str | None = None
    ok: bool = True
    error: str | None = None
    abstain_reason: AbstainReason = AbstainReason.NONE

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None

    attempts: int = 1
    used_strict_json_retry: bool = False
    is_regeneration: bool = False
    # True when the model itself declined for lack of evidence, which is a
    # legitimate abstention rather than a failure.
    model_refused: bool = False


class SentenceGrounding(BaseModel):
    """Per-sentence entailment verdict from the NLI guardrail."""

    sentence: str
    status: GroundingStatus
    score: float | None = None
    best_context_id: str | None = None
    is_factual: bool = True
    method: str = "nli"
    """nli | deterministic_span | trivial"""


class ValidationResult(BaseModel):
    """Output guardrail verdict: citation validity + entailment grounding."""

    action: ValidationAction = ValidationAction.PASS
    reason: AbstainReason = AbstainReason.NONE
    explanation: str | None = None

    citations_valid: bool = True
    valid_citations: list[str] = Field(default_factory=list)
    invalid_citations: list[str] = Field(default_factory=list)

    grounding_status: GroundingStatus = GroundingStatus.SKIPPED
    sentence_results: list[SentenceGrounding] = Field(default_factory=list)
    unsupported_sentences: list[str] = Field(default_factory=list)
    nli_model: str | None = None
    nli_ran: bool = False

    @property
    def grounded(self) -> bool:
        """Grounded == citations check out AND nothing material is unentailed.

        ``UNKNOWN`` is treated as *not* grounded when it applies to a factual
        sentence: the guardrail fails closed by design.
        """
        return self.citations_valid and self.grounding_status in (
            GroundingStatus.ENTAILED,
            GroundingStatus.SKIPPED,
        )
