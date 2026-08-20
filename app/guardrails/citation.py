"""Citation validation: every cited id must exist in the retrieved set.

A fabricated source id is the most deceptive hallucination a RAG system can emit,
because the citation is exactly what makes the answer look verifiable. So the
check is pure set membership against the ids actually sent to the model - there is
no fuzzy matching that could let a near-miss through.

Policy
------
* Any citation not in the retrieved set -> **reject the output**. Not "drop the
  bad citation and keep the answer": if the model referenced evidence that does
  not exist, the reasoning behind the answer is already untrustworthy.
* A non-empty answer with **zero** valid citations is also rejected. Under this
  prompt, an answer with no citations is ungrounded by construction.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.observability.tracing import get_logger
from app.schemas.common import AbstainReason, ValidationAction
from app.schemas.generation import ValidationResult

logger = get_logger(__name__)

__all__ = ["validate_citations", "normalise_citation"]


def normalise_citation(raw: str) -> str:
    """Strip decoration a model may add around an id.

    Tolerates ``[hi:abc]``, ``"hi:abc"``, ``id=hi:abc`` and trailing punctuation.
    This is *presentation* normalisation only - the resulting string must still
    match a retrieved id exactly.
    """
    value = (raw or "").strip()
    for prefix in ("id=", "id:", "chunk_id=", "chunk_id:"):
        if value.lower().startswith(prefix):
            value = value[len(prefix) :]
    return value.strip().strip("[]()<>\"'`").rstrip(".,;:").strip()


def validate_citations(
    citations: Iterable[str],
    retrieved_ids: Sequence[str],
    *,
    answer: str = "",
    require_at_least_one: bool = True,
) -> ValidationResult:
    """Check cited ids against the retrieved set."""
    allowed = set(retrieved_ids)
    valid: list[str] = []
    invalid: list[str] = []

    for raw in citations:
        candidate = normalise_citation(raw)
        if not candidate:
            continue
        if candidate in allowed:
            if candidate not in valid:
                valid.append(candidate)
        else:
            invalid.append(raw.strip() if isinstance(raw, str) else str(raw))

    if invalid:
        logger.warning(
            "rejecting output: %d fabricated citation(s) %s not in retrieved set",
            len(invalid), invalid[:3],
        )
        return ValidationResult(
            action=ValidationAction.ABSTAIN,
            reason=AbstainReason.INVALID_CITATION,
            explanation=(
                f"Generated answer cited {len(invalid)} source id(s) that were not "
                f"in the retrieved set: {invalid[:3]}."
            ),
            citations_valid=False,
            valid_citations=valid,
            invalid_citations=invalid,
        )

    if require_at_least_one and answer.strip() and not valid:
        logger.warning("rejecting output: answer present but no valid citations")
        return ValidationResult(
            action=ValidationAction.ABSTAIN,
            reason=AbstainReason.INVALID_CITATION,
            explanation="Generated answer cited no retrievable source.",
            citations_valid=False,
            valid_citations=[],
            invalid_citations=[],
        )

    return ValidationResult(
        action=ValidationAction.PASS,
        reason=AbstainReason.NONE,
        citations_valid=True,
        valid_citations=valid,
        invalid_citations=[],
    )
