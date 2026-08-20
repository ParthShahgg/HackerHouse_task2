"""Input guardrail stage: normalise the transcript, then screen it for safety.

Order matters. Normalisation runs first so that safety rules see a canonical
string - otherwise an attacker could evade a regex with zero-width joiners or
padded whitespace, and ASR filler words could break pattern anchors.

What this stage does *not* decide
--------------------------------
Whether the question is answerable. This corpus is broad web text with no topical
domain, so "out-of-corpus" is measured by retrieval, not predicted by rules. See
:mod:`app.retrieval.confidence`.
"""

from __future__ import annotations

from app.config import get_settings
from app.guardrails.safety import check_safety, safety_response
from app.indexing.normalize import normalize_query
from app.observability.tracing import get_logger
from app.schemas.common import SafetyCategory
from app.schemas.query import GuardrailResult

logger = get_logger(__name__)

__all__ = ["apply_input_guardrail", "InputGuardrail"]


class InputGuardrail:
    def __init__(self, *, enable_deep_check: bool | None = None) -> None:
        settings = get_settings()
        self.enable_deep_check = (
            settings.enable_deep_safety_check if enable_deep_check is None else enable_deep_check
        )

    def apply(self, raw_query: str) -> GuardrailResult:
        original = raw_query or ""

        normalized, artifacts = normalize_query(original)

        if not normalized:
            return GuardrailResult(
                allowed=False,
                category=SafetyCategory.EMPTY,
                reason="transcript was empty after normalisation",
                normalized_query="",
                original_query=original,
                artifacts_removed=artifacts,
            )

        verdict = check_safety(normalized, enable_deep_check=self.enable_deep_check)

        deep_ran = False
        if not verdict.safe and verdict.needs_deep_check:
            # Tier 2 only runs when tier 1 already fired, so it can never slow the
            # normal path. It is allowed to *overturn* a tier-1 block, which is how
            # false positives on legitimate informational questions get recovered.
            deep_ran = True
            try:
                from app.guardrails.deep_safety import deep_safety_check

                deep = deep_safety_check(normalized, verdict)
                if deep is not None:
                    verdict = deep
            except Exception as exc:  # noqa: BLE001
                # Fail closed: keep the tier-1 block if the deeper check errors.
                logger.warning("deep safety check unavailable (%s); keeping tier-1 verdict", exc)

        if not verdict.safe:
            logger.info(
                "input blocked: category=%s reason=%s", verdict.category.value, verdict.reason
            )
            return GuardrailResult(
                allowed=False,
                category=verdict.category,
                reason=verdict.reason,
                normalized_query=normalized,
                original_query=original,
                deep_check_ran=deep_ran,
                artifacts_removed=artifacts,
            )

        return GuardrailResult(
            allowed=True,
            category=SafetyCategory.SAFE,
            normalized_query=normalized,
            original_query=original,
            deep_check_ran=deep_ran,
            artifacts_removed=artifacts,
        )


def apply_input_guardrail(query: str) -> GuardrailResult:
    return InputGuardrail().apply(query)


def blocked_message(result: GuardrailResult) -> str:
    return safety_response(result.category)
