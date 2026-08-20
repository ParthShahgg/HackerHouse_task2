"""Optional tier-2 safety check.

Runs **only** when a cheap rule in :mod:`app.guardrails.safety` has already
fired, and only when ``ENABLE_DEEP_SAFETY_CHECK=true``. Its job is to reduce
false positives: the tier-1 regexes are intentionally blunt, and a broad web-QA
system that refuses "how was the atomic bomb developed?" is not useful.

No model is bundled. Loading a fourth transformer purely for a path that fires on
a small fraction of queries is not a good trade on a CPU deployment, so the
default implementation is a conservative lexical adjudicator that looks for
*informational* framing around an otherwise-flagged phrase. Point
``DEEP_SAFETY_MODEL`` at a classifier and implement :func:`_model_check` if a real
model is wanted.
"""

from __future__ import annotations

import re

from app.guardrails.safety import SafetyVerdict
from app.observability.tracing import get_logger
from app.schemas.common import SafetyCategory

logger = get_logger(__name__)

__all__ = ["deep_safety_check"]

# Framing that indicates a historical/academic/journalistic question rather than
# an operational request.
_INFORMATIONAL = re.compile(
    r"\b(?:histor(?:y|ical)|who (?:invented|discovered|developed)|when (?:was|did)|"
    r"what (?:year|happened)|documentar|museum|textbook|academic|research paper|"
    r"news report|wikipedia|explain the (?:history|background)|effects? of|"
    r"impact of|casualties|treaty|legislation|law against|prevention|symptoms|"
    r"warning signs|how to (?:help|support) someone)\b",
    re.IGNORECASE,
)

# Categories where informational framing is never sufficient to allow the query.
# Actionable synthesis instructions stay blocked regardless of framing.
_NEVER_OVERTURN = {SafetyCategory.WEAPONS, SafetyCategory.SELF_HARM}


def deep_safety_check(query: str, tier1: SafetyVerdict) -> SafetyVerdict | None:
    """Adjudicate a tier-1 block. Return a new verdict, or ``None`` to keep tier-1."""
    if tier1.category in _NEVER_OVERTURN:
        return None

    if tier1.category == SafetyCategory.PROMPT_INJECTION:
        # Injection attempts are about intent toward the system, not topic;
        # informational framing is irrelevant.
        return None

    if _INFORMATIONAL.search(query):
        logger.info(
            "deep safety check overturned tier-1 block (%s): informational framing",
            tier1.category.value,
        )
        return SafetyVerdict(safe=True, category=SafetyCategory.SAFE)

    return None
