"""Input safety checks: cheap deterministic rules, optional deep model.

Two-tier by design, for latency
-------------------------------
Tier 1 runs on **every** query: compiled regexes over a normalised string, on the
order of tens of microseconds. Tier 2 (a classifier model) runs **only when tier 1
signals something**, so the normal path never pays for it.

Scope discipline
----------------
This layer answers "is this input unsafe or an attack?" It deliberately does
*not* try to answer "can we answer this?". MSMARCO-XI is broad web text with no
topical boundary, so no static rule can predict answerability - that question is
settled empirically by retrieval and the calibrated confidence gate. Conflating
the two would mean either refusing answerable questions or claiming a domain the
corpus does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.schemas.common import SafetyCategory

__all__ = ["SafetyVerdict", "check_safety", "MAX_QUERY_CHARS", "MIN_QUERY_CHARS"]

MAX_QUERY_CHARS = 2000
MIN_QUERY_CHARS = 2


@dataclass
class SafetyVerdict:
    safe: bool
    category: SafetyCategory = SafetyCategory.SAFE
    reason: str | None = None
    needs_deep_check: bool = False
    matched: str | None = None


def _compile(patterns: list[str]) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE | re.UNICODE)


# Narrow, high-precision patterns. Broad matching here would refuse legitimate
# informational questions, which for a web-corpus QA system is the more likely
# and more damaging error.
_SELF_HARM = _compile([
    r"\bhow (?:can|do) i (?:kill|hurt|harm) myself\b",
    r"\b(?:best|easiest|painless) way to (?:die|kill myself|end my life)\b",
    r"\bi want to (?:kill myself|end my life|commit suicide)\b",
    r"\bhow to (?:commit )?suicide\b",
    r"आत्महत्या (?:कैसे|कैसा) कर",
    r"தற்கொலை செய்வது எப்படி",
])

_WEAPONS = _compile([
    r"\bhow to (?:make|build|synthes(?:ise|ize)|manufacture) (?:a |an )?"
    r"(?:bomb|explosive|ied|pipe bomb|nerve agent|sarin|vx|nuclear weapon|dirty bomb)\b",
    r"\b(?:synthes(?:ise|ize)|produce) (?:ricin|anthrax|sarin|novichok|vx nerve)\b",
    r"\bbuild (?:a )?(?:nuclear|chemical|biological) weapon\b",
    r"\bबम (?:कैसे )?बनाने? (?:की विधि|का तरीका|कैसे)\b",
])

_ILLICIT = _compile([
    r"\bhow to (?:synthes(?:ise|ize)|cook|manufacture) (?:meth|methamphetamine|fentanyl|heroin|mdma)\b",
    r"\bhow to (?:hack|break) into (?:someone(?:'s)?|a|his|her|their) (?:account|email|phone|wifi|bank)\b",
    r"\bhow to (?:launder money|make (?:a )?fake (?:passport|id|currency))\b",
    r"\b(?:credit card|ssn) (?:dump|skimm?er|generator)\b",
    r"\bhow to (?:stalk|track) (?:someone|my (?:wife|husband|ex|girlfriend|boyfriend))\b",
])

_HATE = _compile([
    r"\b(?:all|every) (?:muslims?|hindus?|christians?|sikhs?|jews?|dalits?) (?:are|should be) "
    r"(?:killed|destroyed|eliminated|subhuman|vermin)\b",
    r"\bwhy (?:are|is) (?:the )?\w+ (?:race|caste|religion) (?:inferior|subhuman)\b",
    r"\bgenocide of\b",
])

# Attempts to subvert *our* instructions (as opposed to injection arriving via
# retrieved passages, which is handled in app.generation.prompts).
_INJECTION = _compile([
    r"\bignore (?:all )?(?:your |the |previous |above )?(?:instructions|rules|prompt)\b",
    r"\bdisregard (?:all )?(?:previous|prior|above) (?:instructions|rules)\b",
    r"\b(?:reveal|show|print|repeat|output) (?:me )?(?:your )?(?:system )?prompt\b",
    r"\byou are now (?:a|an|in) \w+",
    r"\b(?:developer|dev|debug|god) mode\b",
    r"\bDAN mode\b",
    r"<\|im_(?:start|end)\|>",
    r"\bpretend (?:you are|to be) (?:an? )?(?:unrestricted|uncensored|evil)\b",
])

_RULES: list[tuple[re.Pattern[str], SafetyCategory, str]] = [
    (_SELF_HARM, SafetyCategory.SELF_HARM, "self-harm content"),
    (_WEAPONS, SafetyCategory.WEAPONS, "weapons / CBRN synthesis"),
    (_ILLICIT, SafetyCategory.ILLICIT, "illicit activity"),
    (_HATE, SafetyCategory.HATE, "hate speech"),
    (_INJECTION, SafetyCategory.PROMPT_INJECTION, "prompt-injection attempt"),
]


def check_safety(query: str, *, enable_deep_check: bool = False) -> SafetyVerdict:
    """Tier-1 safety screen.

    Returns ``needs_deep_check=True`` when a rule fired and a second-opinion model
    is enabled, letting the caller decide whether to spend that latency.
    """
    if query is None:
        return SafetyVerdict(False, SafetyCategory.EMPTY, "empty query")

    stripped = query.strip()
    if len(stripped) < MIN_QUERY_CHARS:
        return SafetyVerdict(False, SafetyCategory.EMPTY, "query too short to be meaningful")
    if len(stripped) > MAX_QUERY_CHARS:
        return SafetyVerdict(
            False,
            SafetyCategory.TOO_LONG,
            f"query exceeds {MAX_QUERY_CHARS} characters",
        )

    for pattern, category, reason in _RULES:
        match = pattern.search(stripped)
        if match:
            return SafetyVerdict(
                safe=False,
                category=category,
                reason=reason,
                needs_deep_check=enable_deep_check,
                matched=match.group(0)[:80],
            )

    return SafetyVerdict(safe=True)


# Category-specific responses. Self-harm gets crisis resources rather than a bare
# refusal; the rest get a short, non-preachy decline.
SAFETY_RESPONSES: dict[SafetyCategory, str] = {
    SafetyCategory.SELF_HARM: (
        "I can't help with that. If you're in distress, please contact emergency "
        "services (112 in India, 911 in the US) or a crisis line such as "
        "Tele-MANAS at 14416 (India) or 988 (US). You deserve support."
    ),
    SafetyCategory.WEAPONS: "I can't help with that request.",
    SafetyCategory.ILLICIT: "I can't help with that request.",
    SafetyCategory.HATE: "I can't help with that request.",
    SafetyCategory.PROMPT_INJECTION: (
        "I can only answer questions using my retrieved sources. Please ask a "
        "question about the information you're looking for."
    ),
    SafetyCategory.EMPTY: "I didn't catch a question. Could you say that again?",
    SafetyCategory.TOO_LONG: "That query is too long. Please ask something shorter.",
}


def safety_response(category: SafetyCategory) -> str:
    return SAFETY_RESPONSES.get(category, "I can't help with that request.")
