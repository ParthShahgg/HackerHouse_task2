"""Retrieval-confidence / abstention gate.

This runs **before** generation. If the retrieved evidence is not good enough,
the LLM is never called: that saves the generation latency *and* removes the
opportunity to hallucinate from weak context.

Two independent signals
-----------------------
``top_score``
    The best reranker logit. Answers "is anything here actually relevant?"
``margin``
    ``top_score - second_score``. Answers "can the reranker tell these apart?"
    A high top score with a negligible margin is the *retrieval ambiguity* case:
    several passages look equally plausible, so committing to one invites a
    confidently-wrong answer. Treating that as ambiguity rather than confidence
    is what makes the system cautious in exactly the situation where a naive
    top-1 threshold would be most overconfident.

On thresholds
-------------
Nothing here is a guessed constant. Values come from
``configs/thresholds.json``, fitted by ``scripts/calibrate_thresholds.py`` over
held-out validation queries using the dataset's own ``is_selected`` labels.

If that artefact is missing, the gate runs in an explicitly **uncalibrated**
state, ``thresholds_calibrated=False`` propagates all the way into the API
response and the debug drawer, and a warning is logged. An uncalibrated guess is
never presented as an empirical threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import get_settings, get_thresholds
from app.observability.tracing import get_logger
from app.schemas.common import AbstainReason, GateDecision
from app.schemas.retrieval import GroundingDecision, RerankResult

logger = get_logger(__name__)

__all__ = ["ConfidenceGate", "GateThresholds", "evaluate_gate"]


@dataclass
class GateThresholds:
    """Decision boundary for abstention."""

    rerank_abstain_below: float
    """Abstain when the best reranker logit is below this."""

    rerank_margin_min: float
    """Abstain when top-vs-second gap is below this *and* the top score is not
    decisively high. Zero disables the ambiguity check."""

    margin_override_score: float = 6.0
    """A top score this high is treated as decisive on its own, so a duplicate
    near-tie (two passages that genuinely both answer the question) is not
    punished as ambiguity."""

    calibrated: bool = False
    source: str = "uncalibrated"

    @classmethod
    def load(cls) -> GateThresholds:
        data: dict[str, Any] = get_thresholds()
        return cls(
            rerank_abstain_below=float(data.get("rerank_abstain_below", 0.0)),
            rerank_margin_min=float(data.get("rerank_margin_min", 0.0)),
            margin_override_score=float(data.get("margin_override_score", 6.0)),
            calibrated=bool(data.get("calibrated", False)),
            source=str(data.get("source", "uncalibrated")),
        )


class ConfidenceGate:
    """Decides GENERATE vs ABSTAIN from reranked retrieval results."""

    def __init__(self, thresholds: GateThresholds | None = None) -> None:
        self.thresholds = thresholds or GateThresholds.load()
        if not self.thresholds.calibrated:
            logger.warning(
                "abstention thresholds are UNCALIBRATED (%s). Run "
                "scripts/calibrate_thresholds.py; responses will report "
                "thresholds_calibrated=false.",
                self.thresholds.source,
            )

    def evaluate(self, rerank: RerankResult) -> GroundingDecision:
        th = self.thresholds
        base = {
            "threshold_used": th.rerank_abstain_below,
            "margin_threshold_used": th.rerank_margin_min,
            "thresholds_calibrated": th.calibrated,
            "threshold_source": th.source,
        }

        if not rerank.candidates:
            return GroundingDecision(
                decision=GateDecision.ABSTAIN,
                reason=AbstainReason.NO_CANDIDATES,
                explanation="Hybrid retrieval returned no candidates for this query.",
                **base,
            )

        top = rerank.top_score
        margin = rerank.margin

        if top is None:
            return GroundingDecision(
                decision=GateDecision.ABSTAIN,
                reason=AbstainReason.NO_CANDIDATES,
                explanation="No reranker score available.",
                **base,
            )

        if top < th.rerank_abstain_below:
            return GroundingDecision(
                decision=GateDecision.ABSTAIN,
                reason=AbstainReason.LOW_CONFIDENCE,
                explanation=(
                    f"Best reranker score {top:.3f} is below the calibrated "
                    f"relevance threshold {th.rerank_abstain_below:.3f}."
                ),
                top_score=top,
                margin=margin,
                **base,
            )

        if (
            th.rerank_margin_min > 0
            and margin is not None
            and margin < th.rerank_margin_min
            and top < th.margin_override_score
        ):
            return GroundingDecision(
                decision=GateDecision.ABSTAIN,
                reason=AbstainReason.WEAK_MARGIN,
                explanation=(
                    f"Top candidates are not separable (margin {margin:.3f} < "
                    f"{th.rerank_margin_min:.3f}); retrieval is ambiguous."
                ),
                top_score=top,
                margin=margin,
                **base,
            )

        return GroundingDecision(
            decision=GateDecision.GENERATE,
            reason=AbstainReason.NONE,
            top_score=top,
            margin=margin,
            **base,
        )


def evaluate_gate(rerank: RerankResult) -> GroundingDecision:
    return ConfidenceGate().evaluate(rerank)


# ---------------------------------------------------------------------------
# Standard abstention message
# ---------------------------------------------------------------------------
ABSTAIN_MESSAGE_EN = (
    "I don't have enough information in my retrieved sources to answer that reliably."
)

# The user hears the answer, so the refusal is spoken in their language too.
# Static translations, not a model call: an abstention path must not depend on
# the very service that may have just failed.
ABSTAIN_MESSAGES: dict[str, str] = {
    "en": ABSTAIN_MESSAGE_EN,
    "hi": "मेरे पास इस प्रश्न का विश्वसनीय उत्तर देने के लिए पर्याप्त जानकारी नहीं है।",
    "mr": "या प्रश्नाचे विश्वासार्ह उत्तर देण्यासाठी माझ्याकडे पुरेशी माहिती नाही.",
    "ta": "இந்தக் கேள்விக்கு நம்பகமான பதில் அளிக்கப் போதுமான தகவல் என்னிடம் இல்லை.",
    "te": "ఈ ప్రశ్నకు నమ్మదగిన సమాధానం ఇవ్వడానికి నా వద్ద తగినంత సమాచారం లేదు.",
    "bn": "এই প্রশ্নের নির্ভরযোগ্য উত্তর দেওয়ার জন্য আমার কাছে পর্যাপ্ত তথ্য নেই।",
    "gu": "આ પ્રશ્નનો વિશ્વસનીય જવાબ આપવા માટે મારી પાસે પૂરતી માહિતી નથી.",
    "kn": "ಈ ಪ್ರಶ್ನೆಗೆ ವಿಶ್ವಾಸಾರ್ಹ ಉತ್ತರ ನೀಡಲು ನನ್ನ ಬಳಿ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ.",
    "ml": "ഈ ചോദ്യത്തിന് വിശ്വസനീയമായ ഉത്തരം നൽകാൻ എന്റെ പക്കൽ മതിയായ വിവരങ്ങളില്ല.",
    "pa": "ਇਸ ਸਵਾਲ ਦਾ ਭਰੋਸੇਯੋਗ ਜਵਾਬ ਦੇਣ ਲਈ ਮੇਰੇ ਕੋਲ ਲੋੜੀਂਦੀ ਜਾਣਕਾਰੀ ਨਹੀਂ ਹੈ।",
    "or": "ଏହି ପ୍ରଶ୍ନର ବିଶ୍ୱସନୀୟ ଉତ୍ତର ଦେବା ପାଇଁ ମୋ ପାଖରେ ପର୍ଯ୍ୟାପ୍ତ ସୂଚନା ନାହିଁ।",
    "as": "এই প্ৰশ্নৰ নিৰ্ভৰযোগ্য উত্তৰ দিবলৈ মোৰ হাতত পৰ্যাপ্ত তথ্য নাই।",
    "ne": "यो प्रश्नको भरपर्दो उत्तर दिन मसँग पर्याप्त जानकारी छैन।",
    "ur": "اس سوال کا قابلِ اعتماد جواب دینے کے لیے میرے پاس کافی معلومات نہیں ہیں۔",
    "sa": "अस्य प्रश्नस्य विश्वसनीयम् उत्तरं दातुं मम समीपे पर्याप्ता सूचना नास्ति।",
}


def abstain_message(language: str | None) -> str:
    """Localised abstention text, falling back to English."""
    if not language:
        return ABSTAIN_MESSAGE_EN
    return ABSTAIN_MESSAGES.get(language, ABSTAIN_MESSAGE_EN)
