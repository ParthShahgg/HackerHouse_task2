"""Multilingual NLI entailment check for output grounding.

Purpose
-------
Citation validation proves the model *pointed at* real evidence. It cannot prove
the answer actually follows from that evidence - a model can cite a valid passage
and still assert something the passage does not support. Entailment closes that
gap: each factual sentence of the answer becomes a hypothesis, the retrieved
context is the premise, and we ask whether the premise entails it.

Model
-----
``MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`` by default,
configurable via ``NLI_MODEL``. Chosen because it is genuinely multilingual (XNLI
covers Hindi, Urdu and Swahili among others), and at ~560MB it is small enough to
sit alongside a 2.3GB embedder and a 2.3GB reranker on one CPU box. Label order is
read from ``config.id2label`` rather than assumed - the index of "entailment"
varies between NLI checkpoints, and hardcoding it silently inverts the guardrail.

Cost control
------------
Entailment runs over the **final answer sentences vs the selected retrieved
context only** - never over the corpus. Three further reductions:

* Non-factual sentences (greetings, hedges) are skipped.
* A sentence that is a near-verbatim span of the context is grounded
  *deterministically* by string containment, with no model call. This is the
  common case for extractive answers and is provably safe: if the text appears in
  the evidence, it is supported by it.
* All remaining (sentence, context) pairs go through as one batch.

Never fails open: a model error, or an inconclusive verdict on a factual
sentence, yields ``UNKNOWN``, which the grounding policy treats as not-grounded.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.indexing.normalize import normalize_text, split_sentences
from app.observability.tracing import get_logger
from app.schemas.common import GroundingStatus
from app.schemas.generation import SentenceGrounding

logger = get_logger(__name__)

__all__ = ["NLIGrounder", "get_nli_grounder", "reset_nli_grounder", "is_factual_sentence"]

# Sentences that assert nothing checkable. Excluded so a polite lead-in cannot
# fail the grounding check for the whole answer.
_NON_FACTUAL = re.compile(
    r"^(?:"
    r"(?:hi|hello|hey|sure|okay|ok|yes|no|thanks|thank you)\b"
    r"|i (?:hope|think|believe) (?:this|that) helps"
    r"|(?:let me know|feel free)\b"
    r"|based on (?:the )?(?:retrieved )?(?:passages?|sources?|evidence|context)\b"
    r"|according to (?:the )?(?:passages?|sources?|evidence|context)\b"
    r")",
    re.IGNORECASE,
)

_MIN_FACTUAL_CHARS = 12


def is_factual_sentence(sentence: str) -> bool:
    """Whether a sentence makes a checkable factual claim."""
    text = sentence.strip()
    if len(text) < _MIN_FACTUAL_CHARS:
        return False
    if _NON_FACTUAL.match(text):
        return False
    # A clarifying question asserts nothing.
    if text.endswith("?") and len(text) < 80:
        return False
    return True


def _normalise_for_containment(text: str) -> str:
    """Aggressive fold used only for the deterministic containment shortcut."""
    folded = normalize_text(text).lower()
    # Drop punctuation/space differences so minor reformatting still matches.
    return re.sub(r"[\s\.,;:!\?\-–—'\"()\[\]।॥۔]+", "", folded)


class NLIGrounder:
    """Sentence-level entailment verification."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        threshold: float | None = None,
        max_length: int = 512,
        allow_deterministic_skip: bool | None = None,
        num_threads: int | None = None,
    ) -> None:
        s = get_settings()
        self.model_name = model_name or s.nli_model
        self.device = device or s.resolved_device()
        self.threshold = s.nli_entailment_threshold if threshold is None else threshold
        self.max_length = max_length
        self.allow_deterministic_skip = (
            s.nli_skip_on_deterministic_grounding
            if allow_deterministic_skip is None
            else allow_deterministic_skip
        )
        self.num_threads = num_threads or s.torch_num_threads
        self.revision = s.nli_model_revision

        self._model: Any = None
        self._tokenizer: Any = None
        self._entail_idx: int | None = None
        self._contra_idx: int | None = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            s = get_settings()
            if self.device == "cpu" and self.num_threads:
                torch.set_num_threads(self.num_threads)

            logger.info(
                "loading NLI model %s@%s on %s", self.model_name, self.revision[:12], self.device
            )
            token = s.hf_token or None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            self._model.eval().to(self.device)

            # Read label positions from the checkpoint. Hardcoding index 0 for
            # entailment is a real footgun: several NLI checkpoints order labels
            # contradiction-first, which would inverse the guardrail.
            id2label = getattr(self._model.config, "id2label", {}) or {}
            for idx, label in id2label.items():
                name = str(label).lower()
                if "entail" in name:
                    self._entail_idx = int(idx)
                elif "contra" in name:
                    self._contra_idx = int(idx)
            if self._entail_idx is None:
                logger.warning(
                    "could not locate an 'entailment' label in %s (id2label=%s); "
                    "defaulting to index 0",
                    self.model_name, id2label,
                )
                self._entail_idx = 0
            logger.info(
                "NLI ready: entailment=%s contradiction=%s threshold=%.2f",
                self._entail_idx, self._contra_idx, self.threshold,
            )

    # ------------------------------------------------------------------- scoring
    def _score_pairs(self, pairs: Sequence[tuple[str, str]]) -> list[tuple[float, float]]:
        """Return ``(p_entail, p_contradiction)`` for each (premise, hypothesis)."""
        if not pairs:
            return []
        import torch

        self.load()
        with self._infer_lock, torch.inference_mode():
            encoded = self._tokenizer(
                [p for p, _ in pairs],
                [h for _, h in pairs],
                padding=True,
                truncation="only_first",
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            logits = self._model(**encoded).logits
            probs = torch.softmax(logits.float(), dim=-1).cpu()

        out: list[tuple[float, float]] = []
        for row in probs:
            entail = float(row[self._entail_idx])
            contra = float(row[self._contra_idx]) if self._contra_idx is not None else 0.0
            out.append((entail, contra))
        return out

    def verify(
        self,
        answer: str,
        contexts: Sequence[str],
        *,
        context_ids: Sequence[str] | None = None,
        max_contexts: int = 3,
    ) -> tuple[GroundingStatus, list[SentenceGrounding]]:
        """Verify each factual sentence of ``answer`` against ``contexts``.

        Returns the overall status plus per-sentence detail. Overall status is the
        *worst* verdict across factual sentences: one unsupported claim makes the
        answer unsupported.
        """
        sentences = split_sentences(answer)
        if not sentences:
            return GroundingStatus.SKIPPED, []

        pool = list(contexts[:max_contexts])
        ids = list(context_ids or [])[: len(pool)]
        if not pool:
            return (
                GroundingStatus.UNKNOWN,
                [
                    SentenceGrounding(
                        sentence=s, status=GroundingStatus.UNKNOWN, method="no_context"
                    )
                    for s in sentences
                ],
            )

        folded_contexts = [_normalise_for_containment(c) for c in pool]
        results: list[SentenceGrounding] = []
        pending: list[tuple[int, int, str, str]] = []  # (result_idx, ctx_idx, premise, hypothesis)

        for sentence in sentences:
            if not is_factual_sentence(sentence):
                results.append(
                    SentenceGrounding(
                        sentence=sentence,
                        status=GroundingStatus.SKIPPED,
                        is_factual=False,
                        method="trivial",
                    )
                )
                continue

            # Deterministic shortcut: verbatim span of the evidence.
            if self.allow_deterministic_skip:
                folded_sentence = _normalise_for_containment(sentence)
                hit = next(
                    (
                        i
                        for i, ctx in enumerate(folded_contexts)
                        if folded_sentence and folded_sentence in ctx
                    ),
                    None,
                )
                if hit is not None:
                    results.append(
                        SentenceGrounding(
                            sentence=sentence,
                            status=GroundingStatus.ENTAILED,
                            score=1.0,
                            best_context_id=ids[hit] if hit < len(ids) else None,
                            method="deterministic_span",
                        )
                    )
                    continue

            placeholder = SentenceGrounding(
                sentence=sentence, status=GroundingStatus.UNKNOWN, method="nli"
            )
            results.append(placeholder)
            idx = len(results) - 1
            for ctx_idx, premise in enumerate(pool):
                pending.append((idx, ctx_idx, premise, sentence))

        if pending:
            try:
                scores = self._score_pairs([(p, h) for _, _, p, h in pending])
            except Exception as exc:  # noqa: BLE001
                # Fail closed - leave the sentences UNKNOWN.
                logger.error("NLI scoring failed (%s); treating as UNKNOWN", exc)
                scores = []

            if scores:
                best: dict[int, tuple[float, float, int]] = {}
                for (res_idx, ctx_idx, _, _), (entail, contra) in zip(
                    pending, scores, strict=True
                ):
                    current = best.get(res_idx)
                    if current is None or entail > current[0]:
                        best[res_idx] = (entail, contra, ctx_idx)

                for res_idx, (entail, contra, ctx_idx) in best.items():
                    target = results[res_idx]
                    target.score = round(entail, 4)
                    target.best_context_id = ids[ctx_idx] if ctx_idx < len(ids) else None
                    if entail >= self.threshold:
                        target.status = GroundingStatus.ENTAILED
                    elif contra > entail and contra >= self.threshold:
                        target.status = GroundingStatus.NOT_ENTAILED
                    else:
                        # Neutral: the evidence neither supports nor refutes it.
                        # Not grounded, by policy.
                        target.status = GroundingStatus.UNKNOWN

        factual = [r for r in results if r.is_factual]
        if not factual:
            return GroundingStatus.SKIPPED, results
        if any(r.status == GroundingStatus.NOT_ENTAILED for r in factual):
            return GroundingStatus.NOT_ENTAILED, results
        if any(r.status == GroundingStatus.UNKNOWN for r in factual):
            return GroundingStatus.UNKNOWN, results
        return GroundingStatus.ENTAILED, results


_GROUNDER: NLIGrounder | None = None
_GROUNDER_LOCK = threading.Lock()


def get_nli_grounder(**kwargs: Any) -> NLIGrounder:
    global _GROUNDER
    if _GROUNDER is None:
        with _GROUNDER_LOCK:
            if _GROUNDER is None:
                _GROUNDER = NLIGrounder(**kwargs)
    return _GROUNDER


def reset_nli_grounder() -> None:
    global _GROUNDER
    with _GROUNDER_LOCK:
        _GROUNDER = None
