"""Cross-encoder reranking with ``BAAI/bge-reranker-v2-m3``.

Why a reranker at all
---------------------
Dense and sparse retrieval score a query against a *precomputed* document
vector, so the document never actually sees the query. A cross-encoder attends
over the concatenated pair, which resolves the cases bi-encoders systematically
get wrong: negation, quantities, and near-duplicate passages that differ in one
decisive detail. In this system it also supplies the **calibrated signal the
abstention gate depends on** - fused RRF scores are rank-derived and carry no
information about whether the top document is actually relevant, so they cannot
gate anything.

Why this model
--------------
``bge-reranker-v2-m3`` shares the XLM-RoBERTa backbone and tokenizer with
BGE-M3, so it is multilingual over the same vocabulary as the retriever and
needs no per-language configuration.

Cost control
------------
Reranking is O(candidates), each a full forward pass over query+passage. It runs
**only** on the fused candidate set (default 30), never the corpus. On CPU this
is the dominant stage of the pipeline; the measured figures are in
``reports/latency.md`` and are not smoothed over.

Scores are raw logits, roughly (-12, +12). They are deliberately *not* squashed
before thresholding: sigmoid saturates at both tails, which destroys the margin
resolution the ambiguity check relies on.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.observability.tracing import get_logger

logger = get_logger(__name__)

__all__ = ["BGEReranker", "get_reranker", "reset_reranker", "sigmoid"]


def sigmoid(x: float) -> float:
    import math

    # Branch to avoid overflow in exp for large-magnitude logits.
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class BGEReranker:
    """Lazy-loading multilingual cross-encoder."""

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        use_fp16: bool | None = None,
        num_threads: int | None = None,
        quantize_int8: bool | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.resolved_device()
        self.max_length = max_length or settings.rerank_max_length
        self.batch_size = batch_size or settings.rerank_batch_size
        self.use_fp16 = settings.fp16_enabled() if use_fp16 is None else use_fp16
        self.num_threads = num_threads or settings.torch_num_threads
        self.revision = settings.reranker_model_revision
        self.use_int8 = settings.int8_reranker_enabled() if quantize_int8 is None else quantize_int8
        self.quantized = False

        self._model: Any = None
        self._tokenizer: Any = None
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

            settings = get_settings()
            if self.device == "cpu" and self.num_threads:
                torch.set_num_threads(self.num_threads)

            logger.info(
                "loading reranker %s@%s on %s", self.model_name, self.revision[:12], self.device
            )
            token = settings.hf_token or None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            self._model.eval().to(self.device)
            if self.use_fp16:
                self._model.half()
            elif self.use_int8:
                self._apply_int8()
            logger.info(
                "reranker ready max_len=%d fp16=%s int8=%s threads=%s",
                self.max_length, self.use_fp16, self.quantized,
                self.num_threads if self.device == "cpu" else "n/a",
            )

    def _apply_int8(self) -> None:
        """Dynamically quantize Linear layers to int8 (CPU only).

        Chosen after measurement, not on principle: this model dominates CPU
        latency, and int8 cut 30-candidate reranking from 5230ms to 4063ms while
        leaving the relevant-vs-irrelevant logit margin essentially unchanged
        (11.41 -> 11.06).

        Consequence for calibration: absolute logits shift slightly, so
        abstention thresholds must be calibrated with the SAME setting used at
        serving time. ``scripts/calibrate_thresholds.py`` therefore runs through
        this identical code path.
        """
        import torch

        try:
            self._model = torch.quantization.quantize_dynamic(
                self._model, {torch.nn.Linear}, dtype=torch.qint8
            ).eval()
            self.quantized = True
        except Exception as exc:  # noqa: BLE001
            # Quality/latency optimisation only - never fatal.
            logger.warning("int8 quantization unavailable (%s); staying fp32", exc)
            self.quantized = False

    def score(
        self,
        query: str,
        passages: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> list[float]:
        """Return one relevance logit per passage, in input order."""
        if not passages:
            return []
        import torch

        self.load()
        bs = batch_size or self.batch_size

        # Length-sorted batching: the shortest passages no longer pay the padding
        # cost of the longest one in the batch.
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        scores: list[float] = [0.0] * len(passages)

        with self._infer_lock, torch.inference_mode():
            for start in range(0, len(order), bs):
                idx_batch = order[start : start + bs]
                pairs = [[query, passages[i] or " "] for i in idx_batch]
                encoded = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits.view(-1).float().cpu()
                for row, target in enumerate(idx_batch):
                    scores[target] = float(logits[row])
        return scores

    def rerank(
        self,
        query: str,
        passages: Sequence[str],
        *,
        top_k: int | None = None,
        batch_size: int | None = None,
    ) -> list[tuple[int, float]]:
        """Return ``(original_index, score)`` sorted best-first."""
        scores = self.score(query, passages, batch_size=batch_size)
        ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k] if top_k else ranked


_RERANKER: BGEReranker | None = None
_RERANKER_LOCK = threading.Lock()


def get_reranker(**kwargs: Any) -> BGEReranker:
    global _RERANKER
    if _RERANKER is None:
        with _RERANKER_LOCK:
            if _RERANKER is None:
                _RERANKER = BGEReranker(**kwargs)
    return _RERANKER


def reset_reranker() -> None:
    global _RERANKER
    with _RERANKER_LOCK:
        _RERANKER = None
