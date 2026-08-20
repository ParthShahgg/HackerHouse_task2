"""BGE-M3 encoder producing **dense + learned-sparse** representations.

Why BGE-M3
----------
It is one model that emits both retrieval representations we need, in 100+
languages, and it was explicitly trained for hybrid retrieval plus reranking.
Using it for both branches means dense and sparse see the *same* tokenisation
and the same multilingual vocabulary, so a query never matches lexically in one
space and misses in the other for tokenisation reasons. It also means one model
load (~2.3GB) instead of two, which on a CPU deployment is the difference
between a warm process and an OOM.

Why implemented directly on ``transformers`` rather than via ``FlagEmbedding``
-----------------------------------------------------------------------------
``FlagEmbedding`` is the reference implementation, but it pulls a large
transitive dependency tree (``datasets``, ``accelerate``, ``peft``,
``sentence-transformers``), hides batching/threading behind its own scheduler,
and instantiates its own model copy. On the latency-critical query path we need
explicit control over thread counts, padding and batch composition. The maths
here is exactly the reference behaviour:

* **dense**  = L2-normalised CLS token of the last hidden state.
* **sparse**  = ``relu(sparse_linear(hidden_states))``, max-pooled per token id,
  with special tokens dropped. ``sparse_linear.pt`` is the head published in the
  model repo; without it there is no lexical branch and "hybrid" would be a
  misnomer.

No instruction prefix is prepended. BGE-M3 is trained for symmetric
query/passage encoding, so queries and passages go through the identical path -
which is also why the same encoder can embed sentences for semantic chunking.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.config import get_settings
from app.observability.tracing import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

logger = get_logger(__name__)

__all__ = ["BGEM3Embedder", "EmbeddingOutput", "get_embedder", "reset_embedder"]


@dataclass
class EmbeddingOutput:
    """Batch encoding result."""

    dense: "np.ndarray"
    """Shape ``(n, dim)``, float32, L2-normalised."""

    sparse: list[dict[int, float]]
    """Per-item ``{token_id: weight}`` with weights > 0."""

    dim: int
    n_truncated: int = 0

    def __len__(self) -> int:
        return int(self.dense.shape[0])


class BGEM3Embedder:
    """Lazy-loading BGE-M3 encoder.

    Thread-safe: model loading is guarded, and inference is serialised with a
    lock because torch CPU inference with a fixed intra-op thread pool degrades
    badly under concurrent calls (threads oversubscribe and p99 latency
    explodes). Serialising keeps latency predictable, which matters more here
    than raw throughput on the query path.
    """

    def __init__(
        self,
        model_name: str | None = None,
        *,
        device: str | None = None,
        max_length: int | None = None,
        batch_size: int | None = None,
        use_fp16: bool | None = None,
        num_threads: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.resolved_device()
        self.max_length = max_length or settings.embed_max_length
        self.batch_size = batch_size or settings.embed_batch_size
        self.use_fp16 = settings.fp16_enabled() if use_fp16 is None else use_fp16
        self.num_threads = num_threads or settings.torch_num_threads
        # Pinned so the backbone and the sparse head are guaranteed to come from
        # the same commit.
        self.revision = settings.embedding_model_revision

        self._model: Any = None
        self._tokenizer: Any = None
        self._sparse_linear: Any = None
        self._dim: int | None = None
        self._special_ids: set[int] = set()
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    # ------------------------------------------------------------------ loading
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
            from transformers import AutoConfig, AutoModel, AutoTokenizer

            settings = get_settings()
            if self.device == "cpu" and self.num_threads:
                # Bound the pool explicitly. Left unset, torch grabs every core
                # and latency becomes a function of whatever else is running.
                torch.set_num_threads(self.num_threads)

            logger.info(
                "loading embedding model %s@%s on %s",
                self.model_name, self.revision[:12], self.device,
            )
            token = settings.hf_token or None
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            config = AutoConfig.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            self._model = AutoModel.from_pretrained(
                self.model_name, token=token, revision=self.revision
            )
            self._model.eval().to(self.device)
            if self.use_fp16:
                self._model.half()

            # Single source of truth for the dense dimension: the model config.
            # BGE-M3 is 1024-d, but nothing here hardcodes that.
            self._dim = int(getattr(config, "hidden_size"))

            self._sparse_linear = self._load_sparse_head(config.hidden_size, token)

            tok = self._tokenizer
            self._special_ids = {
                i
                for i in (
                    tok.cls_token_id,
                    tok.eos_token_id,
                    tok.sep_token_id,
                    tok.pad_token_id,
                    tok.unk_token_id,
                    tok.bos_token_id,
                )
                if i is not None
            }
            logger.info(
                "embedder ready dim=%d max_len=%d fp16=%s threads=%s",
                self._dim, self.max_length, self.use_fp16,
                self.num_threads if self.device == "cpu" else "n/a",
            )

    def _load_sparse_head(self, hidden_size: int, token: str | None):
        """Load ``sparse_linear.pt`` - the published lexical-weight head."""
        import torch
        from huggingface_hub import hf_hub_download

        linear = torch.nn.Linear(hidden_size, 1)
        try:
            path = hf_hub_download(
                repo_id=self.model_name,
                filename="sparse_linear.pt",
                token=token,
                revision=self.revision,
            )
            state = torch.load(path, map_location="cpu", weights_only=True)
            linear.load_state_dict(state)
            logger.info("loaded BGE-M3 sparse head from %s", path)
        except Exception as exc:  # noqa: BLE001
            # Fail loudly rather than silently serving an untrained sparse head,
            # which would look like working hybrid retrieval while contributing
            # pure noise to fusion.
            raise RuntimeError(
                f"could not load sparse_linear.pt for {self.model_name}: {exc}. "
                "The sparse branch of hybrid retrieval cannot work without it."
            ) from exc
        linear.eval().to(self.device)
        if self.use_fp16:
            linear.half()
        return linear

    # --------------------------------------------------------------- properties
    @property
    def dim(self) -> int:
        if self._dim is None:
            self.load()
        assert self._dim is not None
        return self._dim

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    # ---------------------------------------------------------------- tokenizer
    def count_tokens(self, text: str) -> int:
        """Exact subword length (no special tokens). Used by chunking."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def encode_token_ids(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token_ids(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    def token_to_text(self, token_id: int) -> str:
        return self.tokenizer.convert_ids_to_tokens([int(token_id)])[0]

    # ----------------------------------------------------------------- encoding
    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
        return_sparse: bool = True,
        show_progress: bool = False,
    ) -> EmbeddingOutput:
        """Encode texts to dense (+ optionally sparse) representations."""
        import numpy as np
        import torch

        if not texts:
            return EmbeddingOutput(dense=np.zeros((0, self.dim), dtype="float32"), sparse=[], dim=self.dim)

        self.load()
        bs = batch_size or self.batch_size

        # Sort by length so each batch pads to a similar width. On CPU this is a
        # large win: padding to the longest item in a mixed batch wastes most of
        # the compute.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        dense_out: list[Any] = [None] * len(texts)
        sparse_out: list[dict[int, float]] = [{} for _ in texts]
        truncated = 0

        with self._infer_lock, torch.inference_mode():
            for start in range(0, len(order), bs):
                idx_batch = order[start : start + bs]
                batch_texts = [texts[i] if texts[i] else " " for i in idx_batch]

                encoded = self._tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                truncated += int(
                    sum(
                        1
                        for i, t in zip(idx_batch, batch_texts, strict=True)
                        if len(self._tokenizer.encode(t, add_special_tokens=True)) > self.max_length
                    )
                    if show_progress
                    else 0
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                hidden = self._model(**encoded).last_hidden_state  # (b, L, H)

                # ---- dense: CLS + L2 norm ----
                cls = hidden[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=-1)
                cls_np = cls.float().cpu().numpy()
                for row, target in enumerate(idx_batch):
                    dense_out[target] = cls_np[row]

                # ---- sparse: relu(W h), max-pooled per token id ----
                if return_sparse:
                    weights = torch.relu(self._sparse_linear(hidden)).squeeze(-1)  # (b, L)
                    weights = weights * encoded["attention_mask"]
                    w_np = weights.float().cpu().numpy()
                    ids_np = encoded["input_ids"].cpu().numpy()
                    for row, target in enumerate(idx_batch):
                        sparse_out[target] = self._collapse_sparse(ids_np[row], w_np[row])

                if show_progress and (start // bs) % 20 == 0:
                    logger.info("  encoded %d/%d", min(start + bs, len(order)), len(order))

        dense = np.vstack([d.reshape(1, -1) for d in dense_out]).astype("float32")
        return EmbeddingOutput(dense=dense, sparse=sparse_out, dim=self.dim, n_truncated=truncated)

    def _collapse_sparse(self, input_ids, weights) -> dict[int, float]:
        """Max-pool weights per token id, dropping special tokens and zeros.

        Max rather than sum: a term repeated five times should not get five times
        the weight, which would let repetition dominate lexical matching.
        """
        out: dict[int, float] = {}
        for token_id, weight in zip(input_ids, weights, strict=True):
            tid = int(token_id)
            if tid in self._special_ids:
                continue
            w = float(weight)
            if w <= 0.0:
                continue
            if w > out.get(tid, 0.0):
                out[tid] = w
        return out

    # ------------------------------------------------------------- convenience
    def encode_query(self, text: str) -> tuple["np.ndarray", dict[int, float]]:
        """Encode a single query. The hot path - one text, one batch."""
        result = self.encode([text], batch_size=1, return_sparse=True)
        return result.dense[0], result.sparse[0]

    def encode_passages(
        self, texts: Sequence[str], *, batch_size: int | None = None, show_progress: bool = True
    ) -> EmbeddingOutput:
        return self.encode(texts, batch_size=batch_size, return_sparse=True, show_progress=show_progress)

    def embed_sentences(self, sentences: Sequence[str]) -> "np.ndarray":
        """Dense-only encoding for the semantic chunker (offline).

        Sparse weights are irrelevant to cosine-similarity breakpoint detection,
        so skipping them saves the per-token pooling loop.
        """
        return self.encode(sentences, return_sparse=False).dense


_EMBEDDER: BGEM3Embedder | None = None
_EMBEDDER_LOCK = threading.Lock()


def get_embedder(**kwargs: Any) -> BGEM3Embedder:
    """Process-wide singleton. One 2.3GB model load, shared by all callers."""
    global _EMBEDDER
    if _EMBEDDER is None:
        with _EMBEDDER_LOCK:
            if _EMBEDDER is None:
                _EMBEDDER = BGEM3Embedder(**kwargs)
    return _EMBEDDER


def reset_embedder() -> None:
    """Test hook."""
    global _EMBEDDER
    with _EMBEDDER_LOCK:
        _EMBEDDER = None
