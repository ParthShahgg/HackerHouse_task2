"""Shared types for the chunking strategies.

Guiding principle for this whole package
----------------------------------------
MSMARCO passages are *already* human-curated retrieval units (median ~250-400
characters, 1-6 sentences). So the default representation must be the passage
itself. Blindly cutting every passage into 500-token windows would be strictly
destructive here, and merging passages from different query rows would fabricate
documents that never existed and create false co-occurrence evidence.

Every strategy in this package therefore obeys two invariants:

1. **No cross-passage merging.** A chunk's text is always a contiguous span of
   exactly one source passage.
2. **The parent is always retained.** Child chunks exist to sharpen *retrieval*;
   generation always reads a coherent parent passage.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from app.indexing.records import Chunk, ParentPassage

__all__ = [
    "STRATEGY_NATIVE",
    "STRATEGY_SENTENCE_WINDOW",
    "STRATEGY_SEMANTIC_SPLIT",
    "STRATEGY_FIXED_FALLBACK",
    "ALL_STRATEGIES",
    "ChunkerContext",
    "SentenceEmbedder",
    "TokenCounter",
    "make_chunk",
]

STRATEGY_NATIVE = "native"
STRATEGY_SENTENCE_WINDOW = "sentence_window"
STRATEGY_SEMANTIC_SPLIT = "semantic_split"
STRATEGY_FIXED_FALLBACK = "fixed_fallback"

ALL_STRATEGIES: tuple[str, ...] = (
    STRATEGY_NATIVE,
    STRATEGY_SENTENCE_WINDOW,
    STRATEGY_SEMANTIC_SPLIT,
    STRATEGY_FIXED_FALLBACK,
)


class SentenceEmbedder(Protocol):
    """Minimal interface the semantic splitter needs.

    Kept as a Protocol so chunking never imports the embedder module: that
    avoids a circular dependency and lets tests inject a deterministic stub
    instead of loading 2.3GB of weights.
    """

    def __call__(self, sentences: Sequence[str]) -> object:  # -> np.ndarray (n, d), L2-normalised
        ...


class TokenCounter(Protocol):
    def __call__(self, text: str) -> int: ...


@dataclass
class ChunkerContext:
    """Tunables + injected collaborators for the chunking strategies."""

    sentence_window_min_sentences: int = 3
    sentence_window_size: int = 2
    sentence_window_stride: int = 1

    semantic_split_min_tokens: int = 320
    semantic_split_percentile: float = 25.0
    semantic_min_segment_tokens: int = 32

    fixed_fallback_min_tokens: int = 1024
    fixed_chunk_tokens: int = 256
    fixed_chunk_overlap_tokens: int = 45

    # Injected. When absent the strategies degrade gracefully rather than crash:
    # semantic split falls back to sentence windows, and fixed fallback uses a
    # whitespace tokenizer.
    embed_sentences: SentenceEmbedder | None = None
    count_tokens: TokenCounter | None = None
    encode_tokens: Callable[[str], list[int]] | None = None
    decode_tokens: Callable[[list[int]], str] | None = None

    @classmethod
    def from_settings(cls, settings=None, **overrides) -> ChunkerContext:
        from app.config import get_settings

        s = settings or get_settings()
        base = cls(
            sentence_window_min_sentences=s.sentence_window_min_sentences,
            sentence_window_size=s.sentence_window_size,
            sentence_window_stride=s.sentence_window_stride,
            semantic_split_min_tokens=s.semantic_split_min_tokens,
            semantic_split_percentile=s.semantic_split_percentile,
            fixed_fallback_min_tokens=s.fixed_fallback_min_tokens,
            fixed_chunk_tokens=s.fixed_chunk_tokens,
            fixed_chunk_overlap_tokens=s.fixed_chunk_overlap_tokens,
        )
        for key, value in overrides.items():
            if value is not None:
                setattr(base, key, value)
        return base


def make_chunk(
    parent: ParentPassage,
    *,
    strategy: str,
    text: str,
    suffix: str | None = None,
    sentence_start: int | None = None,
    sentence_end: int | None = None,
    n_tokens: int | None = None,
) -> Chunk:
    """Build a :class:`Chunk` with consistent identifiers.

    For the native strategy ``chunk_id == parent_id == doc_id``, which is what
    makes "retrieved a native chunk" and "retrieved the parent" the same event
    and keeps parent-child collapsing uniform across strategies.
    """
    chunk_id = parent.doc_id if suffix is None else f"{parent.doc_id}#{suffix}"
    return Chunk(
        chunk_id=chunk_id,
        doc_id=parent.doc_id,
        parent_id=parent.doc_id,
        language=parent.language,
        strategy=strategy,
        text=text,
        content_hash=parent.content_hash,
        source_split=parent.source_split,
        sentence_start=sentence_start,
        sentence_end=sentence_end,
        n_chars=len(text),
        n_tokens=n_tokens,
    )
