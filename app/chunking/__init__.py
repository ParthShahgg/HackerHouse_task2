"""Chunking strategies and the router that selects between them.

The four strategies are *not* applied all at once. Every passage gets its native
representation, plus **at most one** child representation chosen by passage
shape:

===============================  ==========================================
passage shape                    child strategy
===============================  ==========================================
< 3 sentences                    none (native only)
>= 3 sentences, normal length    ``sentence_window``
>= ``semantic_split_min_tokens`` ``semantic_split``
>= ``fixed_fallback_min_tokens`` ``fixed_fallback``
===============================  ==========================================

Choosing one child strategy per passage - rather than emitting all four - keeps
the index from carrying three near-duplicate representations of the same text,
which would inflate vector count, slow retrieval, and let one passage crowd out
genuine diversity in the candidate list.

``chunk_forced`` exists so ``scripts/evaluate_chunking.py`` can build a
single-strategy index per arm and compare them on equal footing.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from app.chunking.base import (
    ALL_STRATEGIES,
    STRATEGY_FIXED_FALLBACK,
    STRATEGY_NATIVE,
    STRATEGY_SEMANTIC_SPLIT,
    STRATEGY_SENTENCE_WINDOW,
    ChunkerContext,
)
from app.chunking.fixed_fallback import chunk_fixed_fallback
from app.chunking.native import chunk_native
from app.chunking.semantic_split import chunk_semantic_split
from app.chunking.sentence_window import chunk_sentence_window
from app.indexing.normalize import approx_token_count, split_sentences
from app.indexing.records import Chunk, ParentPassage

__all__ = [
    "ALL_STRATEGIES",
    "STRATEGY_FIXED_FALLBACK",
    "STRATEGY_NATIVE",
    "STRATEGY_SEMANTIC_SPLIT",
    "STRATEGY_SENTENCE_WINDOW",
    "ChunkerContext",
    "ChunkingEngine",
    "chunk_fixed_fallback",
    "chunk_native",
    "chunk_semantic_split",
    "chunk_sentence_window",
]


class ChunkingEngine:
    """Routes passages to strategies and produces the final chunk list."""

    def __init__(
        self,
        ctx: ChunkerContext | None = None,
        enabled: Sequence[str] | None = None,
    ) -> None:
        self.ctx = ctx or ChunkerContext.from_settings()
        requested = list(enabled) if enabled is not None else list(ALL_STRATEGIES)
        unknown = set(requested) - set(ALL_STRATEGIES)
        if unknown:
            raise ValueError(f"Unknown chunking strategies: {sorted(unknown)}")
        # native is non-negotiable: it is the generation context and the
        # evaluation baseline.
        if STRATEGY_NATIVE not in requested:
            requested.insert(0, STRATEGY_NATIVE)
        self.enabled: tuple[str, ...] = tuple(requested)

    # ------------------------------------------------------------------ helpers
    def _count_tokens(self, text: str) -> int:
        counter = self.ctx.count_tokens or approx_token_count
        return counter(text)

    def route(self, parent: ParentPassage, sentences: list[str] | None = None) -> str | None:
        """Which child strategy applies to this passage, if any."""
        text = parent.text
        if not text:
            return None
        sents = sentences if sentences is not None else split_sentences(text)
        n_tokens = self._count_tokens(text)

        if (
            n_tokens >= self.ctx.fixed_fallback_min_tokens
            and STRATEGY_FIXED_FALLBACK in self.enabled
        ):
            return STRATEGY_FIXED_FALLBACK
        if (
            n_tokens >= self.ctx.semantic_split_min_tokens
            and STRATEGY_SEMANTIC_SPLIT in self.enabled
        ):
            return STRATEGY_SEMANTIC_SPLIT
        if (
            len(sents) >= max(2, self.ctx.sentence_window_min_sentences)
            and STRATEGY_SENTENCE_WINDOW in self.enabled
        ):
            return STRATEGY_SENTENCE_WINDOW
        return None

    # ------------------------------------------------------------------- public
    def chunk(self, parent: ParentPassage) -> list[Chunk]:
        """Production chunking: native + at most one child representation."""
        if not parent.text:
            return []
        sentences = split_sentences(parent.text)
        parent.n_sentences = len(sentences)
        parent.n_chars = len(parent.text)

        chunks = chunk_native(parent, self.ctx)

        strategy = self.route(parent, sentences)
        children: list[Chunk] = []
        if strategy == STRATEGY_SENTENCE_WINDOW:
            children = chunk_sentence_window(parent, self.ctx, sentences=sentences)
        elif strategy == STRATEGY_SEMANTIC_SPLIT:
            children = chunk_semantic_split(parent, self.ctx, sentences=sentences)
            # A long passage that resists semantic segmentation still deserves a
            # finer representation; fall through to fixed windows.
            if not children and STRATEGY_FIXED_FALLBACK in self.enabled:
                children = chunk_fixed_fallback(parent, self.ctx)
        elif strategy == STRATEGY_FIXED_FALLBACK:
            children = chunk_fixed_fallback(parent, self.ctx)

        chunks.extend(children)
        return chunks

    def chunk_forced(self, parent: ParentPassage, strategy: str) -> list[Chunk]:
        """Apply exactly one strategy, ignoring routing. For evaluation arms.

        ``native`` returns the passage. Any other strategy returns *only* its
        children, falling back to the native chunk when the passage is too short
        for that strategy to produce anything - otherwise an evaluation arm would
        silently lose documents and report misleadingly low recall.
        """
        if strategy not in ALL_STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}")
        if not parent.text:
            return []

        sentences = split_sentences(parent.text)
        parent.n_sentences = len(sentences)
        parent.n_chars = len(parent.text)

        if strategy == STRATEGY_NATIVE:
            return chunk_native(parent, self.ctx)
        if strategy == STRATEGY_SENTENCE_WINDOW:
            children = chunk_sentence_window(parent, self.ctx, sentences=sentences)
        elif strategy == STRATEGY_SEMANTIC_SPLIT:
            children = chunk_semantic_split(parent, self.ctx, sentences=sentences)
        else:
            children = chunk_fixed_fallback(parent, self.ctx)
        return children or chunk_native(parent, self.ctx)

    def chunk_many(self, parents: Iterable[ParentPassage]) -> list[Chunk]:
        out: list[Chunk] = []
        for parent in parents:
            out.extend(self.chunk(parent))
        return out
