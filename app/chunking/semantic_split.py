"""Strategy C - semantic splitting, for unusually long passages only.

Algorithm
---------
1. split the passage into sentences
2. embed each sentence with BGE-M3 (the same model used for retrieval, so the
   similarity space matches what retrieval will actually see)
3. cosine similarity between *neighbouring* sentence embeddings
4. treat the high-distance boundaries as semantic breakpoints
5. cut there, then merge undersized segments back into a neighbour

Scope limits that matter
------------------------
* Runs **only** on long passages (``semantic_split_min_tokens``). On a typical
  3-sentence MSMARCO passage the boundary statistics are meaningless - with two
  distances there is no distribution to take a percentile of.
* Operates strictly **within one original passage**. There is no path in this
  module that could join text from two dataset rows. "Semantic merging" of
  independent MSMARCO passages is explicitly not implemented: it would invent
  documents and manufacture false evidence.
* It is an **offline** cost. Sentence embedding never happens on the query path.

The breakpoint threshold is a percentile of the observed distances for *this
passage*, not a global constant, because absolute cosine distance is not
comparable across languages or topics.
"""

from __future__ import annotations

from app.chunking.base import (
    STRATEGY_SEMANTIC_SPLIT,
    ChunkerContext,
    make_chunk,
)
from app.indexing.normalize import approx_token_count, split_sentences
from app.indexing.records import Chunk, ParentPassage

__all__ = ["chunk_semantic_split", "find_breakpoints"]


def find_breakpoints(similarities: list[float], percentile: float) -> list[int]:
    """Indices *after* which to cut.

    ``similarities[i]`` is cos(sentence i, sentence i+1). A cut after sentence
    ``i`` means ``i`` ends a segment. ``percentile`` is the fraction of
    boundaries (as a percentage) allowed to become breakpoints, so 25.0 keeps
    the quarter most-dissimilar boundaries.
    """
    if not similarities:
        return []
    distances = [1.0 - s for s in similarities]

    ordered = sorted(distances)
    n = len(ordered)
    # Keep the top `percentile`% of distances -> cut above the (100-percentile)th.
    q = max(0.0, min(100.0, 100.0 - percentile))

    # LINEAR INTERPOLATION, deliberately - not nearest-rank. Nearest-rank returns
    # an observed distance, and since the test below is strictly `>`, the single
    # largest gap could never become a breakpoint. With few sentences (the common
    # case) that silently disabled the strategy entirely: a passage with one
    # obvious topic shift produced zero segments.
    position = (q / 100.0) * (n - 1)
    low = int(position)
    high = min(low + 1, n - 1)
    fraction = position - low
    threshold = ordered[low] + (ordered[high] - ordered[low]) * fraction

    # Strictly greater-than, so a passage with uniform distances (all equal)
    # produces no breakpoints instead of shattering into single sentences.
    return [i for i, d in enumerate(distances) if d > threshold]


def _neighbour_similarities(vectors) -> list[float]:
    """Cosine similarity of consecutive rows. Assumes L2-normalised input."""
    import numpy as np

    arr = np.asarray(vectors, dtype="float32")
    if arr.ndim != 2 or arr.shape[0] < 2:
        return []
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = arr / norms
    return [float(np.dot(unit[i], unit[i + 1])) for i in range(arr.shape[0] - 1)]


def chunk_semantic_split(
    parent: ParentPassage,
    ctx: ChunkerContext,
    *,
    sentences: list[str] | None = None,
) -> list[Chunk]:
    """Emit semantically coherent segments of a long passage.

    Falls back to sentence windows when no sentence embedder is available, so a
    misconfigured context degrades to a working strategy rather than dropping
    the passage's child representation entirely.
    """
    sents = sentences if sentences is not None else split_sentences(parent.text)
    if len(sents) < 3:
        return []

    if ctx.embed_sentences is None:
        from app.chunking.sentence_window import chunk_sentence_window

        return chunk_sentence_window(parent, ctx, sentences=sents)

    vectors = ctx.embed_sentences(sents)
    similarities = _neighbour_similarities(vectors)
    if not similarities:
        return []

    cut_after = set(find_breakpoints(similarities, ctx.semantic_split_percentile))

    # Build segments as (start, end) inclusive sentence index ranges.
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(len(sents)):
        if i in cut_after or i == len(sents) - 1:
            segments.append((start, i))
            start = i + 1

    counter = ctx.count_tokens or approx_token_count
    min_tokens = max(1, ctx.semantic_min_segment_tokens)

    # Merge undersized segments forward into the next one (or backward for the
    # final segment), so a stray one-clause sentence never becomes its own
    # vector.
    merged: list[tuple[int, int]] = []
    for seg in segments:
        text = " ".join(sents[seg[0] : seg[1] + 1])
        if merged and counter(text) < min_tokens:
            merged[-1] = (merged[-1][0], seg[1])
        else:
            merged.append(seg)
    if len(merged) >= 2:
        tail_text = " ".join(sents[merged[-1][0] : merged[-1][1] + 1])
        if counter(tail_text) < min_tokens:
            last = merged.pop()
            merged[-1] = (merged[-1][0], last[1])

    # One segment spanning everything == the parent; no child needed.
    if len(merged) <= 1:
        return []

    chunks: list[Chunk] = []
    for idx, (s0, s1) in enumerate(merged):
        text = " ".join(sents[s0 : s1 + 1]).strip()
        if not text or text == parent.text:
            continue
        chunks.append(
            make_chunk(
                parent,
                strategy=STRATEGY_SEMANTIC_SPLIT,
                text=text,
                suffix=f"ss{idx}",
                sentence_start=s0,
                sentence_end=s1,
                n_tokens=counter(text),
            )
        )
    return chunks
