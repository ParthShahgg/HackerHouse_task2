"""Strategy B - sentence-window child chunks.

Rationale
---------
A passage can carry several distinct facts. Embedding it as one vector averages
them, so a query matching one fact competes with the rest of the passage as
noise. Overlapping sentence windows give each fact its own, tighter vector.

::

    parent : S1 S2 S3 S4
    children: [S1 S2]  [S2 S3]  [S3 S4]

Windows overlap (stride 1 with size 2) so a fact spanning a sentence boundary is
still wholly contained in at least one child.

This is *not* fixed-size splitting. Boundaries land on sentence terminators, so
no child ever begins or ends mid-clause. Short passages get no children at all -
for a 2-sentence passage the windows would simply reproduce the parent and
double the index for nothing.

At query time the retrieved child is collapsed back to its parent
(``app.retrieval.parent_expansion``): *retrieve precisely, generate from
coherent context*.
"""

from __future__ import annotations

from app.chunking.base import STRATEGY_SENTENCE_WINDOW, ChunkerContext, make_chunk
from app.indexing.normalize import split_sentences
from app.indexing.records import Chunk, ParentPassage

__all__ = ["chunk_sentence_window"]


def chunk_sentence_window(
    parent: ParentPassage,
    ctx: ChunkerContext,
    *,
    sentences: list[str] | None = None,
) -> list[Chunk]:
    """Emit overlapping sentence-window children.

    Returns ``[]`` when the passage is too short to benefit, which the caller
    treats as "native representation only".
    """
    sents = sentences if sentences is not None else split_sentences(parent.text)
    if len(sents) < max(2, ctx.sentence_window_min_sentences):
        return []

    size = max(1, ctx.sentence_window_size)
    stride = max(1, ctx.sentence_window_stride)

    # A window covering the whole passage duplicates the parent vector, so skip
    # the degenerate case where size >= len(sents).
    if size >= len(sents):
        return []

    chunks: list[Chunk] = []
    seen_spans: set[tuple[int, int]] = set()
    start = 0
    while start < len(sents):
        end = min(start + size, len(sents))
        span = (start, end - 1)
        if span not in seen_spans:
            text = " ".join(sents[start:end]).strip()
            if text and text != parent.text:
                seen_spans.add(span)
                chunks.append(
                    make_chunk(
                        parent,
                        strategy=STRATEGY_SENTENCE_WINDOW,
                        text=text,
                        suffix=f"sw{start}_{end - 1}",
                        sentence_start=start,
                        sentence_end=end - 1,
                        n_tokens=ctx.count_tokens(text) if ctx.count_tokens else None,
                    )
                )
        if end >= len(sents):
            break
        start += stride

    return chunks
