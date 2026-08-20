"""Strategy A - native passage. The default representation and the baseline.

``1 MSMARCO passage = 1 parent chunk``

This is the control condition in the chunking evaluation. Any other strategy has
to beat it on measured Recall@5 / MRR / nDCG@10 to justify its extra vectors,
and the default configuration keeps native precisely because the source data is
already passage-segmented.
"""

from __future__ import annotations

from app.chunking.base import STRATEGY_NATIVE, ChunkerContext, make_chunk
from app.indexing.records import Chunk, ParentPassage

__all__ = ["chunk_native"]


def chunk_native(parent: ParentPassage, ctx: ChunkerContext | None = None) -> list[Chunk]:
    """Return the passage as a single chunk, verbatim after normalisation.

    No truncation happens here. Length capping is the embedder's concern (it
    truncates at ``EMBED_MAX_LENGTH`` tokens), and passages long enough for that
    to matter are routed to semantic/fixed strategies for their *child*
    representation anyway.
    """
    if not parent.text:
        return []
    n_tokens = ctx.count_tokens(parent.text) if ctx and ctx.count_tokens else None
    return [
        make_chunk(
            parent,
            strategy=STRATEGY_NATIVE,
            text=parent.text,
            suffix=None,  # chunk_id == parent_id == doc_id
            n_tokens=n_tokens,
        )
    ]
