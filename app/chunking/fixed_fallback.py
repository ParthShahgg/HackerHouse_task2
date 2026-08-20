"""Strategy D - fixed-size token chunks. Fallback only, never the primary path.

Reserved for pathological passages that exceed the semantic-splitting budget:
scraped pages with no usable sentence punctuation, giant concatenated tables,
navigation dumps. For those, sentence segmentation yields one enormous
"sentence" and semantic splitting has nothing to work with.

Two rules:

* **Token-based, not character-based.** Splitting Devanagari or Tamil on
  character counts cuts inside grapheme clusters and produces garbage. We use
  the real XLM-RoBERTa tokenizer (the same one BGE-M3 uses) so chunk boundaries
  align with what the embedder will actually consume. Character splitting exists
  only as a last resort when no tokenizer was injected.
* **~15-20% overlap**, so a fact straddling a boundary survives in one piece in
  at least one chunk.
"""

from __future__ import annotations

from app.chunking.base import STRATEGY_FIXED_FALLBACK, ChunkerContext, make_chunk
from app.indexing.normalize import approx_token_count
from app.indexing.records import Chunk, ParentPassage

__all__ = ["chunk_fixed_fallback"]


def _character_windows(text: str, ctx: ChunkerContext) -> list[str]:
    """Degraded path used only when no tokenizer is available.

    Still avoids splitting mid-word by snapping to the nearest whitespace.
    """
    approx_chars = max(64, ctx.fixed_chunk_tokens * 4)
    overlap_chars = max(16, ctx.fixed_chunk_overlap_tokens * 4)
    out: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + approx_chars, len(text))
        if end < len(text):
            pivot = text.rfind(" ", start + approx_chars // 2, end)
            if pivot > start:
                end = pivot
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= len(text):
            break
        next_start = max(end - overlap_chars, start + 1)
        # Snap the overlapped start forward to a word boundary. Without this the
        # overlap offset lands mid-word and the following chunk begins with a
        # word fragment ("d105" instead of "word105").
        if next_start > 0 and not text[next_start - 1].isspace():
            boundary = text.find(" ", next_start)
            if 0 <= boundary < end:
                next_start = boundary + 1
        start = next_start
    return out


def chunk_fixed_fallback(parent: ParentPassage, ctx: ChunkerContext) -> list[Chunk]:
    """Emit overlapping fixed-size token chunks for a pathological passage."""
    text = parent.text
    if not text:
        return []

    size = max(32, ctx.fixed_chunk_tokens)
    overlap = max(1, min(ctx.fixed_chunk_overlap_tokens, size // 2))

    pieces: list[str] = []
    if ctx.encode_tokens is not None and ctx.decode_tokens is not None:
        token_ids = ctx.encode_tokens(text)
        if len(token_ids) <= size:
            return []
        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + size]
            if not window:
                break
            decoded = ctx.decode_tokens(window).strip()
            if decoded:
                pieces.append(decoded)
            if start + size >= len(token_ids):
                break
            start += size - overlap
    else:
        if approx_token_count(text) <= size:
            return []
        pieces = _character_windows(text, ctx)

    if len(pieces) <= 1:
        return []

    counter = ctx.count_tokens or approx_token_count
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        if piece == parent.text:
            continue
        chunks.append(
            make_chunk(
                parent,
                strategy=STRATEGY_FIXED_FALLBACK,
                text=piece,
                suffix=f"fx{idx}",
                n_tokens=counter(piece),
            )
        )
    return chunks
