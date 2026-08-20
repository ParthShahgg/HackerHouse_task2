"""Collapse retrieved child chunks back onto their parent passages.

The parent-child idea in one line: **retrieve precisely, generate from coherent
context.**

Child chunks (sentence windows, semantic segments) are tighter retrieval targets
than a whole passage, but they are poor *generation* context - a two-sentence
window can strand a pronoun whose referent lived in the previous sentence. So
after ranking, every retrieved child is mapped back to the full parent passage.

Deduplication is the point of this module. Overlapping windows mean several
children of the same passage routinely rank highly together::

    child_17 -> parent_A      final context:
    child_22 -> parent_A          parent_A
    child_51 -> parent_B   =>     parent_B
    child_73 -> parent_C          parent_C

Without collapsing, the LLM would receive parent_A's text twice, wasting the
context budget and biasing generation toward whichever passage happened to
fragment most. A parent's score is the **best** score among its children (not a
sum), so a passage cannot climb the ranking merely by producing many windows.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.observability.tracing import get_logger
from app.schemas.retrieval import ParentContext, RerankedCandidate

logger = get_logger(__name__)

__all__ = ["expand_to_parents", "build_citation_id"]


def build_citation_id(parent_id: str) -> str:
    """The identifier the LLM is asked to cite.

    Parents are cited rather than children: a citation should point at a passage
    a human can read, and child IDs are an internal indexing detail.
    """
    return parent_id


def expand_to_parents(
    candidates: Sequence[RerankedCandidate],
    *,
    limit: int,
    parent_text_lookup=None,
    enabled: bool = True,
) -> list[ParentContext]:
    """Group reranked candidates by ``parent_id``, preserving rank order.

    Parameters
    ----------
    limit:
        Maximum number of distinct parent contexts to return.
    parent_text_lookup:
        Optional ``callable(parent_id) -> str | None`` used to fetch the full
        parent passage when only a child was retrieved. When it is unavailable or
        returns nothing, the best child's text is used - degraded but never empty.
    enabled:
        When ``False``, each candidate becomes its own context (child text as-is).
        Used by the chunking evaluation to measure retrieval without expansion.
    """
    if not candidates:
        return []

    if not enabled:
        return [
            ParentContext(
                parent_id=c.parent_id,
                doc_id=c.doc_id,
                language=c.language,
                text=c.text,
                best_score=c.rerank_score,
                supporting_chunk_ids=[c.chunk_id],
                strategies=[c.strategy],
                citation_id=c.chunk_id,
            )
            for c in candidates[:limit]
        ]

    grouped: dict[str, ParentContext] = {}
    for candidate in candidates:
        parent_id = candidate.parent_id or candidate.doc_id or candidate.chunk_id
        existing = grouped.get(parent_id)
        if existing is None:
            # Candidates arrive best-first, so the first child seen for a parent
            # is that parent's best child. `best_score` is therefore set once.
            grouped[parent_id] = ParentContext(
                parent_id=parent_id,
                doc_id=candidate.doc_id,
                language=candidate.language,
                text=candidate.text,
                best_score=candidate.rerank_score,
                supporting_chunk_ids=[candidate.chunk_id],
                strategies=[candidate.strategy],
                citation_id=build_citation_id(parent_id),
            )
        else:
            existing.supporting_chunk_ids.append(candidate.chunk_id)
            if candidate.strategy not in existing.strategies:
                existing.strategies.append(candidate.strategy)
            # A native chunk *is* the parent passage, so prefer its text over a
            # window's when both were retrieved.
            if candidate.strategy == "native" and len(candidate.text) > len(existing.text):
                existing.text = candidate.text

    contexts = list(grouped.values())[:limit]

    # Fill in full parent text where we only ever saw a child.
    if parent_text_lookup is not None:
        for context in contexts:
            if "native" in context.strategies:
                continue
            try:
                full = parent_text_lookup(context.parent_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug("parent lookup failed for %s: %s", context.parent_id, exc)
                continue
            if full and len(full) > len(context.text):
                context.text = full

    return contexts
