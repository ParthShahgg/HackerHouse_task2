"""Passage deduplication keyed on a stable content hash.

Why this is essential rather than an optimisation
-------------------------------------------------
MS MARCO is query-centric: each row carries ~10 candidate passages, and popular
passages recur across many queries. Ingesting rows naively creates one document
per (row, passage) pair, which:

* inflates the index with exact duplicates,
* lets a single passage occupy several slots in the candidate list, starving
  reranking of genuine alternatives,
* and corrupts retrieval metrics, because "the relevant passage" would exist
  under several different IDs and Recall@k becomes ill-defined.

Deduplication uses ``sha256(language + normalized_text)`` (see
:mod:`app.indexing.normalize`). It is scoped per language: the same passage in
Hindi and Marathi is genuinely two retrieval units.

Provenance is preserved, not discarded - each parent keeps the list of
``query_id``s that referenced it, which is what lets the evaluation store map
queries to passage hashes without putting labels in the index.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.indexing.normalize import content_hash, make_doc_id, normalize_text
from app.indexing.records import ParentPassage

__all__ = ["PassageDeduplicator"]

# Cap provenance lists so a passage referenced by 50k queries cannot blow up
# memory during a full-corpus run. Only used for debugging/inspection.
_MAX_PROVENANCE = 32


class PassageDeduplicator:
    """Accumulates unique parent passages across a stream of dataset rows."""

    def __init__(self, *, max_provenance: int = _MAX_PROVENANCE) -> None:
        self._parents: dict[str, ParentPassage] = {}
        self.duplicates_removed = 0
        self.empty_skipped = 0
        self.passages_seen = 0
        self._max_provenance = max_provenance

    # ------------------------------------------------------------------ ingest
    def add(
        self,
        text: str,
        *,
        language: str,
        source_split: str,
        query_id: int | None = None,
    ) -> tuple[str | None, bool]:
        """Register one passage.

        Returns ``(content_hash, is_new)``. ``(None, False)`` when the passage is
        empty after normalisation.
        """
        self.passages_seen += 1
        normalized = normalize_text(text)
        if not normalized:
            self.empty_skipped += 1
            return None, False

        chash = content_hash(language, normalized)
        existing = self._parents.get(chash)
        if existing is not None:
            self.duplicates_removed += 1
            if query_id is not None and len(existing.source_query_ids) < self._max_provenance:
                if query_id not in existing.source_query_ids:
                    existing.source_query_ids.append(query_id)
            return chash, False

        self._parents[chash] = ParentPassage(
            doc_id=make_doc_id(language, chash),
            content_hash=chash,
            language=language,
            text=normalized,
            source_split=source_split,
            n_chars=len(normalized),
            source_query_ids=[query_id] if query_id is not None else [],
        )
        return chash, True

    def add_many(
        self,
        texts: Iterable[str],
        *,
        language: str,
        source_split: str,
        query_id: int | None = None,
    ) -> list[str]:
        """Add several passages, returning the hash of each non-empty one."""
        hashes: list[str] = []
        for text in texts:
            chash, _ = self.add(
                text, language=language, source_split=source_split, query_id=query_id
            )
            if chash is not None:
                hashes.append(chash)
        return hashes

    # ------------------------------------------------------------------ access
    def get(self, chash: str) -> ParentPassage | None:
        return self._parents.get(chash)

    @property
    def parents(self) -> list[ParentPassage]:
        return list(self._parents.values())

    @property
    def by_hash(self) -> dict[str, ParentPassage]:
        return self._parents

    def __len__(self) -> int:
        return len(self._parents)

    def __contains__(self, chash: object) -> bool:
        return chash in self._parents

    @property
    def unique_count(self) -> int:
        return len(self._parents)

    @property
    def dedup_ratio(self) -> float:
        """Fraction of encountered passages that were duplicates."""
        if not self.passages_seen:
            return 0.0
        return round(self.duplicates_removed / self.passages_seen, 4)
