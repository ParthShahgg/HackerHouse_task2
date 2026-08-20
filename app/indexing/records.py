"""Dataclasses for the offline corpus.

Deliberate split of concerns:

* :class:`ParentPassage` / :class:`Chunk` are *retrieval* objects. They contain
  passage content and provenance only.
* :class:`EvalExample` is a *label* object. It holds the query, the ground-truth
  answer and the ``is_selected`` relevance judgements.

The two are persisted to different files and only the former is ever uploaded
to Qdrant. That separation is the mechanism that prevents answer leakage and
retrieval-label leakage - it is structural rather than a convention someone has
to remember.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["DatasetRow", "ParentPassage", "Chunk", "EvalExample", "IngestStats"]


@dataclass(slots=True)
class DatasetRow:
    """One raw MSMARCO-XI record after column projection."""

    query_id: int
    language: str
    split: str
    query: str
    answer: str
    query_type: str
    passages: list[str]
    is_selected: list[int]
    english_passages: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ParentPassage:
    """A deduplicated MSMARCO passage - the natural retrieval unit.

    ``doc_id`` is derived from the content hash, so the same passage appearing
    in 40 different query rows collapses onto one document with one ID.
    """

    doc_id: str
    content_hash: str
    language: str
    text: str
    source_split: str
    n_sentences: int = 0
    n_chars: int = 0
    source_query_ids: list[int] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content_hash": self.content_hash,
            "language": self.language,
            "source_split": self.source_split,
            "text": self.text,
        }


@dataclass(slots=True)
class Chunk:
    """An indexed retrieval representation of (part of) a parent passage.

    For ``strategy == "native"`` the chunk *is* the parent passage and
    ``chunk_id == parent_id``.
    """

    chunk_id: str
    doc_id: str
    parent_id: str
    language: str
    strategy: str
    text: str
    content_hash: str
    source_split: str
    sentence_start: int | None = None
    sentence_end: int | None = None
    n_chars: int = 0
    n_tokens: int | None = None

    def to_payload(self) -> dict[str, Any]:
        """Qdrant payload.

        Note what is *absent*: ``is_selected``, the dataset query, and the
        dataset answer. Those are evaluation-only and must not exist in the
        live index.
        """
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "parent_id": self.parent_id,
            "chunk_id": self.chunk_id,
            "language": self.language,
            "strategy": self.strategy,
            "source_split": self.source_split,
            "content_hash": self.content_hash,
            "text": self.text,
        }
        if self.sentence_start is not None:
            payload["sentence_start"] = self.sentence_start
            payload["sentence_end"] = self.sentence_end
        return payload


@dataclass(slots=True)
class EvalExample:
    """Ground truth for one query. Never indexed as a retrieval document."""

    query_id: int
    language: str
    split: str
    query: str
    answer: str
    query_type: str
    relevant_hashes: list[str] = field(default_factory=list)
    candidate_hashes: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> EvalExample:
        return cls(**data)

    @property
    def has_label(self) -> bool:
        return bool(self.relevant_hashes)


@dataclass(slots=True)
class IngestStats:
    """Per-language ingestion counters, printed by the build script."""

    language: str = ""
    split: str = ""
    rows_processed: int = 0
    passages_seen: int = 0
    unique_parents: int = 0
    duplicates_removed: int = 0
    empty_skipped: int = 0
    child_chunks: int = 0
    chunks_by_strategy: dict[str, int] = field(default_factory=dict)
    vectors_generated: int = 0
    eval_examples: int = 0
    eval_examples_with_labels: int = 0
    stream_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    upload_seconds: float = 0.0
    total_chars: int = 0

    @property
    def avg_chunk_chars(self) -> float:
        total_chunks = self.unique_parents + self.child_chunks
        return round(self.total_chars / total_chunks, 1) if total_chunks else 0.0

    @property
    def total_chunks(self) -> int:
        return self.unique_parents + self.child_chunks

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["avg_chunk_chars"] = self.avg_chunk_chars
        data["total_chunks"] = self.total_chunks
        return data
