"""Test object factories.

Kept in a separately-importable module rather than in ``conftest.py`` because
``from tests.conftest import ...`` is not safe: an unrelated ``tests`` package
exists in some site-packages trees (Anaconda ships one) and shadows the local
directory. pytest puts this file's directory on ``sys.path``, so
``from _factories import ...`` resolves unambiguously.
"""

from __future__ import annotations

from app.schemas.retrieval import RerankedCandidate, RetrievalCandidate

__all__ = ["make_candidate", "make_reranked"]


def make_candidate(
    chunk_id: str,
    *,
    parent_id: str | None = None,
    text: str = "some passage text",
    language: str = "hi",
    strategy: str = "native",
    fused: float = 0.5,
    content_hash: str = "hash",
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk_id=chunk_id,
        parent_id=parent_id or chunk_id,
        doc_id=parent_id or chunk_id,
        language=language,
        strategy=strategy,
        text=text,
        content_hash=content_hash,
        fused_score=fused,
        retrieved_by=["dense"],
    )


def make_reranked(
    chunk_id: str,
    score: float,
    *,
    parent_id: str | None = None,
    text: str = "passage",
    **kwargs,
) -> RerankedCandidate:
    from app.retrieval.reranker import sigmoid

    base = make_candidate(chunk_id, parent_id=parent_id, text=text, **kwargs)
    return RerankedCandidate(
        **base.model_dump(), rerank_score=score, rerank_prob=sigmoid(score)
    )
