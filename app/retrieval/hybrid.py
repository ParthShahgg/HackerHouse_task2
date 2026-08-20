"""Hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion.

Why RRF instead of a weighted score blend
-----------------------------------------
Cosine similarity (roughly 0.3-0.95, tightly clustered) and BGE-M3 lexical
overlap scores (unbounded, magnitude depends on query length and how many terms
matched) live on incomparable scales. ``alpha * dense + (1-alpha) * sparse``
therefore silently lets whichever branch happens to have the larger numeric range
dominate, and the "optimal" alpha shifts per query and per language. RRF uses only
*ranks*::

    RRF(d) = sum over branches of  1 / (k + rank_b(d))

so it needs no calibration, no per-language tuning, and is robust to one branch
producing pathological magnitudes. ``k = 60`` is the standard constant from
Cormack et al.; it damps the influence of the very top ranks just enough that a
document has to do well in *both* branches to win.

Two fusion implementations
--------------------------
``server``
    One round trip using Qdrant's native ``FusionQuery(RRF)`` over two
    prefetches. Fewest round trips.
``client``
    Two branch queries issued **concurrently**, fused here. Costs one extra
    round trip but yields per-branch latency and per-branch ranks - both required
    for the latency report and the debug drawer.

Because the branches run concurrently, client-mode wall clock is
``max(dense, sparse)`` plus fusion, not their sum. Both modes are benchmarked in
``reports/latency.md``; the default is chosen from those measurements rather than
by assumption.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.observability.tracing import get_logger
from app.schemas.common import LatencyBreakdown, RetrievalMode, Stopwatch
from app.schemas.query import QueryEmbeddingResult
from app.schemas.retrieval import RetrievalCandidate, RetrievalResult

logger = get_logger(__name__)

__all__ = ["HybridRetriever", "reciprocal_rank_fusion", "decide_languages", "BranchHit"]


@dataclass
class BranchHit:
    """One hit from a single retrieval branch."""

    chunk_id: str
    payload: dict[str, Any]
    score: float
    rank: int


def reciprocal_rank_fusion(
    branches: dict[str, Sequence[BranchHit]],
    *,
    k: int = 60,
    limit: int | None = None,
) -> list[tuple[str, float, dict[str, BranchHit]]]:
    """Fuse ranked branch results.

    Returns ``(chunk_id, rrf_score, {branch: hit})`` sorted best-first. Ties are
    broken deterministically by chunk_id so results are reproducible across runs
    - important because non-deterministic ordering would make benchmark
    percentiles jitter for reasons unrelated to latency.
    """
    scores: dict[str, float] = {}
    provenance: dict[str, dict[str, BranchHit]] = {}

    for branch_name, hits in branches.items():
        for hit in hits:
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + hit.rank)
            provenance.setdefault(hit.chunk_id, {})[branch_name] = hit

    fused = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        fused = fused[:limit]
    return [(cid, score, provenance[cid]) for cid, score in fused]


def decide_languages(
    *,
    detected_language: str | None,
    confidence: float | None,
    is_code_mixed: bool,
    configured: Sequence[str],
    min_confidence: float,
) -> tuple[list[str], RetrievalMode]:
    """Choose the language filter for this query.

    Policy:

    * **Code-mixed** speech -> never filter. A Hindi-English utterance may have
      its answer in either namespace, and filtering to one would silently make
      the correct passage unreachable.
    * **Confident detection** of a language we actually indexed -> filter to it.
      This is both a quality win (no cross-language distractors) and a latency
      win (smaller HNSW candidate set).
    * **Uncertain or unknown** -> search everything cross-lingually. BGE-M3 is
      multilingual, so this degrades recall gracefully instead of guessing.

    Note there is no translation step. Translating the query would add a model
    hop to the critical path; BGE-M3 embeds Hindi and English into a shared
    space already, so translation would have to *prove* a retrieval gain to earn
    its latency. It has not, so it is not implemented.
    """
    pool = [lang for lang in configured if lang]
    if is_code_mixed:
        return pool, RetrievalMode.CODE_MIXED_CROSS_LINGUAL
    if (
        detected_language
        and detected_language in pool
        and confidence is not None
        and confidence >= min_confidence
    ):
        return [detected_language], RetrievalMode.LANGUAGE_FILTERED
    return pool, RetrievalMode.CROSS_LINGUAL


class HybridRetriever:
    """Dense + sparse retrieval against one Qdrant collection."""

    def __init__(
        self,
        store=None,
        *,
        collection: str | None = None,
        settings=None,
    ) -> None:
        from app.retrieval.store import get_store

        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.collection = collection or self.store.collection
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval")

    # ------------------------------------------------------------------ filters
    def build_filter(self, languages: Sequence[str] | None, strategies: Sequence[str] | None = None):
        from qdrant_client import models

        conditions = []
        if languages:
            conditions.append(
                models.FieldCondition(
                    key="language", match=models.MatchAny(any=list(languages))
                )
            )
        if strategies:
            conditions.append(
                models.FieldCondition(
                    key="strategy", match=models.MatchAny(any=list(strategies))
                )
            )
        return models.Filter(must=conditions) if conditions else None

    # ----------------------------------------------------------------- branches
    def search_dense(
        self,
        dense: Sequence[float],
        *,
        limit: int,
        query_filter=None,
        collection: str | None = None,
    ) -> list[BranchHit]:
        from app.retrieval.store import DENSE_VECTOR

        response = self.store.client.query_points(
            collection_name=collection or self.collection,
            query=list(dense),
            using=DENSE_VECTOR,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            BranchHit(
                chunk_id=point.payload.get("chunk_id", str(point.id)),
                payload=point.payload or {},
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    def search_sparse(
        self,
        sparse: dict[int, float],
        *,
        limit: int,
        query_filter=None,
        collection: str | None = None,
    ) -> list[BranchHit]:
        from qdrant_client import models

        from app.retrieval.store import SPARSE_VECTOR

        if not sparse:
            return []
        response = self.store.client.query_points(
            collection_name=collection or self.collection,
            query=models.SparseVector(
                indices=list(sparse.keys()), values=[float(v) for v in sparse.values()]
            ),
            using=SPARSE_VECTOR,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            BranchHit(
                chunk_id=point.payload.get("chunk_id", str(point.id)),
                payload=point.payload or {},
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    def search_server_fusion(
        self,
        embedding: QueryEmbeddingResult,
        *,
        limit: int,
        query_filter=None,
        collection: str | None = None,
    ) -> list[BranchHit]:
        """Qdrant-native RRF over dense + sparse prefetches (one round trip)."""
        from qdrant_client import models

        from app.retrieval.store import DENSE_VECTOR, SPARSE_VECTOR

        prefetch = [
            models.Prefetch(
                query=list(embedding.dense),
                using=DENSE_VECTOR,
                limit=self.settings.dense_top_k,
                filter=query_filter,
            )
        ]
        if embedding.sparse_indices:
            prefetch.append(
                models.Prefetch(
                    query=models.SparseVector(
                        indices=embedding.sparse_indices,
                        values=[float(v) for v in embedding.sparse_values],
                    ),
                    using=SPARSE_VECTOR,
                    limit=self.settings.sparse_top_k,
                    filter=query_filter,
                )
            )
        response = self.store.client.query_points(
            collection_name=collection or self.collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            BranchHit(
                chunk_id=point.payload.get("chunk_id", str(point.id)),
                payload=point.payload or {},
                score=float(point.score),
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    # ----------------------------------------------------------------- retrieve
    def retrieve(
        self,
        embedding: QueryEmbeddingResult,
        *,
        languages: Sequence[str] | None = None,
        mode: RetrievalMode = RetrievalMode.CROSS_LINGUAL,
        latency: LatencyBreakdown | None = None,
        limit: int | None = None,
        fusion_mode: str | None = None,
        strategies: Sequence[str] | None = None,
        collection: str | None = None,
    ) -> RetrievalResult:
        """Run hybrid retrieval and return fused candidates."""
        settings = self.settings
        limit = limit or settings.rrf_top_k
        fusion = (fusion_mode or settings.retrieval_fusion_mode).lower()
        query_filter = self.build_filter(languages, strategies)
        lat = latency or LatencyBreakdown()

        try:
            if fusion == "server":
                with Stopwatch(lat, "rrf_latency"):
                    fused_hits = self.search_server_fusion(
                        embedding, limit=limit, query_filter=query_filter, collection=collection
                    )
                candidates = [
                    self._to_candidate(hit, fused_score=hit.score, branches={"fused": hit})
                    for hit in fused_hits
                ]
                return RetrievalResult(
                    candidates=candidates,
                    mode=mode,
                    languages_searched=list(languages or []),
                    fused_count=len(candidates),
                    used_server_side_fusion=True,
                )

            # ---- client-side fusion, branches concurrent ----
            dense_sw = Stopwatch()
            sparse_sw = Stopwatch()

            def run_dense() -> list[BranchHit]:
                with dense_sw:
                    return self.search_dense(
                        embedding.dense,
                        limit=settings.dense_top_k,
                        query_filter=query_filter,
                        collection=collection,
                    )

            def run_sparse() -> list[BranchHit]:
                with sparse_sw:
                    return self.search_sparse(
                        embedding.sparse_dict(),
                        limit=settings.sparse_top_k,
                        query_filter=query_filter,
                        collection=collection,
                    )

            dense_future = self._pool.submit(run_dense)
            sparse_future = self._pool.submit(run_sparse)
            dense_hits = dense_future.result()
            sparse_hits = sparse_future.result()

            lat.dense_latency = dense_sw.elapsed_ms
            lat.sparse_latency = sparse_sw.elapsed_ms

            with Stopwatch(lat, "rrf_latency"):
                fused = reciprocal_rank_fusion(
                    {"dense": dense_hits, "sparse": sparse_hits},
                    k=settings.rrf_k,
                    limit=limit,
                )

            candidates = [
                self._to_candidate(
                    next(iter(branches.values())), fused_score=score, branches=branches
                )
                for _cid, score, branches in fused
            ]
            return RetrievalResult(
                candidates=candidates,
                mode=mode,
                languages_searched=list(languages or []),
                dense_count=len(dense_hits),
                sparse_count=len(sparse_hits),
                fused_count=len(candidates),
                used_server_side_fusion=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("retrieval failed: %s: %s", type(exc).__name__, exc)
            return RetrievalResult(
                candidates=[],
                mode=mode,
                languages_searched=list(languages or []),
                degraded=True,
                degraded_reason=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    def _to_candidate(
        hit: BranchHit, *, fused_score: float, branches: dict[str, BranchHit]
    ) -> RetrievalCandidate:
        payload = hit.payload
        dense_hit = branches.get("dense")
        sparse_hit = branches.get("sparse")
        return RetrievalCandidate(
            chunk_id=payload.get("chunk_id", hit.chunk_id),
            parent_id=payload.get("parent_id", payload.get("doc_id", hit.chunk_id)),
            doc_id=payload.get("doc_id", ""),
            language=payload.get("language", ""),
            strategy=payload.get("strategy", "native"),
            text=payload.get("text", ""),
            source_split=payload.get("source_split"),
            content_hash=payload.get("content_hash"),
            sentence_start=payload.get("sentence_start"),
            sentence_end=payload.get("sentence_end"),
            dense_score=dense_hit.score if dense_hit else None,
            sparse_score=sparse_hit.score if sparse_hit else None,
            dense_rank=dense_hit.rank if dense_hit else None,
            sparse_rank=sparse_hit.rank if sparse_hit else None,
            fused_score=fused_score,
            retrieved_by=[name for name in branches if name != "fused"] or ["fused"],
        )

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
