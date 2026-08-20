"""Retrieval service: embed -> hybrid retrieve -> rerank -> parent expansion.

Shared by the live orchestrator *and* the evaluation/benchmark scripts. That
sharing is deliberate: if the evaluator used its own retrieval code path, the
reported Recall/MRR figures would describe code that never serves a request.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import get_settings
from app.observability.tracing import get_logger
from app.schemas.common import LatencyBreakdown, RetrievalMode, Stopwatch
from app.schemas.query import QueryEmbeddingResult
from app.schemas.retrieval import RerankedCandidate, RerankResult, RetrievalResult

logger = get_logger(__name__)

__all__ = ["RetrievalService"]


class RetrievalService:
    """Composes the retrieval stages behind one interface."""

    def __init__(
        self,
        *,
        embedder=None,
        retriever=None,
        reranker=None,
        store=None,
        collection: str | None = None,
        settings=None,
    ) -> None:
        from app.retrieval.embedder import get_embedder
        from app.retrieval.hybrid import HybridRetriever
        from app.retrieval.reranker import get_reranker
        from app.retrieval.store import get_store

        self.settings = settings or get_settings()
        self.store = store or get_store()
        self.collection = collection or self.store.collection
        self.embedder = embedder or get_embedder()
        self.retriever = retriever or HybridRetriever(
            store=self.store, collection=self.collection, settings=self.settings
        )
        self._reranker = reranker
        self._reranker_factory = get_reranker
        self._parent_cache: dict[str, str] = {}

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = self._reranker_factory()
        return self._reranker

    # ---------------------------------------------------------------- embedding
    def embed_query(
        self, query: str, latency: LatencyBreakdown | None = None
    ) -> QueryEmbeddingResult:
        lat = latency or LatencyBreakdown()
        with Stopwatch(lat, "query_embedding_latency"):
            dense, sparse = self.embedder.encode_query(query)
        return QueryEmbeddingResult(
            dense=[float(x) for x in dense],
            sparse_indices=list(sparse.keys()),
            sparse_values=[float(v) for v in sparse.values()],
            dim=self.embedder.dim,
            model=self.embedder.model_name,
        )

    # ---------------------------------------------------------------- retrieval
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
        return self.retriever.retrieve(
            embedding,
            languages=languages,
            mode=mode,
            latency=latency,
            limit=limit,
            fusion_mode=fusion_mode,
            strategies=strategies,
            collection=collection or self.collection,
        )

    # ----------------------------------------------------------------- reranking
    def rerank(
        self,
        query: str,
        retrieval: RetrievalResult,
        *,
        latency: LatencyBreakdown | None = None,
        rerank_top_k: int | None = None,
        final_top_k: int | None = None,
        expand_parents: bool | None = None,
        allow_fallback: bool = True,
    ) -> RerankResult:
        """Rerank the fused candidate set and build parent contexts.

        On reranker failure, falls back to the RRF ordering when
        ``allow_fallback`` is set. The fallback is *recorded* (``fallback_used``)
        and propagated to the response, because RRF scores are rank-derived and
        cannot feed the calibrated confidence gate - so a fallback response is
        strictly less trustworthy and must not look identical to a normal one.
        """
        from app.retrieval.parent_expansion import expand_to_parents

        settings = self.settings
        lat = latency or LatencyBreakdown()
        top_k = rerank_top_k or settings.rerank_top_k
        final_k = final_top_k or settings.final_top_k
        expand = settings.enable_parent_expansion if expand_parents is None else expand_parents

        pool = retrieval.candidates[:top_k]
        if not pool:
            return RerankResult(considered=0, model=None)

        try:
            with Stopwatch(lat, "rerank_latency"):
                scores = self.reranker.score(query, [c.text for c in pool])
            model_name = self.reranker.model_name
            fallback = False
            fallback_reason = None
        except Exception as exc:  # noqa: BLE001
            if not allow_fallback:
                raise
            logger.error("reranker failed (%s); falling back to RRF order", exc)
            # Preserve RRF order by synthesising a monotonically decreasing score.
            scores = [-float(i) for i in range(len(pool))]
            model_name = None
            fallback = True
            fallback_reason = f"{type(exc).__name__}: {exc}"

        from app.retrieval.reranker import sigmoid

        scored = [
            RerankedCandidate(
                **candidate.model_dump(),
                rerank_score=float(score),
                rerank_prob=sigmoid(float(score)),
            )
            for candidate, score in zip(pool, scores, strict=True)
        ]
        scored.sort(key=lambda c: (-c.rerank_score, c.chunk_id))
        for rank, candidate in enumerate(scored, start=1):
            candidate.rerank_rank = rank

        contexts = expand_to_parents(
            scored,
            limit=final_k,
            parent_text_lookup=self._lookup_parent_text if expand else None,
            enabled=expand,
        )

        return RerankResult(
            candidates=scored,
            contexts=contexts,
            considered=len(pool),
            fallback_used=fallback,
            fallback_reason=fallback_reason,
            model=model_name,
        )

    # ------------------------------------------------------------------ helpers
    def _lookup_parent_text(self, parent_id: str) -> str | None:
        """Fetch a full parent passage by its native chunk_id.

        Native chunks satisfy ``chunk_id == parent_id``, so the parent passage is
        addressable directly with no extra bookkeeping.
        """
        if parent_id in self._parent_cache:
            return self._parent_cache[parent_id]
        try:
            payloads = self.store.fetch_by_chunk_ids([parent_id], collection=self.collection)
        except Exception as exc:  # noqa: BLE001
            logger.debug("parent fetch failed %s: %s", parent_id, exc)
            return None
        payload = payloads.get(parent_id)
        if not payload:
            return None
        text = payload.get("text")
        if text:
            # Bounded cache; parents are small and this is a hot lookup.
            if len(self._parent_cache) > 4096:
                self._parent_cache.clear()
            self._parent_cache[parent_id] = text
        return text

    def search(
        self,
        query: str,
        *,
        languages: Sequence[str] | None = None,
        mode: RetrievalMode = RetrievalMode.CROSS_LINGUAL,
        latency: LatencyBreakdown | None = None,
        rerank: bool = True,
        limit: int | None = None,
        fusion_mode: str | None = None,
        strategies: Sequence[str] | None = None,
        collection: str | None = None,
        final_top_k: int | None = None,
        expand_parents: bool | None = None,
    ) -> tuple[QueryEmbeddingResult, RetrievalResult, RerankResult | None]:
        """One-shot convenience path used by the evaluation scripts."""
        lat = latency or LatencyBreakdown()
        embedding = self.embed_query(query, lat)
        retrieval = self.retrieve(
            embedding,
            languages=languages,
            mode=mode,
            latency=lat,
            limit=limit,
            fusion_mode=fusion_mode,
            strategies=strategies,
            collection=collection,
        )
        rerank_result = None
        if rerank and retrieval.candidates:
            rerank_result = self.rerank(
                query,
                retrieval,
                latency=lat,
                final_top_k=final_top_k,
                expand_parents=expand_parents,
            )
        return embedding, retrieval, rerank_result

    def warmup(self) -> None:
        """Force model loads and one throwaway query.

        First inference pays lazy-init costs (weight materialisation, kernel
        autotuning) that would otherwise land on a real user and pollute the
        first sample of every benchmark.
        """
        self.embedder.load()
        try:
            self.reranker.load()
        except Exception as exc:  # noqa: BLE001
            logger.warning("reranker warmup failed: %s", exc)
        try:
            embedding = self.embed_query("warmup")
            self.retrieve(embedding, languages=None, limit=4)
            self.reranker.score("warmup", ["warmup passage"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("warmup query skipped: %s", exc)
