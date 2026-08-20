"""Health and metrics endpoints.

``/health`` reports *actionable* degradation rather than a bare boolean. A demo
box with no Sarvam key and uncalibrated thresholds is a legitimate state, but it
must never be mistaken for a fully configured deployment - so missing secrets and
calibration status are first-class fields in the response.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app import __version__
from app.api.deps import settings_dep
from app.config import Settings, get_thresholds
from app.observability.metrics import METRICS
from app.schemas.common import Stopwatch
from app.schemas.response import ComponentHealth, HealthResponse, MetricsResponse

router = APIRouter(tags=["ops"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(settings_dep)) -> HealthResponse:
    components: list[ComponentHealth] = []
    status: str = "ok"

    # ---- Qdrant ----
    vectors: int | None = None
    try:
        from app.retrieval.store import get_store

        store = get_store()
        with Stopwatch() as sw:
            ok, detail = store.health()
        if ok:
            vectors = store.count()
            if not store.exists():
                ok, detail = False, f"collection '{store.collection}' does not exist - build the index"
            elif vectors == 0:
                ok, detail = False, f"collection '{store.collection}' is empty - build the index"
            else:
                detail = f"{detail}, {vectors} points"
        components.append(
            ComponentHealth(name="qdrant", ok=ok, detail=detail, latency_ms=sw.elapsed_ms)
        )
        if not ok:
            status = "degraded"
    except Exception as exc:  # noqa: BLE001
        components.append(ComponentHealth(name="qdrant", ok=False, detail=str(exc)))
        status = "error"

    # ---- models (reported without forcing a load) ----
    try:
        from app.retrieval.embedder import get_embedder
        from app.retrieval.reranker import get_reranker

        embedder = get_embedder()
        reranker = get_reranker()
        components.append(
            ComponentHealth(
                name="embedder",
                ok=True,
                detail=(
                    f"{embedder.model_name} "
                    f"({'loaded, dim=' + str(embedder.dim) if embedder.is_loaded else 'lazy'})"
                ),
            )
        )
        components.append(
            ComponentHealth(
                name="reranker",
                ok=True,
                detail=f"{reranker.model_name} ({'loaded' if reranker.is_loaded else 'lazy'})",
            )
        )
    except Exception as exc:  # noqa: BLE001
        components.append(ComponentHealth(name="models", ok=False, detail=str(exc)))
        status = "degraded"

    # ---- external services: configuration only, no live calls ----
    components.append(
        ComponentHealth(
            name="groq",
            ok=bool(settings.groq_api_key),
            detail=(
                f"model={settings.groq_model}"
                if settings.groq_api_key
                else "GROQ_API_KEY not set - generation will abstain (fail closed)"
            ),
        )
    )
    components.append(
        ComponentHealth(
            name="sarvam",
            ok=bool(settings.sarvam_api_key),
            detail=(
                f"model={settings.sarvam_stt_model} mode={settings.sarvam_stt_mode}"
                if settings.sarvam_api_key
                else "SARVAM_API_KEY not set - /api/voice unavailable, /api/query works"
            ),
        )
    )

    missing = settings.missing_secrets(require_stt=False)
    if missing and status == "ok":
        status = "degraded"

    thresholds = get_thresholds()
    return HealthResponse(
        status=status,  # type: ignore[arg-type]
        version=__version__,
        corpus_mode=settings.ingest_mode,
        collection=settings.qdrant_collection,
        languages=settings.language_list,
        vectors=vectors,
        device=settings.resolved_device(),
        components=components,
        missing_secrets=missing,
        thresholds_calibrated=bool(thresholds.get("calibrated")),
    )


@router.get("/api/metrics", response_model=MetricsResponse)
def metrics(settings: Settings = Depends(settings_dep)) -> MetricsResponse:
    """In-process latency percentiles and abstention counters.

    Percentiles use nearest-rank over a bounded ring buffer, matching the
    convention in ``scripts/benchmark_latency.py`` so the two are comparable.
    Unmeasured stages are absent rather than reported as zero.
    """
    snapshot = METRICS.snapshot()
    return MetricsResponse(
        uptime_s=snapshot["uptime_s"],
        requests_total=snapshot["requests_total"],
        abstentions_total=snapshot["abstentions_total"],
        abstention_rate=snapshot["abstention_rate"],
        errors_total=snapshot["errors_total"],
        by_stage_latency_ms=snapshot["by_stage_latency_ms"],
        counters=snapshot["counters"],
        abstain_reasons=snapshot["abstain_reasons"],
        extra={
            "corpus_mode": settings.ingest_mode,
            "collection": settings.qdrant_collection,
            "device": settings.resolved_device(),
            "generation_model": settings.groq_model,
            "fusion_mode": settings.retrieval_fusion_mode,
            "thresholds_calibrated": bool(get_thresholds().get("calibrated")),
        },
    )
