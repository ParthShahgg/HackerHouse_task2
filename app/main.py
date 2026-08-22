"""FastAPI application entry point.

    uvicorn app.main:app --reload

Startup behaviour worth knowing:

* Required secrets are **validated and reported**, not silently ignored. A
  missing ``GROQ_API_KEY`` is logged loudly and surfaced on ``/health``; the
  service still starts so retrieval can be inspected, but generation fails
  closed rather than fabricating answers.
* Models are warmed up in a background thread. Lazy first-inference cost (weight
  materialisation, kernel selection) is ~2-8s on CPU and would otherwise be paid
  by the first real user and pollute the first benchmark sample.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import REPO_ROOT, get_settings, get_thresholds
from app.observability.tracing import get_logger, setup_logging

logger = get_logger(__name__)

FRONTEND_DIR = REPO_ROOT / "frontend"


def _warmup() -> None:
    try:
        from app.pipeline.orchestrator import get_orchestrator

        get_orchestrator().warmup()
        logger.info("warmup complete")
    except Exception as exc:  # noqa: BLE001
        logger.warning("warmup skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()

    logger.info("=" * 72)
    logger.info("Voice RAG over %s  (v%s)", settings.dataset_id, __version__)
    logger.info(
        "mode=%s collection=%s languages=%s",
        settings.ingest_mode, settings.qdrant_collection, ",".join(settings.language_list),
    )
    logger.info(
        "device=%s fp16=%s threads=%s fusion=%s",
        settings.resolved_device(), settings.fp16_enabled(),
        settings.torch_num_threads, settings.retrieval_fusion_mode,
    )
    logger.info("generation=%s", settings.groq_model)

    missing = settings.missing_secrets()
    if missing:
        logger.warning(
            "MISSING SECRETS: %s. Affected stages fail closed (abstain) rather "
            "than degrade silently. See .env.example.",
            ", ".join(missing),
        )
    if not settings.sarvam_api_key:
        logger.warning(
            "SARVAM_API_KEY not set: /api/voice will return 503. "
            "POST /api/query (text) works normally."
        )
    if not get_thresholds().get("calibrated"):
        logger.warning(
            "Abstention thresholds are UNCALIBRATED. Run "
            "scripts/calibrate_thresholds.py; responses report "
            "thresholds_calibrated=false until then."
        )
    logger.info("=" * 72)

    threading.Thread(target=_warmup, name="warmup", daemon=True).start()
    yield

    try:
        from app.retrieval.store import reset_store

        reset_store()
    except Exception:  # noqa: BLE001
        pass
    logger.info("shutdown complete")


app = FastAPI(
    title="Voice-Enabled RAG over MSMARCO-XI",
    description=(
        "Multilingual voice RAG: Sarvam Saaras v3 STT -> input guardrail -> "
        "BGE-M3 hybrid retrieval (dense + sparse, RRF) -> bge-reranker-v2-m3 -> "
        "calibrated abstention gate -> Groq generation -> citation + NLI "
        "grounding validation."
    ),
    version=__version__,
    lifespan=lifespan,
)

_settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api import routes_health, routes_query, routes_voice  # noqa: E402

app.include_router(routes_health.router)
app.include_router(routes_query.router)
app.include_router(routes_voice.router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    """Never leak a stack trace or internal path to a client."""
    logger.exception("unhandled error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "type": type(exc).__name__},
    )


# ---------------------------------------------------------------------------
# Frontend (served from the same origin so the browser needs no CORS or config)
# ---------------------------------------------------------------------------
if FRONTEND_DIR.exists():
    _NO_STORE = {
        # The frontend is edited during development and served from the same
        # origin. A cached app.js paired with a newer index.html produces a page
        # that silently does nothing (listeners never attach, the form falls back
        # to a native GET submit). Never cache these.
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    class _NoCacheStatic(StaticFiles):
        def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: D102
            # Defeat 304 revalidation so a stale script can't be reused.
            return False

        def file_response(self, *args, **kwargs):  # noqa: D102
            response = super().file_response(*args, **kwargs)
            response.headers.update(_NO_STORE)
            return response

    app.mount("/static", _NoCacheStatic(directory=str(FRONTEND_DIR)), name="static")

    def _asset_fingerprint(name: str) -> str:
        """Short content hash of a frontend asset, or 'dev' if unreadable."""
        import hashlib

        try:
            return hashlib.sha256((FRONTEND_DIR / name).read_bytes()).hexdigest()[:12]
        except OSError:
            return "dev"

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        """Serve index.html with content-hashed asset URLs.

        `no-store` stops the browser caching assets *from now on*, but it cannot
        evict a copy cached BEFORE that header existed - and that stale copy is
        reused until the user happens to hard-reload. A corrupted app.js kept
        resurfacing for exactly this reason.

        Rewriting the asset URLs to `app.js?v=<content-hash>` makes the fix
        independent of browser cache behaviour: index.html itself is no-store, so
        it is always re-fetched, and any change to app.js yields a URL the browser
        has never seen and therefore cannot serve from cache.
        """
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        for asset in ("app.js", "styles.css"):
            html = html.replace(
                f"/static/{asset}", f"/static/{asset}?v={_asset_fingerprint(asset)}"
            )
        return HTMLResponse(content=html, headers=_NO_STORE)


def main() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
