"""Text query endpoint and feedback collection."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, Header

from app.api.deps import orchestrator_dep
from app.config import DATA_DIR
from app.languages import normalize_language
from app.observability.tracing import Trace, get_logger
from app.pipeline.orchestrator import RAGOrchestrator
from app.schemas.query import QueryRequest
from app.schemas.response import FeedbackRequest, FeedbackResponse, FinalResponse

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["rag"])


@router.post("/query", response_model=FinalResponse)
async def query(
    request: QueryRequest,
    orchestrator: RAGOrchestrator = Depends(orchestrator_dep),
    x_trace_id: str | None = Header(default=None, alias="X-Trace-Id"),
) -> FinalResponse:
    """Typed text entry point - the debugging path for the RAG pipeline.

    Runs every stage except STT, so retrieval, gating, generation and grounding
    can be exercised and benchmarked without audio or a Sarvam key.
    """
    trace = Trace(x_trace_id)
    language = normalize_language(request.language)
    if request.language and language is None:
        logger.warning("ignoring unrecognised language hint %r", request.language)

    return await orchestrator.run_text(
        request.query,
        language=language,
        top_k=request.top_k,
        include_debug=request.include_debug,
        trace=trace,
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Record thumbs up/down against a trace id.

    Appended to a local JSONL file. Deliberately not a database: the useful
    artefact is the (trace_id, rating, reason) triple joined against the trace
    log, and a file keeps the deployment to a single container.
    """
    target = DATA_DIR / "feedback" / "feedback.jsonl"
    stored = False
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        import time

        with target.open("a", encoding="utf-8") as fh:
            fh.write(
                json.dumps(
                    {
                        "trace_id": request.trace_id,
                        "rating": request.rating,
                        "reason": request.reason,
                        "expected_answer": request.expected_answer,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        stored = True
    except OSError as exc:
        logger.error("could not persist feedback: %s", exc)

    from app.observability.metrics import METRICS

    METRICS.incr(f"feedback_{request.rating}")
    return FeedbackResponse(ok=True, stored=stored, trace_id=request.trace_id)
