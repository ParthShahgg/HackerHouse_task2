"""Trace IDs, stage-path recording and privacy-aware structured logging.

Privacy rules enforced here rather than trusted to call sites:

* API keys are never logged - they are not even attached to trace context.
* Raw microphone audio is never logged; only byte counts and duration.
* User transcripts are redacted unless ``LOG_REQUEST_BODIES=true``.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from typing import Any

from app.config import get_settings
from app.schemas.common import PipelineStage, StageEvent, now_ns, ns_to_ms

__all__ = [
    "Trace",
    "current_trace",
    "get_logger",
    "new_trace_id",
    "setup_logging",
    "redact",
]

_TRACE: contextvars.ContextVar["Trace | None"] = contextvars.ContextVar("trace", default=None)

_SENSITIVE_KEYS = {
    "sarvam_api_key",
    "groq_api_key",
    "qdrant_api_key",
    "hf_token",
    "authorization",
    "api-subscription-key",
    "api_key",
    "token",
    "password",
    "secret",
}


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def redact(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip credentials from a dict before it reaches a log sink."""
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key.lower() in _SENSITIVE_KEYS:
            out[key] = "***"
        elif isinstance(value, dict):
            out[key] = redact(value)
        elif isinstance(value, bytes):
            out[key] = f"<{len(value)} bytes>"
        else:
            out[key] = value
    return out


class Trace:
    """Per-request trace: unique ID, stage path and timed stage events."""

    # `_token` must be declared: __slots__ blocks attribute creation, so the
    # context-manager form (`with trace:`) fails at runtime without it.
    __slots__ = ("trace_id", "events", "stage_path", "warnings", "_t0", "meta", "_token")

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id = trace_id or new_trace_id()
        self.events: list[StageEvent] = []
        self.stage_path: list[PipelineStage] = []
        self.warnings: list[str] = []
        self.meta: dict[str, Any] = {}
        self._t0 = now_ns()

    # -------------------------------------------------------------- recording
    def enter(self, stage: PipelineStage) -> None:
        self.stage_path.append(stage)

    def record(
        self,
        stage: PipelineStage,
        *,
        ok: bool = True,
        duration_ms: float | None = None,
        error: str | None = None,
        **detail: Any,
    ) -> None:
        self.events.append(
            StageEvent(
                stage=stage,
                ok=ok,
                duration_ms=duration_ms,
                error=error,
                detail=redact(detail),
            )
        )

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)

    @property
    def elapsed_ms(self) -> float:
        return ns_to_ms(now_ns() - self._t0)

    def as_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "elapsed_ms": self.elapsed_ms,
            "stage_path": [s.value for s in self.stage_path],
            "warnings": self.warnings,
            "events": [e.model_dump(mode="json") for e in self.events],
        }

    # ------------------------------------------------------------- contextvar
    def bind(self) -> contextvars.Token:
        return _TRACE.set(self)

    @staticmethod
    def unbind(token: contextvars.Token) -> None:
        _TRACE.reset(token)

    def __enter__(self) -> Trace:
        self._token = self.bind()
        return self

    def __exit__(self, *exc: Any) -> None:
        token = getattr(self, "_token", None)
        if token is not None:
            self.unbind(token)
            self._token = None


def current_trace() -> Trace | None:
    return _TRACE.get()


class _TraceFilter(logging.Filter):
    """Injects the active trace ID into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        trace = _TRACE.get()
        record.trace_id = trace.trace_id if trace else "-"
        return True


_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    lvl = (level or settings.log_level).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-7s [%(trace_id)s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    handler.addFilter(_TraceFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, lvl, logging.INFO))

    # These are chatty at INFO and drown out pipeline logs.
    for noisy in ("httpx", "httpcore", "urllib3", "qdrant_client", "websockets", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)


def safe_text(text: str | None, limit: int = 120) -> str:
    """Render user text for logs, honouring LOG_REQUEST_BODIES."""
    if text is None:
        return "-"
    if not get_settings().log_request_bodies:
        return f"<redacted {len(text)} chars>"
    return text[:limit] + ("..." if len(text) > limit else "")
