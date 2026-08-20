"""Shared API dependencies."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.observability.tracing import Trace
from app.pipeline.orchestrator import RAGOrchestrator, get_orchestrator

__all__ = ["orchestrator_dep", "settings_dep", "new_trace"]


def orchestrator_dep() -> RAGOrchestrator:
    return get_orchestrator()


def settings_dep() -> Settings:
    return get_settings()


def new_trace(trace_id: str | None = None) -> Trace:
    return Trace(trace_id)
