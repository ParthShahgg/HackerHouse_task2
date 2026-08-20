"""Pipeline orchestration (state machine over typed stages)."""

from app.pipeline.orchestrator import RAGOrchestrator, get_orchestrator, reset_orchestrator
from app.pipeline.states import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STAGES,
    StateMachineError,
    validate_transition,
)

__all__ = [
    "ALLOWED_TRANSITIONS",
    "RAGOrchestrator",
    "StateMachineError",
    "TERMINAL_STAGES",
    "get_orchestrator",
    "reset_orchestrator",
    "validate_transition",
]
