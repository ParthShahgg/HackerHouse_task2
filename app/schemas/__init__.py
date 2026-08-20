"""Pydantic models forming the typed boundaries between pipeline stages.

Every stage of the orchestrator consumes one model and produces another. That
makes the state machine in ``app.pipeline`` checkable rather than a bag of
dicts, and it is what lets the debug drawer in the UI render stage-by-stage
provenance without bespoke serialisation.
"""

from app.schemas.common import (
    AbstainReason,
    GateDecision,
    GroundingStatus,
    LatencyBreakdown,
    PipelineStage,
    RetrievalMode,
    SafetyCategory,
    Stopwatch,
    ValidationAction,
)
from app.schemas.generation import GeneratedAnswer, GenerationResult, ValidationResult
from app.schemas.query import GuardrailResult, QueryEmbeddingResult, QueryRequest, STTResult
from app.schemas.response import (
    Citation,
    DebugInfo,
    FeedbackRequest,
    FinalResponse,
    HealthResponse,
)
from app.schemas.retrieval import (
    GroundingDecision,
    ParentContext,
    RerankedCandidate,
    RerankResult,
    RetrievalCandidate,
    RetrievalResult,
)

__all__ = [
    "AbstainReason",
    "Citation",
    "DebugInfo",
    "FeedbackRequest",
    "FinalResponse",
    "GateDecision",
    "GeneratedAnswer",
    "GenerationResult",
    "GroundingDecision",
    "GroundingStatus",
    "GuardrailResult",
    "HealthResponse",
    "LatencyBreakdown",
    "ParentContext",
    "PipelineStage",
    "QueryEmbeddingResult",
    "QueryRequest",
    "RerankResult",
    "RerankedCandidate",
    "RetrievalCandidate",
    "RetrievalMode",
    "RetrievalResult",
    "STTResult",
    "SafetyCategory",
    "Stopwatch",
    "ValidationAction",
    "ValidationResult",
]
