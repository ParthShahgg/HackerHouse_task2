"""Formal state machine for the RAG pipeline.

The transition table is data, not control flow buried in ``if`` statements. That
buys three things:

* illegal transitions are caught rather than silently producing a half-built
  response;
* the executed path is recorded per request and surfaced in the debug drawer,
  so "why did this abstain?" is answerable from the trace alone;
* the diagram below is verified by :func:`validate_transition` instead of being
  documentation that drifts.

::

    START
     |
     v
    STT ................ (voice path only)
     |
     v
    INPUT_GUARD --------> ABSTAIN        (unsafe / empty input)
     |
     v
    EMBED
     |
     v
    RETRIEVE -----------> ABSTAIN        (retrieval error / no candidates)
     |
     v
    RERANK
     |
     v
    CONFIDENCE_GATE ----> ABSTAIN        (low score / weak margin)
     |
     v
    GENERATE -----------> ABSTAIN        (unavailable / malformed / refused)
     |
     v
    OUTPUT_VALIDATE
     |-- PASS ---------> DONE
     |-- REGENERATE ---> GENERATE  (once)
     +-- ABSTAIN ------> ABSTAIN
"""

from __future__ import annotations

from app.schemas.common import PipelineStage

__all__ = ["ALLOWED_TRANSITIONS", "TERMINAL_STAGES", "validate_transition", "StateMachineError"]


class StateMachineError(RuntimeError):
    """An illegal stage transition was attempted."""


TERMINAL_STAGES: frozenset[PipelineStage] = frozenset(
    {PipelineStage.DONE, PipelineStage.ERROR}
)

# Every stage may transition to ERROR (unhandled failure) and to ABSTAIN, since
# abstention is a legitimate outcome of nearly every stage.
_UNIVERSAL = {PipelineStage.ERROR, PipelineStage.ABSTAIN}

ALLOWED_TRANSITIONS: dict[PipelineStage, frozenset[PipelineStage]] = {
    PipelineStage.START: frozenset({PipelineStage.STT, PipelineStage.INPUT_GUARD} | _UNIVERSAL),
    PipelineStage.STT: frozenset({PipelineStage.INPUT_GUARD} | _UNIVERSAL),
    PipelineStage.INPUT_GUARD: frozenset({PipelineStage.EMBED} | _UNIVERSAL),
    PipelineStage.EMBED: frozenset({PipelineStage.RETRIEVE} | _UNIVERSAL),
    PipelineStage.RETRIEVE: frozenset({PipelineStage.RERANK} | _UNIVERSAL),
    PipelineStage.RERANK: frozenset({PipelineStage.CONFIDENCE_GATE} | _UNIVERSAL),
    PipelineStage.CONFIDENCE_GATE: frozenset({PipelineStage.GENERATE} | _UNIVERSAL),
    PipelineStage.GENERATE: frozenset({PipelineStage.OUTPUT_VALIDATE} | _UNIVERSAL),
    PipelineStage.OUTPUT_VALIDATE: frozenset(
        {PipelineStage.DONE, PipelineStage.REGENERATE} | _UNIVERSAL
    ),
    # Regeneration re-enters generation exactly once; the orchestrator enforces
    # the count, this table enforces the shape.
    PipelineStage.REGENERATE: frozenset({PipelineStage.GENERATE} | _UNIVERSAL),
    PipelineStage.ABSTAIN: frozenset({PipelineStage.DONE}),
    PipelineStage.DONE: frozenset(),
    PipelineStage.ERROR: frozenset({PipelineStage.DONE}),
}


def validate_transition(current: PipelineStage, nxt: PipelineStage) -> None:
    allowed = ALLOWED_TRANSITIONS.get(current, frozenset())
    if nxt not in allowed:
        raise StateMachineError(
            f"illegal pipeline transition {current.value} -> {nxt.value}; "
            f"allowed: {sorted(s.value for s in allowed)}"
        )
