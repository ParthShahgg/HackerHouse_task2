"""Input and output guardrails.

Input side  : normalisation + tiered safety screening.
Output side : citation validation + multilingual NLI entailment grounding.

Both sides fail closed. A guardrail that cannot run is treated as a failed
guardrail, never as a pass.
"""

from app.guardrails.citation import normalise_citation, validate_citations
from app.guardrails.grounding import OutputGuardrail, validate_output
from app.guardrails.input import InputGuardrail, apply_input_guardrail, blocked_message
from app.guardrails.safety import SafetyVerdict, check_safety, safety_response

__all__ = [
    "InputGuardrail",
    "OutputGuardrail",
    "SafetyVerdict",
    "apply_input_guardrail",
    "blocked_message",
    "check_safety",
    "normalise_citation",
    "safety_response",
    "validate_citations",
    "validate_output",
]
