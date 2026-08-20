"""Output grounding guardrail: citation validation, then entailment.

Ordering is a cost decision. Citation validation is set membership - microseconds -
and catches the most severe failure (fabricated sources). Running it first means a
response with invented citations is rejected without ever loading or invoking the
NLI model.

Escalation policy
-----------------
::

    citations invalid            -> ABSTAIN   (never repaired, never retried)
    factual sentence unsupported -> REGENERATE once, restricted to supported
                                    context; if still unsupported -> ABSTAIN
    all good                     -> PASS

Never fails open. ``UNKNOWN`` on a factual sentence is treated as not-grounded:
"the evidence does not say" is not permission to assert.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import get_settings
from app.guardrails.citation import validate_citations
from app.observability.tracing import get_logger
from app.schemas.common import AbstainReason, GroundingStatus, LatencyBreakdown, Stopwatch, ValidationAction
from app.schemas.generation import GenerationResult, ValidationResult
from app.schemas.retrieval import ParentContext

logger = get_logger(__name__)

__all__ = ["OutputGuardrail", "validate_output"]


class OutputGuardrail:
    def __init__(self, *, enable_nli: bool | None = None) -> None:
        settings = get_settings()
        self.enable_nli = settings.enable_nli_grounding if enable_nli is None else enable_nli

    def validate(
        self,
        generation: GenerationResult,
        contexts: Sequence[ParentContext],
        *,
        latency: LatencyBreakdown | None = None,
        allow_regeneration: bool = True,
    ) -> ValidationResult:
        lat = latency or LatencyBreakdown()
        retrieved_ids = [c.citation_id for c in contexts]

        with Stopwatch(lat, "output_guardrail_latency"):
            # ---- 1. citations ----
            citation_result = validate_citations(
                generation.citations, retrieved_ids, answer=generation.answer
            )
            if citation_result.action != ValidationAction.PASS:
                return citation_result

            # ---- 2. entailment ----
            if not self.enable_nli or not generation.answer.strip():
                citation_result.grounding_status = GroundingStatus.SKIPPED
                return citation_result

            # Verify against the *cited* contexts when the model named them,
            # otherwise against all supplied contexts. Restricting to cited
            # evidence is the stricter test: it checks the answer follows from
            # what the model claimed to use.
            cited = [c for c in contexts if c.citation_id in set(citation_result.valid_citations)]
            pool = cited or list(contexts)

            try:
                from app.guardrails.nli import get_nli_grounder

                grounder = get_nli_grounder()
                with Stopwatch(lat, "nli_latency"):
                    status, sentences = grounder.verify(
                        generation.answer,
                        [c.text for c in pool],
                        context_ids=[c.citation_id for c in pool],
                    )
                nli_model = grounder.model_name
                nli_ran = any(s.method == "nli" for s in sentences)
            except Exception as exc:  # noqa: BLE001
                # Fail closed: an unavailable grounding check must not silently
                # become a pass.
                logger.error("grounding check unavailable (%s); abstaining", exc)
                return ValidationResult(
                    action=ValidationAction.ABSTAIN,
                    reason=AbstainReason.NOT_GROUNDED,
                    explanation=f"Grounding verification unavailable: {exc}",
                    citations_valid=True,
                    valid_citations=citation_result.valid_citations,
                    grounding_status=GroundingStatus.UNKNOWN,
                )

            unsupported = [
                s.sentence
                for s in sentences
                if s.is_factual
                and s.status in (GroundingStatus.NOT_ENTAILED, GroundingStatus.UNKNOWN)
            ]

            if status in (GroundingStatus.ENTAILED, GroundingStatus.SKIPPED):
                citation_result.grounding_status = status
                citation_result.sentence_results = sentences
                citation_result.nli_model = nli_model
                citation_result.nli_ran = nli_ran
                return citation_result

            action = (
                ValidationAction.REGENERATE
                if allow_regeneration and not generation.is_regeneration
                else ValidationAction.ABSTAIN
            )
            logger.warning(
                "grounding failed (%s): %d unsupported sentence(s) -> %s",
                status.value, len(unsupported), action.value,
            )
            return ValidationResult(
                action=action,
                reason=AbstainReason.NOT_GROUNDED,
                explanation=(
                    f"{len(unsupported)} factual statement(s) were not supported by the "
                    f"retrieved evidence (status={status.value})."
                ),
                citations_valid=True,
                valid_citations=citation_result.valid_citations,
                grounding_status=status,
                sentence_results=sentences,
                unsupported_sentences=unsupported,
                nli_model=nli_model,
                nli_ran=nli_ran,
            )


def validate_output(
    generation: GenerationResult,
    contexts: Sequence[ParentContext],
    *,
    latency: LatencyBreakdown | None = None,
    allow_regeneration: bool = True,
) -> ValidationResult:
    return OutputGuardrail().validate(
        generation, contexts, latency=latency, allow_regeneration=allow_regeneration
    )
