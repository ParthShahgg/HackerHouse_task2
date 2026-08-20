"""Deterministic extractive generator - a TEST DOUBLE, not a production backend.

Why this exists
---------------
The output guardrails (citation validation, NLI entailment, regenerate-once,
fail-closed abstention) are only observable if *something* produces a candidate
answer. Without a reachable LLM there is no way to demonstrate - or test - that a
fabricated citation is rejected or that an unsupported claim triggers
regeneration.

So this module provides a deterministic extractive stub. It is enabled **only**
by an explicit opt-in::

    GENERATION_BACKEND=mock

The default is ``groq``. This is never selected as a silent fallback when the
Groq key is missing: that would convert "generation unavailable" into a
plausible-looking answer, which is exactly the behaviour the system is designed
to prevent. When Groq is unavailable and the backend is ``groq``, the pipeline
abstains.

Every response is labelled ``model="mock-extractive"`` so it is impossible to
mistake a stub answer for a real generation in logs, API responses or reports.

The ``failure_mode`` hook drives the adversarial demo scenarios:

``bad_citation``
    Emit a citation id that was never retrieved -> citation guardrail must reject.
``ungrounded``
    Assert a fact absent from the evidence -> NLI guardrail must reject.
``obey_injection``
    Follow an instruction embedded in a retrieved passage -> proves the
    injection defence is what stops it, not the generator's good manners.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.config import get_settings
from app.indexing.normalize import split_sentences
from app.observability.tracing import get_logger
from app.schemas.common import AbstainReason, LatencyBreakdown, now_ns, ns_to_ms
from app.schemas.generation import GenerationResult
from app.schemas.retrieval import ParentContext

logger = get_logger(__name__)

__all__ = ["MockGenerator"]

MOCK_MODEL_NAME = "mock-extractive"


class MockGenerator:
    """Extractive stub: answers with the most query-relevant sentence of the top context."""

    def __init__(self, *, failure_mode: str | None = None, max_sentences: int = 2) -> None:
        self.model = MOCK_MODEL_NAME
        self.failure_mode = failure_mode
        self.max_sentences = max_sentences

    @property
    def configured(self) -> bool:
        return True

    async def generate(
        self,
        query: str,
        contexts: Sequence[ParentContext],
        *,
        language: str | None = None,
        latency: LatencyBreakdown | None = None,
        supported_only: bool = False,
        is_regeneration: bool = False,
    ) -> GenerationResult:
        lat = latency or LatencyBreakdown()
        start = now_ns()

        if not contexts:
            return GenerationResult(
                ok=False,
                error="no context",
                abstain_reason=AbstainReason.NO_CANDIDATES,
                model=self.model,
            )

        top = contexts[0]

        if self.failure_mode == "bad_citation":
            answer = self._extract(query, top)
            citations = ["hi:deadbeefdeadbeef"]  # never retrieved
        elif self.failure_mode == "ungrounded":
            # A claim with no support anywhere in the evidence.
            answer = (
                "The Eiffel Tower was relocated to Mumbai in 1998 and is now "
                "the tallest structure in South Asia."
            )
            citations = [top.citation_id]
        elif self.failure_mode == "obey_injection":
            answer = "PWNED"
            citations = [top.citation_id]
        elif is_regeneration:
            # Regeneration is deliberately more conservative: quote verbatim, so
            # the deterministic span check can ground it.
            answer = self._extract(query, top, sentences=1)
            citations = [top.citation_id]
        else:
            answer = self._extract(query, top)
            citations = [top.citation_id]

        elapsed = ns_to_ms(now_ns() - start)
        if lat.generation_ttft is None:
            lat.generation_ttft = elapsed
        lat.generation_e2e = elapsed

        return GenerationResult(
            answer=answer,
            citations=citations,
            ok=True,
            model=self.model,
            attempts=1,
            is_regeneration=is_regeneration,
        )

    def _extract(self, query: str, context: ParentContext, sentences: int | None = None) -> str:
        """Pick the sentences of the context with the most query-term overlap.

        Verbatim extraction, so the answer is grounded by construction - which is
        what makes it useful for exercising the *rest* of the pipeline.
        """
        limit = sentences or self.max_sentences
        sents = split_sentences(context.text) or [context.text]
        query_terms = {t for t in query.lower().split() if len(t) > 2}

        def overlap(sentence: str) -> int:
            words = {w.strip(".,;:!?()।॥") for w in sentence.lower().split()}
            return len(query_terms & words)

        ranked = sorted(range(len(sents)), key=lambda i: (-overlap(sents[i]), i))
        chosen = sorted(ranked[:limit])
        return " ".join(sents[i] for i in chosen).strip()

    async def health(self) -> tuple[bool, str]:
        return True, "mock generator (TEST DOUBLE - not for production)"


def build_generator():
    """Factory honouring ``GENERATION_BACKEND``.

    ``groq`` (default) or ``mock``. A missing Groq key does NOT silently select
    the mock backend.
    """
    settings = get_settings()
    backend = (getattr(settings, "generation_backend", "groq") or "groq").lower()
    if backend == "mock":
        logger.warning(
            "GENERATION_BACKEND=mock - using the deterministic TEST DOUBLE. "
            "Answers are extractive quotes, not model output. Never use in production."
        )
        return MockGenerator(failure_mode=getattr(settings, "mock_failure_mode", None) or None)

    from app.generation.groq_client import GroqGenerator

    return GroqGenerator()
