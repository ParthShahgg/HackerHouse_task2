"""Groq generation with structured output, TTFT measurement and fail-closed errors.

Named ``groq_client`` rather than ``groq``: a module named ``app/generation/groq.py``
can shadow the installed ``groq`` package during intra-package imports, producing a
confusing circular-import failure at ``from groq import AsyncGroq``.


Model choice
------------
Default ``openai/gpt-oss-20b``, configurable via ``GROQ_MODEL``.
``llama-3.1-8b-instant`` and ``llama-3.3-70b-versatile`` are **deprecated** on
Groq and are deliberately not referenced anywhere in this codebase.

Streaming is used even though the response is a JSON object, for one reason:
**time-to-first-token can only be measured by streaming.** A non-streaming call
gives one latency number that conflates queueing, prefill and full decode. TTFT
is a headline requirement here, so it has to be observed rather than inferred.

Structured output
-----------------
Two-step, cheapest first:

1. ``response_format={"type": "json_object"}`` plus the schema in the prompt.
2. On a parse failure, exactly **one** retry with strict JSON instructions.
3. Still unparseable -> **fail closed** (abstain). A malformed model response is
   never forwarded to the user, and never "best-effort" repaired into an answer.

Failure policy
--------------
Every failure mode - no API key, auth rejection, timeout, rate limit, malformed
output - results in an *abstention*, never a fabricated answer. That is the whole
point: when the generator is unavailable the correct behaviour is to say so.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
from collections.abc import Sequence
from typing import Any

from app.config import get_settings
from app.observability.tracing import get_logger
from app.schemas.common import AbstainReason, LatencyBreakdown, now_ns, ns_to_ms
from app.schemas.generation import GeneratedAnswer, GenerationResult
from app.schemas.retrieval import ParentContext

from app.generation.prompts import (
    ANSWER_JSON_SCHEMA,
    INSUFFICIENT_EVIDENCE,
    build_messages,
)

logger = get_logger(__name__)

__all__ = ["GroqGenerator", "get_generator", "reset_generator", "extract_json_object"]

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of one JSON object from model output.

    Handles the two benign deviations small models make - wrapping the object in
    a markdown fence, or emitting a short preamble. It does **not** attempt to
    repair genuinely broken JSON: that path leads to inventing content, so it
    returns ``None`` and lets the caller retry or abstain.
    """
    if not text:
        return None
    candidate = text.strip()

    fenced = _FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    # Balanced-brace scan for the first complete object, respecting strings.
    start = candidate.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(candidate)):
        ch = candidate[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(candidate[start : i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


class GroqGenerator:
    """Async Groq chat-completion client for grounded answering."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = api_key if api_key is not None else s.groq_api_key
        self.model = model or s.groq_model
        self.timeout_s = timeout_s or s.groq_timeout_s
        self.max_retries = s.groq_max_retries if max_retries is None else max_retries
        self.temperature = s.groq_temperature if temperature is None else temperature
        self.max_tokens = max_tokens or s.groq_max_tokens
        self.reasoning_effort = reasoning_effort or getattr(s, "groq_reasoning_effort", "low")
        self._client: Any = None
        self._supports_json_schema: bool | None = None
        self._supports_reasoning_effort: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _get_client(self):
        if self._client is None:
            from groq import AsyncGroq

            self._client = AsyncGroq(api_key=self.api_key, timeout=self.timeout_s)
        return self._client

    # ---------------------------------------------------------------- generation
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
        """Generate a grounded answer. Never raises; failures become abstentions."""
        lat = latency or LatencyBreakdown()

        if not self.configured:
            return GenerationResult(
                ok=False,
                error="GROQ_API_KEY is not configured",
                abstain_reason=AbstainReason.GENERATION_UNAVAILABLE,
                model=self.model,
            )
        if not contexts:
            return GenerationResult(
                ok=False,
                error="no retrieved context supplied to the generator",
                abstain_reason=AbstainReason.NO_CANDIDATES,
                model=self.model,
            )

        attempts = 0
        strict = False
        last_error: str | None = None
        used_strict_retry = False

        # At most 2 parse attempts (normal, then strict JSON), each with bounded
        # transport retries. Deterministic failures are not retried.
        for parse_attempt in range(2):
            messages = build_messages(
                query,
                contexts,
                language=language,
                strict_json=strict,
                supported_only=supported_only,
            )
            try:
                attempts += 1
                raw, ttft_ms, e2e_ms, usage, finish = await self._stream_completion(
                    messages, strict=strict
                )
            except Exception as exc:  # noqa: BLE001
                last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Groq call failed: %s", last_error)
                return GenerationResult(
                    ok=False,
                    error=last_error,
                    abstain_reason=AbstainReason.GENERATION_UNAVAILABLE,
                    model=self.model,
                    attempts=attempts,
                    is_regeneration=is_regeneration,
                )

            # Record timing from the first successful call.
            if lat.generation_ttft is None:
                lat.generation_ttft = ttft_ms
            lat.generation_e2e = e2e_ms

            payload = extract_json_object(raw)
            if payload is None:
                last_error = "model output was not valid JSON"
                logger.warning("%s (attempt %d)", last_error, parse_attempt + 1)
                strict = True
                used_strict_retry = True
                continue

            try:
                parsed = GeneratedAnswer.model_validate(payload)
            except Exception as exc:  # noqa: BLE001
                last_error = f"structured output failed schema validation: {exc}"
                logger.warning("%s (attempt %d)", last_error, parse_attempt + 1)
                strict = True
                used_strict_retry = True
                continue

            answer = parsed.answer.strip()
            # Recognise the abstention sentinel (and defensive variants).
            if not answer or INSUFFICIENT_EVIDENCE in answer.upper():
                return GenerationResult(
                    answer="",
                    citations=[],
                    ok=True,
                    model=self.model,
                    abstain_reason=AbstainReason.MODEL_REFUSED,
                    model_refused=True,
                    attempts=attempts,
                    used_strict_json_retry=used_strict_retry,
                    prompt_tokens=usage.get("prompt_tokens"),
                    completion_tokens=usage.get("completion_tokens"),
                    finish_reason=finish,
                    is_regeneration=is_regeneration,
                )

            return GenerationResult(
                answer=answer,
                citations=[c.strip() for c in parsed.citations if c and c.strip()],
                ok=True,
                model=self.model,
                attempts=attempts,
                used_strict_json_retry=used_strict_retry,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                finish_reason=finish,
                is_regeneration=is_regeneration,
            )

        # Fail closed.
        return GenerationResult(
            ok=False,
            error=last_error or "unparseable model output",
            abstain_reason=AbstainReason.GENERATION_MALFORMED,
            model=self.model,
            attempts=attempts,
            used_strict_json_retry=used_strict_retry,
            is_regeneration=is_regeneration,
        )

    async def _stream_completion(
        self, messages: list[dict[str, str]], *, strict: bool
    ) -> tuple[str, float, float, dict[str, int], str | None]:
        """Stream a completion, returning text, TTFT ms, E2E ms, usage, finish reason."""
        client = self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_completion_tokens": self.max_tokens,
            "stream": True,
        }
        # json_schema when the model supports it (stronger guarantee), else
        # json_object. Probed once, then remembered.
        if self._supports_json_schema is not False:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": ANSWER_JSON_SCHEMA}
        else:
            kwargs["response_format"] = {"type": "json_object"}
        # gpt-oss reasoning models: keep reasoning minimal, it is pure latency here.
        if self._supports_reasoning_effort and "gpt-oss" in self.model:
            kwargs["reasoning_effort"] = self.reasoning_effort

        attempt = 0
        while True:
            attempt += 1
            start = now_ns()
            ttft_ns: int | None = None
            parts: list[str] = []
            usage: dict[str, int] = {}
            finish: str | None = None
            try:
                stream = await client.chat.completions.create(**kwargs)
                async for chunk in stream:
                    choices = getattr(chunk, "choices", None) or []
                    if choices:
                        delta = getattr(choices[0], "delta", None)
                        content = getattr(delta, "content", None) if delta else None
                        if content:
                            if ttft_ns is None:
                                ttft_ns = now_ns()
                            parts.append(content)
                        if getattr(choices[0], "finish_reason", None):
                            finish = choices[0].finish_reason
                    chunk_usage = getattr(chunk, "x_groq", None)
                    if chunk_usage is not None and getattr(chunk_usage, "usage", None):
                        u = chunk_usage.usage
                        usage = {
                            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                        }
                    elif getattr(chunk, "usage", None):
                        u = chunk.usage
                        usage = {
                            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                        }
                end = now_ns()
                text = "".join(parts)
                ttft_ms = ns_to_ms((ttft_ns or end) - start)
                return text, ttft_ms, ns_to_ms(end - start), usage, finish

            except Exception as exc:  # noqa: BLE001
                message = str(exc)
                # Unsupported-parameter probes: drop the feature and retry once
                # each, so a provider change degrades instead of hard-failing.
                if "json_schema" in message and self._supports_json_schema is not False:
                    logger.info("model %s rejected json_schema; using json_object", self.model)
                    self._supports_json_schema = False
                    kwargs["response_format"] = {"type": "json_object"}
                    continue
                if "reasoning_effort" in message and self._supports_reasoning_effort:
                    logger.info("model %s rejected reasoning_effort; dropping it", self.model)
                    self._supports_reasoning_effort = False
                    kwargs.pop("reasoning_effort", None)
                    continue

                transient = any(
                    marker in message.lower()
                    for marker in ("timeout", "timed out", "connection", "429", "500", "502", "503", "504", "rate limit")
                )
                if transient and attempt <= self.max_retries:
                    backoff = min(0.5 * (2 ** (attempt - 1)), 4.0) + random.uniform(0, 0.25)
                    logger.warning("Groq transient error (%s); retrying in %.2fs", exc, backoff)
                    await asyncio.sleep(backoff)
                    continue
                raise

    async def health(self) -> tuple[bool, str]:
        if not self.configured:
            return False, "GROQ_API_KEY not set"
        try:
            client = self._get_client()
            await client.models.list()
            return True, f"ok ({self.model})"
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"


_GENERATOR: GroqGenerator | None = None


def get_generator() -> GroqGenerator:
    global _GENERATOR
    if _GENERATOR is None:
        _GENERATOR = GroqGenerator()
    return _GENERATOR


def reset_generator() -> None:
    global _GENERATOR
    _GENERATOR = None
