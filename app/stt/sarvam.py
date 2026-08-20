"""Sarvam Saaras v3 speech-to-text: streaming WebSocket with REST fallback.

Endpoint choice (this matters and is easy to get wrong)
------------------------------------------------------
``wss://api.sarvam.ai/speech-to-text/ws``
    Saaras v3 streaming transcription. Output stays in the **source language and
    script**, which is what a multilingual Indic corpus needs.

``wss://api.sarvam.ai/speech-to-text-translate/ws``
    *Not used.* It is the translate endpoint - output is English only, which
    would silently break both Indic retrieval and answering in the user's
    language. It also retains ``saaras:v2.5`` for backward compatibility, so
    pointing at it can quietly downgrade the model.

Mode
----
``mode=transcribe`` (default) keeps the original language. ``mode=codemix`` is
available for heavily code-mixed speech and emits English words in Latin script
with Indic words in native script. Configurable via ``SARVAM_STT_MODE``.

Language detection
------------------
``language_code=unknown`` enables auto-detection and makes Sarvam return
``language_probability``. That probability feeds the language-aware retrieval
decision in :func:`app.retrieval.hybrid.decide_languages` - below the confidence
floor we search cross-lingually rather than filtering to a guess.

Fault tolerance
---------------
* Per-close-code handling. ``1006``/``1011``/``1001`` are transient and retried
  with exponential backoff + jitter; ``4xxx`` are application errors (auth,
  quota) and are **not** retried, because retrying a bad key just burns the
  latency budget and rate limit.
* Bounded retries (``SARVAM_MAX_RETRIES``), never unbounded.
* REST fallback for the whole-utterance case when streaming cannot be
  established. Not a silent substitution: ``STTResult.transport`` and
  ``used_fallback`` record what actually happened.

This module deliberately talks the wire protocol via ``websockets``/``httpx``
rather than depending on the ``sarvamai`` SDK: the SDK's ``transcribe()`` helper
hardcodes ``encoding="audio/wav"`` and its JS ``connect()`` silently drops the
``mode`` parameter, and we need raw PCM plus explicit mode control.
"""

from __future__ import annotations

import asyncio
import base64
import json
import random
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.languages import sarvam_to_iso1, script_of
from app.observability.tracing import get_logger
from app.schemas.query import STTResult, TranscriptSegment

logger = get_logger(__name__)

__all__ = [
    "SarvamSTT",
    "SarvamSTTError",
    "SarvamAuthError",
    "detect_code_mixing",
    "wav_header",
]

# WebSocket close codes that justify a retry.
_TRANSIENT_CLOSE_CODES = {1001, 1006, 1011, 1012, 1013, 1014}


class SarvamSTTError(RuntimeError):
    """Transcription failed. Carries whether a retry is sensible."""

    def __init__(self, message: str, *, transient: bool = False, code: int | None = None) -> None:
        super().__init__(message)
        self.transient = transient
        self.code = code


class SarvamAuthError(SarvamSTTError):
    """Missing/invalid credentials or exhausted quota. Never retried."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message, transient=False, code=code)


def wav_header(n_samples_bytes: int, *, sample_rate: int = 16000, channels: int = 1,
               bits: int = 16) -> bytes:
    """Build a 44-byte PCM WAV header.

    Browsers hand us raw PCM or WebM; Sarvam's streaming API accepts only WAV or
    raw PCM. Wrapping PCM in a header locally avoids a transcoding dependency
    (ffmpeg) on the request path.
    """
    import struct

    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    return b"RIFF" + struct.pack("<I", 36 + n_samples_bytes) + b"WAVE" \
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate, byte_rate, block_align, bits) \
        + b"data" + struct.pack("<I", n_samples_bytes)


def detect_code_mixing(text: str, detected_language: str | None) -> bool:
    """Heuristic: does this transcript mix Latin script with an Indic script?

    Cheap and deterministic, run on the transcript rather than the audio. A
    Devanagari utterance containing several Latin-script words is code-mixed
    (very common in Indian speech: "मेरा phone number क्या है").

    Used only to *widen* retrieval to cross-lingual, never to narrow it, so a
    false positive costs a little recall breadth and a false negative simply
    leaves the normal language filter in place.
    """
    if not text:
        return False
    latin = sum(1 for ch in text if "a" <= ch.lower() <= "z")
    indic = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x0D7F or 0x0600 <= ord(ch) <= 0x06FF)
    if not latin or not indic:
        return False
    total = latin + indic
    minority = min(latin, indic) / total
    # Require a real minority presence, not one stray acronym.
    if minority < 0.08:
        return False
    if detected_language and detected_language != "en":
        return True
    return script_of(text) is not None


@dataclass
class _StreamState:
    """Accumulates streaming messages into a final transcript."""

    segments: list[TranscriptSegment]
    language_tag: str | None = None
    language_probability: float | None = None
    partial_count: int = 0
    speech_started: bool = False

    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text).strip()


class SarvamSTT:
    """Client for Sarvam Saaras v3 STT."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        mode: str | None = None,
        ws_url: str | None = None,
        rest_url: str | None = None,
        language_code: str | None = None,
        sample_rate: int | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        enable_rest_fallback: bool | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = api_key if api_key is not None else s.sarvam_api_key
        self.model = model or s.sarvam_stt_model
        self.mode = mode or s.sarvam_stt_mode
        self.ws_url = ws_url or s.sarvam_ws_url
        self.rest_url = rest_url or s.sarvam_rest_url
        self.language_code = language_code or s.sarvam_language_code
        self.sample_rate = sample_rate or s.sarvam_sample_rate
        self.timeout_s = timeout_s or s.sarvam_timeout_s
        self.max_retries = s.sarvam_max_retries if max_retries is None else max_retries
        self.enable_rest_fallback = (
            s.sarvam_enable_rest_fallback if enable_rest_fallback is None else enable_rest_fallback
        )
        self.high_vad_sensitivity = s.sarvam_high_vad_sensitivity
        self.vad_signals = s.sarvam_vad_signals

    # ------------------------------------------------------------------ helpers
    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _require_key(self) -> None:
        if not self.api_key:
            raise SarvamAuthError(
                "SARVAM_API_KEY is not set. The voice path cannot transcribe audio. "
                "Set it in .env (see .env.example) or use POST /api/query with text."
            )

    def _headers(self) -> dict[str, str]:
        return {"api-subscription-key": self.api_key}

    def _ws_query(self, language_code: str | None = None) -> str:
        from urllib.parse import urlencode

        params = {
            "model": self.model,
            "mode": self.mode,
            "language_code": language_code or self.language_code,
            "sample_rate": str(self.sample_rate),
            "high_vad_sensitivity": str(self.high_vad_sensitivity).lower(),
            "vad_signals": str(self.vad_signals).lower(),
            "flush_signal": "true",
        }
        return urlencode(params)

    # ---------------------------------------------------------------- streaming
    async def transcribe_stream(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        language_code: str | None = None,
        is_wav: bool = True,
        on_partial: Callable[[TranscriptSegment], Any] | None = None,
    ) -> STTResult:
        """Transcribe a stream of audio chunks over the streaming WebSocket.

        ``on_partial`` is invoked for each interim transcript so a caller can
        forward live text to the UI before the utterance completes.
        """
        self._require_key()

        attempt = 0
        last_error: Exception | None = None
        # Chunks are buffered so a retry can replay them. Streaming APIs are not
        # resumable mid-utterance, and replaying is the only way a transient drop
        # does not lose the user's speech.
        buffered: list[bytes] = []

        while attempt <= self.max_retries:
            attempt += 1
            try:
                source = self._replay(buffered) if buffered else self._tee(audio_chunks, buffered)
                return await self._stream_once(
                    source,
                    language_code=language_code,
                    is_wav=is_wav,
                    on_partial=on_partial,
                    attempt=attempt,
                )
            except SarvamAuthError:
                raise  # never retry auth/quota
            except SarvamSTTError as exc:
                last_error = exc
                if not exc.transient or attempt > self.max_retries:
                    break
                backoff = min(0.4 * (2 ** (attempt - 1)), 4.0) + random.uniform(0, 0.2)
                logger.warning(
                    "Sarvam stream attempt %d failed (%s); retrying in %.2fs",
                    attempt, exc, backoff,
                )
                await asyncio.sleep(backoff)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                break

        if self.enable_rest_fallback and buffered:
            logger.warning("streaming STT failed (%s); falling back to REST", last_error)
            result = await self.transcribe_rest(
                b"".join(buffered), language_code=language_code, is_wav=is_wav
            )
            result.used_fallback = True
            result.attempts = attempt
            return result

        raise SarvamSTTError(f"streaming transcription failed: {last_error}") from last_error

    @staticmethod
    async def _tee(source: AsyncIterator[bytes], sink: list[bytes]) -> AsyncIterator[bytes]:
        async for chunk in source:
            sink.append(chunk)
            yield chunk

    @staticmethod
    async def _replay(chunks: list[bytes]) -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    async def _stream_once(
        self,
        audio_chunks: AsyncIterator[bytes],
        *,
        language_code: str | None,
        is_wav: bool,
        on_partial: Callable[[TranscriptSegment], Any] | None,
        attempt: int,
    ) -> STTResult:
        import websockets
        from websockets.exceptions import ConnectionClosed, InvalidStatus

        url = f"{self.ws_url}?{self._ws_query(language_code)}"
        state = _StreamState(segments=[])
        encoding = "audio/wav" if is_wav else "pcm_s16le"

        try:
            async with websockets.connect(
                url,
                additional_headers=self._headers(),
                open_timeout=self.timeout_s,
                close_timeout=5,
                max_size=16 * 1024 * 1024,
            ) as ws:

                async def send_audio() -> None:
                    total = 0
                    async for chunk in audio_chunks:
                        if not chunk:
                            continue
                        total += len(chunk)
                        await ws.send(
                            json.dumps(
                                {
                                    "audio": {
                                        "data": base64.b64encode(chunk).decode("ascii"),
                                        "encoding": encoding,
                                        "sample_rate": self.sample_rate,
                                    }
                                }
                            )
                        )
                    # Force the server to emit a final transcript instead of
                    # waiting on VAD silence detection - materially lowers TTFT.
                    await ws.send(json.dumps({"event": "flush"}))
                    logger.debug("sent %d audio bytes + flush", total)

                sender = asyncio.create_task(send_audio())
                try:
                    await asyncio.wait_for(
                        self._consume(ws, state, on_partial, sender), timeout=self.timeout_s
                    )
                finally:
                    if not sender.done():
                        sender.cancel()

        except InvalidStatus as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (401, 403):
                raise SarvamAuthError(f"Sarvam rejected credentials (HTTP {status})", status) from exc
            if status == 429:
                raise SarvamAuthError("Sarvam rate limit / quota exceeded (HTTP 429)", status) from exc
            raise SarvamSTTError(f"handshake failed (HTTP {status})", transient=True) from exc
        except ConnectionClosed as exc:
            code = getattr(exc, "code", None) or getattr(getattr(exc, "rcvd", None), "code", None)
            reason = getattr(getattr(exc, "rcvd", None), "reason", "") or ""
            if code is not None and 4000 <= code <= 4999:
                # Application-specific: read the reason, do not retry blindly.
                raise SarvamAuthError(f"Sarvam closed connection {code}: {reason}", code) from exc
            if state.segments:
                logger.warning("socket closed (%s) but transcript already received", code)
            else:
                raise SarvamSTTError(
                    f"socket closed {code}: {reason}",
                    transient=code in _TRANSIENT_CLOSE_CODES or code is None,
                    code=code,
                ) from exc
        except asyncio.TimeoutError as exc:
            if not state.segments:
                raise SarvamSTTError(
                    f"no transcript within {self.timeout_s}s", transient=True
                ) from exc
        except OSError as exc:
            raise SarvamSTTError(f"network error: {exc}", transient=True) from exc

        return self._finalize(state, transport="websocket", attempts=attempt)

    async def _consume(
        self,
        ws,
        state: _StreamState,
        on_partial: Callable[[TranscriptSegment], Any] | None,
        sender: asyncio.Task,
    ) -> None:
        """Read messages until a final transcript arrives."""
        async for raw in ws:
            message = self._parse(raw)
            if message is None:
                continue
            kind = message.get("type") or message.get("event") or ""
            data = message.get("data") if isinstance(message.get("data"), dict) else message

            if kind in ("events", "speech_start", "speech_end"):
                signal = str(data.get("signal_type", kind)).upper()
                if "START" in signal:
                    state.speech_started = True
                continue

            transcript = (
                data.get("transcript")
                or data.get("translation")
                or data.get("text")
                or ""
            )
            if not transcript:
                continue

            tag = data.get("language_code") or data.get("language")
            if tag:
                state.language_tag = tag
            prob = data.get("language_probability")
            if prob is not None:
                try:
                    state.language_probability = float(prob)
                except (TypeError, ValueError):
                    pass

            is_final = bool(
                data.get("is_final", True)
                if "is_final" in data
                else kind in ("data", "transcript", "translation", "")
            )
            segment = TranscriptSegment(
                text=str(transcript).strip(),
                language=sarvam_to_iso1(tag) if tag else None,
                is_final=is_final,
            )
            if is_final:
                state.segments.append(segment)
                if sender.done():
                    return
                # The utterance is complete; stop reading.
                return
            state.partial_count += 1
            if on_partial is not None:
                try:
                    on_partial(segment)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("on_partial callback raised: %s", exc)

    @staticmethod
    def _parse(raw: Any) -> dict[str, Any] | None:
        if isinstance(raw, bytes):
            try:
                raw = raw.decode("utf-8")
            except UnicodeDecodeError:
                return None
        if not isinstance(raw, str):
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    # --------------------------------------------------------------------- REST
    async def transcribe_rest(
        self,
        audio: bytes,
        *,
        language_code: str | None = None,
        is_wav: bool = True,
        filename: str = "audio.wav",
    ) -> STTResult:
        """Whole-utterance transcription via the REST endpoint.

        Used as the streaming fallback and for benchmarking. Capped at ~30s of
        audio by the service; longer recordings need the Batch API.
        """
        self._require_key()
        import httpx

        if not is_wav:
            audio = wav_header(len(audio), sample_rate=self.sample_rate) + audio

        data: dict[str, str] = {
            "model": self.model,
            "language_code": language_code or self.language_code,
        }
        # `mode` applies to saaras:v3 only.
        if self.model.startswith("saaras:v3"):
            data["mode"] = self.mode

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 2):
            try:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(
                        self.rest_url,
                        headers=self._headers(),
                        files={"file": (filename, audio, "audio/wav")},
                        data=data,
                    )
                if response.status_code in (401, 403):
                    raise SarvamAuthError(
                        f"Sarvam rejected credentials (HTTP {response.status_code})",
                        response.status_code,
                    )
                if response.status_code == 429:
                    raise SarvamAuthError("Sarvam rate limit exceeded (HTTP 429)", 429)
                if response.status_code >= 500:
                    raise SarvamSTTError(
                        f"Sarvam server error {response.status_code}", transient=True
                    )
                response.raise_for_status()
                payload = response.json()
                break
            except SarvamAuthError:
                raise
            except (SarvamSTTError, httpx.HTTPError) as exc:
                last_error = exc
                if attempt > self.max_retries:
                    raise SarvamSTTError(f"REST transcription failed: {exc}") from exc
                await asyncio.sleep(min(0.4 * (2 ** (attempt - 1)), 4.0) + random.uniform(0, 0.2))
        else:  # pragma: no cover
            raise SarvamSTTError(f"REST transcription failed: {last_error}")

        tag = payload.get("language_code")
        state = _StreamState(
            segments=[
                TranscriptSegment(
                    text=str(payload.get("transcript") or "").strip(),
                    language=sarvam_to_iso1(tag) if tag else None,
                    is_final=True,
                )
            ],
            language_tag=tag,
            language_probability=payload.get("language_probability"),
        )
        return self._finalize(state, transport="rest", attempts=1)

    # ----------------------------------------------------------------- finalize
    def _finalize(self, state: _StreamState, *, transport: str, attempts: int) -> STTResult:
        text = state.text()
        iso1 = sarvam_to_iso1(state.language_tag)
        confidence = state.language_probability

        # When Sarvam is pinned to a specific language it omits
        # language_probability. Treating "absent" as 1.0 would be dishonest, but
        # treating it as unknown loses the fact that the caller *specified* the
        # language. Explicit pin => full confidence; auto-detect with no
        # probability => leave None so retrieval stays cross-lingual.
        if confidence is None and self.language_code not in ("unknown", "auto", ""):
            confidence = 1.0
            iso1 = iso1 or sarvam_to_iso1(self.language_code)

        return STTResult(
            transcript=text,
            detected_language=iso1,
            raw_language_tag=state.language_tag,
            language_confidence=confidence,
            is_code_mixed=detect_code_mixing(text, iso1),
            segments=state.segments,
            transport=transport,
            attempts=attempts,
            model=self.model,
            partial_count=state.partial_count,
        )

    # ---------------------------------------------------------------- one-shot
    async def transcribe_bytes(
        self,
        audio: bytes,
        *,
        language_code: str | None = None,
        is_wav: bool = True,
        chunk_size: int = 32 * 1024,
        prefer_streaming: bool = True,
        on_partial: Callable[[TranscriptSegment], Any] | None = None,
    ) -> STTResult:
        """Transcribe a complete recording.

        Prefers the streaming socket (that is the interactive path and the one we
        want exercised in production) and falls back to REST.
        """
        if not prefer_streaming:
            return await self.transcribe_rest(
                audio, language_code=language_code, is_wav=is_wav
            )

        async def chunks() -> AsyncIterator[bytes]:
            for start in range(0, len(audio), chunk_size):
                yield audio[start : start + chunk_size]

        try:
            return await self.transcribe_stream(
                chunks(), language_code=language_code, is_wav=is_wav, on_partial=on_partial
            )
        except SarvamAuthError:
            raise
        except SarvamSTTError as exc:
            if not self.enable_rest_fallback:
                raise
            logger.warning("streaming failed (%s); using REST", exc)
            result = await self.transcribe_rest(
                audio, language_code=language_code, is_wav=is_wav
            )
            result.used_fallback = True
            return result


_STT: SarvamSTT | None = None


def get_stt() -> SarvamSTT:
    global _STT
    if _STT is None:
        _STT = SarvamSTT()
    return _STT


def reset_stt() -> None:
    global _STT
    _STT = None
