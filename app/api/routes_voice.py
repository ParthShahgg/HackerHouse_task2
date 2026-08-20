"""Voice endpoints: multipart upload and a live streaming WebSocket.

Two entry points, because they answer different questions:

``POST /api/voice``
    A complete recording in one request. Simple, easy to curl, and what the
    benchmark harness drives.

``WS /api/voice/stream``
    The genuinely interactive path. The browser streams raw PCM as the user
    speaks; we relay it to Sarvam's streaming socket and push interim transcripts
    back before the utterance ends. This is what makes the transcript appear
    live, and it is the configuration the voice-latency number is measured on.

Audio format note
-----------------
Sarvam's streaming API accepts only WAV or raw PCM - not WebM/Opus, which is
what ``MediaRecorder`` produces by default. The frontend therefore captures via
the Web Audio API and sends 16-bit PCM at 16kHz, avoiding an ffmpeg transcode
step on the request path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from app.api.deps import orchestrator_dep
from app.languages import normalize_language
from app.observability.tracing import Trace, get_logger
from app.pipeline.orchestrator import RAGOrchestrator
from app.schemas.query import TranscriptSegment
from app.schemas.response import FinalResponse
from app.stt.sarvam import SarvamAuthError, SarvamSTTError

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["voice"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024


@router.post("/voice", response_model=FinalResponse)
async def voice(
    audio: UploadFile | None = File(default=None),
    language: str | None = Form(default=None),
    include_debug: bool = Form(default=False),
    is_wav: bool = Form(default=True),
    transcript_override: str | None = Form(default=None),
    orchestrator: RAGOrchestrator = Depends(orchestrator_dep),
) -> FinalResponse:
    """Transcribe an uploaded recording, then run the full RAG pipeline.

    ``transcript_override`` bypasses STT. It exists so the voice path can be
    tested and demoed end-to-end without a Sarvam key, and it is reflected in
    the response as ``transport="injected"`` so a caller can always tell.
    """
    if transcript_override is None and audio is None:
        raise HTTPException(
            status_code=422, detail="provide either an audio file or transcript_override"
        )

    payload = b""
    if audio is not None:
        payload = await audio.read()
        if len(payload) > MAX_AUDIO_BYTES:
            raise HTTPException(status_code=413, detail="audio exceeds 25MB")
        if not payload and transcript_override is None:
            raise HTTPException(status_code=422, detail="audio file was empty")

    try:
        return await orchestrator.run_voice(
            payload,
            language=normalize_language(language),
            is_wav=is_wav,
            transcript_override=transcript_override,
            include_debug=include_debug,
            trace=Trace(),
        )
    except SarvamAuthError as exc:
        # 503, not 500: the service is correctly configured code-wise but a
        # required credential is missing/rejected.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SarvamSTTError as exc:
        raise HTTPException(status_code=502, detail=f"speech recognition failed: {exc}") from exc


@router.websocket("/voice/stream")
async def voice_stream(websocket: WebSocket) -> None:
    """Live microphone streaming.

    Client protocol::

        -> {"event":"config","language":"hi","include_debug":true}
        -> <binary PCM16 frames> ...
        -> {"event":"end"}
        <- {"type":"ready"}
        <- {"type":"partial","text":"..."}          (zero or more)
        <- {"type":"transcript","text":"...","language":"hi"}
        <- {"type":"answer", ...FinalResponse}
        <- {"type":"error","detail":"..."}
    """
    await websocket.accept()
    trace = Trace()
    orchestrator = orchestrator_dep()

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=256)
    config: dict = {}
    loop = asyncio.get_running_loop()

    async def receive_loop() -> None:
        """Pump client frames into the queue until 'end' or disconnect."""
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if (data := message.get("bytes")) is not None:
                    await audio_queue.put(data)
                elif (text := message.get("text")) is not None:
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    event = parsed.get("event")
                    if event == "config":
                        config.update(parsed)
                    elif event in ("end", "flush", "stop"):
                        break
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            await audio_queue.put(None)  # sentinel

    async def audio_iter() -> AsyncIterator[bytes]:
        while True:
            chunk = await audio_queue.get()
            if chunk is None:
                return
            yield chunk

    def on_partial(segment: TranscriptSegment) -> None:
        # Called from the STT consumer; schedule the send on the loop so the
        # callback never blocks socket reading.
        with contextlib.suppress(Exception):
            loop.create_task(
                websocket.send_json({"type": "partial", "text": segment.text})
            )

    receiver = asyncio.create_task(receive_loop())
    try:
        await websocket.send_json({"type": "ready", "trace_id": trace.trace_id})

        # Wait briefly for the config frame so the language hint is honoured.
        for _ in range(50):
            if config:
                break
            await asyncio.sleep(0.01)

        response = await orchestrator.run_voice(
            audio_stream=audio_iter(),
            language=normalize_language(config.get("language")),
            is_wav=bool(config.get("is_wav", False)),
            include_debug=bool(config.get("include_debug", False)),
            on_partial=on_partial,
            trace=trace,
        )

        if response.transcript:
            await websocket.send_json(
                {
                    "type": "transcript",
                    "text": response.transcript,
                    "language": response.detected_language,
                }
            )
        await websocket.send_json(
            {"type": "answer", **response.model_dump(mode="json", exclude_none=True)}
        )
    except SarvamAuthError as exc:
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": str(exc), "code": "stt_auth"})
    except WebSocketDisconnect:
        logger.info("client disconnected during voice stream")
    except Exception as exc:  # noqa: BLE001
        logger.error("voice stream failed: %s: %s", type(exc).__name__, exc)
        with contextlib.suppress(Exception):
            await websocket.send_json({"type": "error", "detail": str(exc)})
    finally:
        receiver.cancel()
        with contextlib.suppress(Exception):
            await websocket.close()
