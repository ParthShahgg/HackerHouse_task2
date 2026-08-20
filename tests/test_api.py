"""API contract tests using FastAPI's TestClient with the pipeline stubbed."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.schemas.common import AbstainReason, LatencyBreakdown
from app.schemas.response import Citation, FinalResponse


class StubOrchestrator:
    """Returns a canned FinalResponse so route wiring is tested, not the models."""

    def __init__(self, *, abstain: bool = False):
        self.abstain = abstain
        self.calls: list[dict] = []

    async def run_text(self, query, *, language=None, top_k=None, include_debug=False, trace=None):
        self.calls.append({"kind": "text", "query": query, "language": language})
        return self._response(language, trace, query)

    async def run_voice(self, audio=None, *, audio_stream=None, language=None, is_wav=True,
                        transcript_override=None, top_k=None, include_debug=False,
                        on_partial=None, trace=None):
        self.calls.append({"kind": "voice", "language": language,
                           "override": transcript_override})
        return self._response(language, trace, transcript_override or "spoken query")

    def _response(self, language, trace, transcript):
        latency = LatencyBreakdown(
            query_embedding_latency=5.0, dense_latency=2.0, sparse_latency=3.0,
            rrf_latency=0.5, rerank_latency=120.0, generation_ttft=None,
            total_rag_latency=140.0,
        )
        return FinalResponse(
            answer="अभी पर्याप्त जानकारी नहीं है।" if self.abstain else "निगम एक कंपनी है।",
            language=language or "hi",
            citations=[] if self.abstain else [
                Citation(chunk_id="hi:aaa", score=0.91, language="hi", text="passage")
            ],
            grounded=not self.abstain,
            abstained=self.abstain,
            abstain_reason=AbstainReason.LOW_CONFIDENCE if self.abstain else AbstainReason.NONE,
            latency_ms=latency.api_view(),
            trace_id=(trace.trace_id if trace else "trace123"),
            transcript=transcript,
            detected_language=language or "hi",
            latency_detail=latency,
        )

    def warmup(self):
        pass


@pytest.fixture
def client(monkeypatch):
    """App with lifespan warmup disabled and the orchestrator stubbed."""
    monkeypatch.setattr("app.main._warmup", lambda: None)
    from app.api.deps import orchestrator_dep
    from app.main import app

    stub = StubOrchestrator()
    # Must use dependency_overrides: FastAPI captured the `orchestrator_dep`
    # function object when the route was declared at import time, so
    # monkeypatching the module attribute has no effect on already-built routes.
    app.dependency_overrides[orchestrator_dep] = lambda: stub

    # The WebSocket route resolves the orchestrator by direct call, not Depends.
    monkeypatch.setattr("app.api.routes_voice.orchestrator_dep", lambda: stub)

    try:
        with TestClient(app) as test_client:
            test_client.stub = stub  # type: ignore[attr-defined]
            yield test_client
    finally:
        app.dependency_overrides.clear()


class TestHealth:
    def test_returns_expected_shape(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        for key in ("status", "version", "corpus_mode", "collection", "languages",
                    "device", "components", "missing_secrets", "thresholds_calibrated"):
            assert key in body

    def test_status_is_valid(self, client):
        assert client.get("/health").json()["status"] in ("ok", "degraded", "error")

    def test_never_leaks_secret_values(self, client):
        """Key *names* may appear (missing_secrets); key *values* must not."""
        text = client.get("/health").text
        for prefix in ("gsk_", "sk-", "sk_live", "Bearer "):
            assert prefix not in text
        body = client.get("/health").json()
        for component in body["components"]:
            detail = (component.get("detail") or "").lower()
            assert "gsk_" not in detail

    def test_reports_missing_secrets_explicitly(self, client):
        """A demo box must not look like a configured deployment."""
        assert isinstance(client.get("/health").json()["missing_secrets"], list)


class TestQuery:
    def test_happy_path(self, client):
        response = client.post("/api/query", json={"query": "निगम क्या है?", "language": "hi"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"]
        assert body["language"] == "hi"
        assert body["grounded"] is True
        assert body["abstained"] is False
        assert body["trace_id"]

    def test_response_contains_required_contract(self, client):
        body = client.post("/api/query", json={"query": "q"}).json()
        for key in ("answer", "language", "citations", "grounded", "abstained",
                    "latency_ms", "trace_id"):
            assert key in body
        assert set(body["latency_ms"]) == {"retrieval", "rerank", "generation_ttft", "total"}

    def test_citation_shape(self, client):
        citations = client.post("/api/query", json={"query": "q"}).json()["citations"]
        assert citations
        assert "chunk_id" in citations[0] and "score" in citations[0]

    def test_unmeasured_latency_is_null_not_zero(self, client):
        body = client.post("/api/query", json={"query": "q"}).json()
        assert body["latency_ms"]["generation_ttft"] is None

    def test_empty_query_rejected(self, client):
        assert client.post("/api/query", json={"query": ""}).status_code == 422

    def test_blank_query_rejected(self, client):
        assert client.post("/api/query", json={"query": "   "}).status_code == 422

    def test_missing_query_rejected(self, client):
        assert client.post("/api/query", json={}).status_code == 422

    def test_overlong_query_rejected(self, client):
        assert client.post("/api/query", json={"query": "x" * 5000}).status_code == 422

    def test_unknown_language_hint_is_ignored_not_fatal(self, client):
        response = client.post("/api/query", json={"query": "q", "language": "klingon"})
        assert response.status_code == 200
        assert client.stub.calls[-1]["language"] is None

    def test_language_hint_normalised(self, client):
        client.post("/api/query", json={"query": "q", "language": "hi-IN"})
        assert client.stub.calls[-1]["language"] == "hi"

    def test_trace_id_header_honoured(self, client):
        response = client.post(
            "/api/query", json={"query": "q"}, headers={"X-Trace-Id": "abcdef1234"}
        )
        assert response.json()["trace_id"] == "abcdef1234"

    def test_top_k_bounds(self, client):
        assert client.post("/api/query", json={"query": "q", "top_k": 0}).status_code == 422
        assert client.post("/api/query", json={"query": "q", "top_k": 999}).status_code == 422


class TestVoice:
    def test_transcript_override_path(self, client):
        response = client.post(
            "/api/voice", data={"transcript_override": "निगम क्या है?", "language": "hi"}
        )
        assert response.status_code == 200
        assert response.json()["transcript"] == "निगम क्या है?"

    def test_audio_upload(self, client):
        response = client.post(
            "/api/voice",
            files={"audio": ("a.wav", b"RIFF0000WAVEfmt ", "audio/wav")},
            data={"language": "hi"},
        )
        assert response.status_code == 200

    def test_requires_audio_or_override(self, client):
        assert client.post("/api/voice", data={}).status_code == 422

    def test_empty_audio_rejected(self, client):
        response = client.post("/api/voice", files={"audio": ("a.wav", b"", "audio/wav")})
        assert response.status_code == 422


class TestFeedback:
    def test_accepts_rating(self, client):
        response = client.post(
            "/api/feedback", json={"trace_id": "t1", "rating": "up", "reason": "good"}
        )
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_rejects_invalid_rating(self, client):
        assert client.post(
            "/api/feedback", json={"trace_id": "t1", "rating": "sideways"}
        ).status_code == 422

    def test_requires_trace_id(self, client):
        assert client.post("/api/feedback", json={"rating": "up"}).status_code == 422


class TestMetrics:
    def test_shape(self, client):
        body = client.get("/api/metrics").json()
        for key in ("uptime_s", "requests_total", "abstentions_total",
                    "abstention_rate", "by_stage_latency_ms", "counters"):
            assert key in body

    def test_reports_configuration(self, client):
        extra = client.get("/api/metrics").json()["extra"]
        assert "corpus_mode" in extra and "device" in extra


class TestFrontendAndDocs:
    def test_openapi_available(self, client):
        assert client.get("/openapi.json").status_code == 200

    def test_index_served_when_frontend_present(self, client):
        from app.main import FRONTEND_DIR

        if (FRONTEND_DIR / "index.html").exists():
            response = client.get("/")
            assert response.status_code == 200
            assert "text/html" in response.headers["content-type"]
