"""Shared fixtures.

Default `pytest` run must stay green on a machine with no Qdrant, no API keys and
no model weights. Anything needing those is marked (`integration`, `models`,
`external`) and skipped unless the dependency is genuinely available - a skipped
test is honest, a test that silently passes because it stubbed out the thing it
was meant to check is not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Deterministic, offline-safe defaults before app.config is imported.
os.environ.setdefault("INGEST_MODE", "dev")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".hf_cache"))

from _factories import make_reranked  # noqa: E402
from app.indexing.records import Chunk, ParentPassage  # noqa: E402
from app.schemas.retrieval import ParentContext, RerankResult  # noqa: E402


# ---------------------------------------------------------------------------
# Capability detection
# ---------------------------------------------------------------------------
def _qdrant_available() -> bool:
    try:
        import httpx

        from app.config import get_settings

        response = httpx.get(f"{get_settings().qdrant_url}/", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


def _weights_available(repo: str) -> bool:
    """True if the model is already in the local HF cache (no download)."""
    from app.config import REPO_ROOT as ROOT

    slug = "models--" + repo.replace("/", "--")
    root = Path(os.environ.get("HF_HOME", ROOT / ".hf_cache")) / "hub" / slug / "snapshots"
    if not root.exists():
        return False
    return any(
        any(p.suffix in {".bin", ".safetensors"} for p in snap.iterdir())
        for snap in root.iterdir()
        if snap.is_dir()
    )


HAS_QDRANT = _qdrant_available()


def pytest_collection_modifyitems(config, items):
    from app.config import get_settings

    settings = get_settings()
    skip_qdrant = pytest.mark.skip(reason="no Qdrant reachable at QDRANT_URL")
    skip_groq = pytest.mark.skip(reason="GROQ_API_KEY not configured")
    skip_sarvam = pytest.mark.skip(reason="SARVAM_API_KEY not configured")

    for item in items:
        if "integration" in item.keywords and not HAS_QDRANT:
            item.add_marker(skip_qdrant)
        if "models" in item.keywords:
            needed = (settings.embedding_model, settings.reranker_model)
            missing = [m for m in needed if not _weights_available(m)]
            if missing:
                item.add_marker(
                    pytest.mark.skip(reason=f"model weights not cached: {missing}")
                )
        if "external" in item.keywords:
            name = item.nodeid.lower()
            if "groq" in name and not settings.groq_api_key:
                item.add_marker(skip_groq)
            if "sarvam" in name and not settings.sarvam_api_key:
                item.add_marker(skip_sarvam)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def hindi_passage() -> str:
    return (
        "निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करने के लिए "
        "अधिकृत है। यह कानून द्वारा मान्यता प्राप्त है। निगम के मालिक शेयरधारक होते हैं। "
        "निदेशक मंडल कंपनी का संचालन करता है।"
    )


@pytest.fixture
def english_passage() -> str:
    return (
        "A corporation is a company or group of people authorized to act as a "
        "single entity. It is recognised as such in law. Owners of a corporation "
        "are called shareholders. A board of directors runs the company."
    )


@pytest.fixture
def parent(hindi_passage: str) -> ParentPassage:
    from app.indexing.normalize import content_hash, make_doc_id, normalize_text

    text = normalize_text(hindi_passage)
    chash = content_hash("hi", text)
    return ParentPassage(
        doc_id=make_doc_id("hi", chash),
        content_hash=chash,
        language="hi",
        text=text,
        source_split="validation",
    )


@pytest.fixture
def chunker_ctx():
    from app.chunking import ChunkerContext

    return ChunkerContext.from_settings()


@pytest.fixture
def rerank_result_factory():
    def _build(scores: list[float], **kw) -> RerankResult:
        candidates = [
            make_reranked(f"chunk{i}", score, parent_id=f"parent{i}")
            for i, score in enumerate(scores)
        ]
        for rank, candidate in enumerate(candidates, start=1):
            candidate.rerank_rank = rank
        return RerankResult(candidates=candidates, considered=len(candidates), **kw)

    return _build


@pytest.fixture
def contexts() -> list[ParentContext]:
    return [
        ParentContext(
            parent_id="hi:aaaa1111",
            doc_id="hi:aaaa1111",
            language="hi",
            text="निगम एक कंपनी या लोगों का समूह होता है जो एक एकल इकाई के रूप में कार्य करता है।",
            best_score=4.2,
            supporting_chunk_ids=["hi:aaaa1111"],
            strategies=["native"],
            citation_id="hi:aaaa1111",
        ),
        ParentContext(
            parent_id="hi:bbbb2222",
            doc_id="hi:bbbb2222",
            language="hi",
            text="शेयरधारक निगम के मालिक होते हैं।",
            best_score=1.1,
            supporting_chunk_ids=["hi:bbbb2222"],
            strategies=["native"],
            citation_id="hi:bbbb2222",
        ),
    ]


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="hi:abc123#sw0_1",
        doc_id="hi:abc123",
        parent_id="hi:abc123",
        language="hi",
        strategy="sentence_window",
        text="कुछ पाठ",
        content_hash="abc123",
        source_split="validation",
        sentence_start=0,
        sentence_end=1,
        n_chars=7,
    )
