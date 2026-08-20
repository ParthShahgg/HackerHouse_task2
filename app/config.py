"""Central configuration.

Layering, lowest precedence first:

1. field defaults below
2. ``configs/<INGEST_MODE>.yaml`` profile (dev | demo | full)
3. ``.env``
4. real process environment

Two rules this module exists to enforce:

* **No magic numbers scattered across modules.** In particular the dense
  embedding dimension is read from the model config at load time (see
  ``app.retrieval.embedder``) and abstention thresholds are read from a
  calibration artefact, not hardcoded at call sites.
* **Corpus size is configuration, never code.** dev/demo/full differ only by
  YAML profile, so scaling the index up requires no source change.
"""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"
REPORTS_DIR = REPO_ROOT / "reports"
DATA_DIR = REPO_ROOT / "data"

# Load .env before anything reads os.environ. Must happen before transformers /
# huggingface_hub are imported anywhere, because those resolve HF_HOME at
# import time and cache it. `bootstrap_hf_env()` below completes the job.
load_dotenv(REPO_ROOT / ".env", override=False)


def bootstrap_hf_env() -> None:
    """Pin HF cache + thread counts *before* torch/transformers get imported.

    The three models (bge-m3, bge-reranker-v2-m3, mDeBERTa-xnli) total roughly
    5GB, which is often more free space than the system drive has. Defaulting
    the cache into the repo (gitignored) keeps that predictable.
    """
    hf_home = os.environ.get("HF_HOME", ".hf_cache")
    resolved = Path(hf_home)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(resolved)
    # Silence the tokenizers fork warning that appears under uvicorn workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


bootstrap_hf_env()


def ensure_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr.

    Required, not cosmetic: on Windows the console defaults to a legacy code page
    (cp1252) and printing Devanagari/Tamil/Telugu raises UnicodeEncodeError. That
    turned a *passing* leakage audit into a crashed one during this build - the
    kind of failure that gets mistaken for a real problem, or silently skipped.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover
                pass


ensure_utf8_stdout()


IngestMode = Literal["dev", "demo", "full"]


class Settings(BaseSettings):
    """Process-wide settings. Instantiate via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- secrets
    sarvam_api_key: str = ""
    groq_api_key: str = ""
    qdrant_api_key: str = ""
    hf_token: str = ""

    # ---------------------------------------------------------------- dataset
    dataset_id: str = "ai4bharat/MSMARCO-XI"
    dataset_split: str = "train"
    # Comma-separated rather than list[str]: pydantic-settings tries json.loads
    # on complex types read from env, so `LANGUAGES=hi,mr` would hard-fail.
    languages: str = "hi,mr,ta,te"
    ingest_mode: IngestMode = "demo"
    max_rows_per_language: int | None = None

    # ----------------------------------------------------------------- qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "msmarco_xi"
    qdrant_local_path: str = ".qdrant_local"
    qdrant_prefer_grpc: bool = False
    # 60s, not 10s. On a CPU-only box the reranker saturates every core in-process,
    # so a co-located Qdrant container can be slow to get scheduled and answer.
    # A 10s ceiling turned that scheduling delay into a hard failure that killed a
    # full evaluation run mid-way.
    qdrant_timeout_s: float = 60.0
    # Bounded retries for transient/connection-level Qdrant faults.
    qdrant_max_retries: int = 3

    # ----------------------------------------------------------------- models
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    nli_model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"

    # Pinned commit SHAs. Not optional pedantry - two concrete reasons:
    #   1. BGE-M3's sparse head (`sparse_linear.pt`) is downloaded separately
    #      from the backbone. If `main` moves between the two fetches, the head
    #      and the encoder come from different revisions, which silently corrupts
    #      the sparse branch. Observed in practice during this build.
    #   2. Reproducible benchmarks. Reported retrieval metrics are meaningless if
    #      the weights can change underneath them.
    # Set to "main" to intentionally track upstream.
    embedding_model_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    reranker_model_revision: str = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    nli_model_revision: str = "b5113eb38ab63efdd7f280f8c144ea8b13f978ce"
    device: str = "auto"
    use_fp16: bool = True
    # Measured on a 12th-gen i5-1240P (4P+8E, 16 logical): reranking 30
    # candidates took 8091ms at 8 threads, 5568ms at 12, 5230ms at 16. Higher is
    # better here up to the logical core count.
    torch_num_threads: int = 16
    hf_home: str = ".hf_cache"

    # int8 dynamic quantization of the RERANKER on CPU.
    # bge-reranker-v2-m3 is XLM-R-large (568M params) and is by far the dominant
    # CPU cost. Measured: 30 candidates 5230ms fp32 -> 4063ms int8 at 16 threads,
    # with ranking preserved (relevant/irrelevant logit margin 11.41 -> 11.06).
    # Ignored on CUDA, where fp16 is the better option.
    #
    # NOTE: the EMBEDDER is deliberately NOT quantized. Index and query vectors
    # must come from the identical model, and quantizing it would either require
    # rebuilding the index or silently mixing precisions.
    quantize_reranker_int8: bool = True

    embed_max_length: int = 512
    embed_batch_size: int = 8
    rerank_max_length: int = 512
    rerank_batch_size: int = 16

    # ------------------------------------------------------------- generation
    groq_model: str = "openai/gpt-oss-20b"
    groq_timeout_s: float = 20.0
    groq_max_retries: int = 1
    groq_temperature: float = 0.0
    groq_max_tokens: int = 320
    # gpt-oss models are reasoning models. Extended reasoning is pure added
    # latency for a two-sentence extractive answer, so keep it minimal.
    groq_reasoning_effort: str = "low"
    # "groq" (production) | "mock" (deterministic extractive TEST DOUBLE).
    # A missing/invalid GROQ_API_KEY does NOT auto-select "mock": that would turn
    # "generation unavailable" into a plausible-looking answer, which is the exact
    # failure mode this system exists to prevent. Unavailable => abstain.
    generation_backend: str = "groq"
    # Forces a specific mock failure for the adversarial demo scenarios:
    # bad_citation | ungrounded | obey_injection
    mock_failure_mode: str = ""

    # -------------------------------------------------------------------- stt
    sarvam_stt_model: str = "saaras:v3"
    # IMPORTANT: the Saaras v3 streaming endpoint is /speech-to-text/ws.
    # /speech-to-text-translate/ws is the translate endpoint and also forces the
    # output to English, which would defeat answering in the user's language.
    sarvam_ws_url: str = "wss://api.sarvam.ai/speech-to-text/ws"
    sarvam_rest_url: str = "https://api.sarvam.ai/speech-to-text"
    # mode=transcribe keeps the transcript in the source language/script, which
    # is what we want for retrieval against an Indic-script corpus.
    # mode=codemix is available for heavily code-mixed speech.
    sarvam_stt_mode: str = "transcribe"
    # "unknown" enables automatic language detection and makes Sarvam return
    # language_probability, which the language-aware retriever needs.
    sarvam_language_code: str = "unknown"
    sarvam_sample_rate: int = 16000
    sarvam_high_vad_sensitivity: bool = True
    sarvam_vad_signals: bool = True
    sarvam_timeout_s: float = 15.0
    sarvam_max_retries: int = 2
    sarvam_enable_rest_fallback: bool = True

    # -------------------------------------------------------------- retrieval
    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_top_k: int = 30
    rerank_top_k: int = 30
    final_top_k: int = 5
    # How dense+sparse get fused:
    #   "server" - one round trip, Qdrant's native RRF via prefetch.
    #   "client" - two concurrent branch queries + RRF here. Costs one extra
    #              round trip but yields per-branch latency and per-branch ranks,
    #              which the required latency report and the debug drawer need.
    # scripts/benchmark_latency.py measures both; see reports/latency.md.
    retrieval_fusion_mode: str = "client"
    rrf_k: int = 60
    enable_parent_expansion: bool = True
    language_filter_confidence: float = 0.65

    # -------------------------------------------------------------- chunking
    # Which representations get written to the live index. Comma-separated
    # subset of: native, sentence_window, semantic_split, fixed_fallback.
    # `native` is always retained (it is the passage-faithful baseline);
    # semantic_split/fixed_fallback only ever fire for long/pathological
    # passages, so listing them is cheap.
    index_strategies: str = "native,sentence_window,semantic_split,fixed_fallback"
    # Passage length (in sentences) above which sentence-window children are
    # emitted. MSMARCO passages are already short retrieval units, so this is
    # deliberately conservative.
    sentence_window_min_sentences: int = 3
    sentence_window_size: int = 2
    sentence_window_stride: int = 1
    # Token budget above which a passage is treated as "unusually long" and
    # routed to semantic splitting.
    semantic_split_min_tokens: int = 320
    semantic_split_percentile: float = 25.0
    # Above this it is pathological; use deterministic fixed chunks.
    fixed_fallback_min_tokens: int = 1024
    fixed_chunk_tokens: int = 256
    fixed_chunk_overlap_ratio: float = 0.175

    # ------------------------------------------------------------- guardrails
    thresholds_path: str = "configs/thresholds.json"
    enable_nli_grounding: bool = True
    nli_entailment_threshold: float = 0.5
    nli_skip_on_deterministic_grounding: bool = True
    enable_deep_safety_check: bool = False

    # ---------------------------------------------------------------- server
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    log_request_bodies: bool = False
    cors_origins: str = "http://localhost:5173,http://localhost:8000"

    # ------------------------------------------------------------- validators
    @field_validator("device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"auto", "cpu", "cuda"}:
            raise ValueError("DEVICE must be one of: auto, cpu, cuda")
        return v

    @field_validator("max_rows_per_language", mode="before")
    @classmethod
    def _blank_to_none(cls, v: Any) -> Any:
        # `MAX_ROWS_PER_LANGUAGE=` in .env arrives as "" and must not become 0.
        if v is None or (isinstance(v, str) and not v.strip()):
            return None
        return v

    # -------------------------------------------------------------- accessors
    @property
    def language_list(self) -> list[str]:
        return [c.strip().lower() for c in self.languages.split(",") if c.strip()]

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def strategy_list(self) -> list[str]:
        items = [s.strip().lower() for s in self.index_strategies.split(",") if s.strip()]
        if "native" not in items:
            items.insert(0, "native")
        return items

    @property
    def fixed_chunk_overlap_tokens(self) -> int:
        return max(1, int(self.fixed_chunk_tokens * self.fixed_chunk_overlap_ratio))

    def resolve_path(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else REPO_ROOT / p

    def resolved_device(self) -> str:
        """Concrete torch device string, honouring DEVICE=auto.

        Imported lazily so that merely reading settings does not drag torch in.
        """
        if self.device != "auto":
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def fp16_enabled(self) -> bool:
        """fp16 only on CUDA. On CPU fp16 matmul is emulated and *slower*."""
        return bool(self.use_fp16) and self.resolved_device() == "cuda"

    def int8_reranker_enabled(self) -> bool:
        """int8 dynamic quantization applies to CPU only."""
        return bool(self.quantize_reranker_int8) and self.resolved_device() == "cpu"

    # ---------------------------------------------------------------- secrets
    def missing_secrets(self, *, require_stt: bool = False) -> list[str]:
        missing: list[str] = []
        if not self.groq_api_key:
            missing.append("GROQ_API_KEY")
        if require_stt and not self.sarvam_api_key:
            missing.append("SARVAM_API_KEY")
        return missing

    def redacted(self) -> dict[str, Any]:
        """Settings dump safe to log or expose on /health."""
        data = self.model_dump()
        for key in ("sarvam_api_key", "groq_api_key", "qdrant_api_key", "hf_token"):
            data[key] = "***set***" if data.get(key) else ""
        return data


def _load_profile(mode: str) -> dict[str, Any]:
    """Read ``configs/<mode>.yaml`` if present."""
    path = CONFIGS_DIR / f"{mode}.yaml"
    if not path.exists():
        return {}
    try:
        import yaml

        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except ModuleNotFoundError:  # PyYAML absent -> profiles simply do not apply
        return {}
    if not isinstance(raw, dict):
        return {}
    # Flatten one nesting level so YAML can be grouped for readability.
    flat: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            flat.update({f"{k}".lower(): v for k, v in value.items()})
        else:
            flat[str(key).lower()] = value
    return flat


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build settings with YAML profile applied *under* env precedence."""
    mode = os.environ.get("INGEST_MODE", "demo").strip().lower() or "demo"
    if mode not in {"dev", "demo", "full"}:
        mode = "demo"

    profile = _load_profile(mode)
    profile["ingest_mode"] = mode

    # Anything explicitly present in the real environment / .env must win over
    # the YAML profile, so drop those keys from the profile overlay.
    env_keys = {k.lower() for k, v in os.environ.items() if str(v).strip() != ""}
    dotenv_path = REPO_ROOT / ".env"
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            if val.strip():
                env_keys.add(key.strip().lower())

    overlay = {k: v for k, v in profile.items() if k not in env_keys or k == "ingest_mode"}
    return Settings(**overlay)


@lru_cache(maxsize=1)
def get_thresholds() -> dict[str, Any]:
    """Load calibrated abstention thresholds.

    Produced by ``scripts/calibrate_thresholds.py``. The fallback values are
    marked ``calibrated: false`` so that every consumer - API responses,
    reports, logs - can tell an empirical threshold from a bootstrap guess. The
    system must never silently present an uncalibrated guess as calibrated.
    """
    settings = get_settings()
    path = settings.resolve_path(settings.thresholds_path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("calibrated", True)
            return data
        except (json.JSONDecodeError, OSError):
            pass
    return {
        "calibrated": False,
        "source": "bootstrap-default (run scripts/calibrate_thresholds.py)",
        "rerank_abstain_below": 0.0,
        "rerank_margin_min": 0.0,
        "rrf_support_min_score": 0.0,
        "notes": (
            "Uncalibrated bootstrap. rerank_abstain_below=0.0 corresponds to "
            "logit 0 / p=0.5 for the bge-reranker-v2-m3 head and is a "
            "placeholder only."
        ),
    }


def reset_settings_cache() -> None:
    """Test hook: drop memoised settings/thresholds."""
    get_settings.cache_clear()
    get_thresholds.cache_clear()
