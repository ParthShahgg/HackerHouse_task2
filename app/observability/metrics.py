"""In-process metrics registry backing ``GET /api/metrics``.

Deliberately dependency-free (no Prometheus client): the deployment target is a
single container and the requirement is observability of stage latencies and
abstention behaviour, not a full TSDB. Percentiles are computed over a bounded
ring buffer so memory stays flat under load.

Percentile convention matches the benchmark harness: *nearest-rank on sorted
samples*, so P100 is the true observed maximum rather than an interpolation.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Any

from app.schemas.common import LatencyBreakdown

__all__ = ["MetricsRegistry", "METRICS", "percentile"]

_MAX_SAMPLES = 2000

_STAGE_FIELDS = (
    "stt_latency",
    "guardrail_latency",
    "query_embedding_latency",
    "dense_latency",
    "sparse_latency",
    "rrf_latency",
    "rerank_latency",
    "grounding_gate_latency",
    "generation_ttft",
    "generation_e2e",
    "output_guardrail_latency",
    "nli_latency",
    "total_rag_latency",
    "total_voice_latency",
    "total_completion_latency",
)


def percentile(sorted_values: list[float], pct: float) -> float:
    """Nearest-rank percentile. ``pct`` in [0, 100]. Assumes sorted input."""
    if not sorted_values:
        return float("nan")
    if pct <= 0:
        return sorted_values[0]
    if pct >= 100:
        return sorted_values[-1]
    # ceil(pct/100 * n) - 1, clamped.
    n = len(sorted_values)
    rank = int(-(-(pct / 100.0 * n) // 1)) - 1
    return sorted_values[max(0, min(n - 1, rank))]


class MetricsRegistry:
    """Thread-safe counters + latency samples."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = time.time()
        self._counters: dict[str, int] = defaultdict(int)
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_SAMPLES)
        )
        self._abstain_reasons: dict[str, int] = defaultdict(int)

    # ---------------------------------------------------------------- counters
    def incr(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value_ms: float) -> None:
        with self._lock:
            self._samples[name].append(value_ms)

    def record_latency(self, breakdown: LatencyBreakdown) -> None:
        """Record every *measured* stage. ``None`` stages are skipped so that
        unmeasured stages never contaminate percentiles with zeros."""
        with self._lock:
            for field in _STAGE_FIELDS:
                value = getattr(breakdown, field, None)
                if value is not None:
                    self._samples[field].append(float(value))

    def record_request(
        self,
        *,
        abstained: bool,
        grounded: bool,
        error: bool = False,
        abstain_reason: str | None = None,
        voice: bool = False,
    ) -> None:
        with self._lock:
            self._counters["requests_total"] += 1
            if voice:
                self._counters["voice_requests_total"] += 1
            if abstained:
                self._counters["abstentions_total"] += 1
            if grounded:
                self._counters["grounded_total"] += 1
            if error:
                self._counters["errors_total"] += 1
            if abstain_reason and abstain_reason != "none":
                self._abstain_reasons[abstain_reason] += 1

    # ----------------------------------------------------------------- readers
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            reasons = dict(self._abstain_reasons)
            samples = {k: sorted(v) for k, v in self._samples.items() if v}

        stage_latency: dict[str, dict[str, float]] = {}
        for name, values in samples.items():
            stage_latency[name] = {
                "count": float(len(values)),
                "p50": round(percentile(values, 50), 3),
                "p70": round(percentile(values, 70), 3),
                "p95": round(percentile(values, 95), 3),
                "p100": round(percentile(values, 100), 3),
                "mean": round(sum(values) / len(values), 3),
            }

        total = counters.get("requests_total", 0)
        abstentions = counters.get("abstentions_total", 0)
        return {
            "uptime_s": round(time.time() - self._started, 2),
            "requests_total": total,
            "abstentions_total": abstentions,
            "abstention_rate": round(abstentions / total, 4) if total else 0.0,
            "errors_total": counters.get("errors_total", 0),
            "by_stage_latency_ms": stage_latency,
            "counters": counters,
            "abstain_reasons": reasons,
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._samples.clear()
            self._abstain_reasons.clear()
            self._started = time.time()


METRICS = MetricsRegistry()
