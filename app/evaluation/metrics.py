"""Ranking metrics for retrieval evaluation.

Evaluation unit: the **passage content hash**, not the chunk id
---------------------------------------------------------------
A passage may be indexed as a native chunk plus several sentence-window children,
all sharing one ``content_hash``. If ranks were computed over chunk ids, a single
passage could occupy ranks 1-4 and Recall@5 would be measuring how fragmented the
passage is rather than whether the right passage was found. So every ranked list
is first collapsed to unique content hashes, keeping each hash's best rank.

This also makes the chunking strategies directly comparable: an arm that emits 6
vectors per passage is scored on the same footing as native, which emits 1.

Ground truth comes from MSMARCO-XI's own ``is_selected`` labels, mapped to
content hashes offline (see :mod:`app.indexing.corpus`). Labels never touch the
live index.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

__all__ = [
    "RankingMetrics",
    "collapse_to_unique",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "evaluate_ranking",
    "MetricAccumulator",
]

DEFAULT_KS: tuple[int, ...] = (1, 3, 5, 10)


def collapse_to_unique(ranked: Iterable[str]) -> list[str]:
    """Deduplicate a ranked list, keeping first (best) occurrence order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in ranked:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant items appearing in the top ``k``.

    Denominator is ``|relevant|`` (standard Recall), so a query with 2 relevant
    passages of which 1 is found at rank 1 scores 0.5 at k=1 - not 1.0. Using
    ``min(|R|, k)`` instead would inflate Recall@1 whenever multiple passages are
    relevant.
    """
    if not relevant:
        return float("nan")
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return float("nan")
    top = ranked[:k]
    if not top:
        return 0.0
    return sum(1 for item in top if item in relevant) / len(top)


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant item; 0 if none retrieved."""
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Binary-gain nDCG@k.

    Binary gains because ``is_selected`` is binary; there are no graded judgements
    in MSMARCO-XI to exploit.
    """
    if not relevant:
        return float("nan")
    dcg = 0.0
    for index, item in enumerate(ranked[:k], start=1):
        if item in relevant:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


@dataclass
class RankingMetrics:
    """Aggregated metrics for one evaluation arm."""

    arm: str = ""
    queries: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg10: float = 0.0
    # Latency of the arm itself, so quality and cost are reported together.
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    mean_latency_ms: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    def to_row(self) -> dict[str, object]:
        row: dict[str, object] = {"arm": self.arm, "queries": self.queries}
        for k in sorted(self.recall):
            row[f"recall@{k}"] = round(self.recall[k], 4)
        row["mrr"] = round(self.mrr, 4)
        row["ndcg@10"] = round(self.ndcg10, 4)
        if self.mean_latency_ms is not None:
            row["latency_mean_ms"] = round(self.mean_latency_ms, 2)
        if self.latency_p50_ms is not None:
            row["latency_p50_ms"] = round(self.latency_p50_ms, 2)
        if self.latency_p95_ms is not None:
            row["latency_p95_ms"] = round(self.latency_p95_ms, 2)
        for key, value in self.extra.items():
            row[key] = round(value, 4) if isinstance(value, float) else value
        return row


def evaluate_ranking(
    ranked: Sequence[str], relevant: set[str], ks: Sequence[int] = DEFAULT_KS
) -> dict[str, float]:
    """Per-query metrics for one ranked list."""
    unique = collapse_to_unique(ranked)
    result: dict[str, float] = {}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(unique, relevant, k)
        result[f"precision@{k}"] = precision_at_k(unique, relevant, k)
    result["mrr"] = reciprocal_rank(unique, relevant)
    result["ndcg@10"] = ndcg_at_k(unique, relevant, 10)
    return result


class MetricAccumulator:
    """Accumulates per-query metrics into a macro average.

    Macro (mean over queries) rather than micro, so a query with many relevant
    passages does not dominate the reported score.
    """

    def __init__(self, arm: str, ks: Sequence[int] = DEFAULT_KS) -> None:
        self.arm = arm
        self.ks = tuple(ks)
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._latencies: list[float] = []

    def add(self, per_query: dict[str, float], latency_ms: float | None = None) -> None:
        for key, value in per_query.items():
            if value is None or (isinstance(value, float) and math.isnan(value)):
                continue
            self._sums[key] = self._sums.get(key, 0.0) + value
            self._counts[key] = self._counts.get(key, 0) + 1
        if latency_ms is not None:
            self._latencies.append(latency_ms)

    @property
    def n_queries(self) -> int:
        return self._counts.get("mrr", 0)

    def mean(self, key: str) -> float:
        count = self._counts.get(key, 0)
        return self._sums.get(key, 0.0) / count if count else float("nan")

    def finalize(self, extra: dict[str, float] | None = None) -> RankingMetrics:
        from app.observability.metrics import percentile

        ordered = sorted(self._latencies)
        return RankingMetrics(
            arm=self.arm,
            queries=self.n_queries,
            recall={k: self.mean(f"recall@{k}") for k in self.ks},
            precision={k: self.mean(f"precision@{k}") for k in self.ks},
            mrr=self.mean("mrr"),
            ndcg10=self.mean("ndcg@10"),
            latency_p50_ms=percentile(ordered, 50) if ordered else None,
            latency_p95_ms=percentile(ordered, 95) if ordered else None,
            mean_latency_ms=(sum(ordered) / len(ordered)) if ordered else None,
            extra=extra or {},
        )
