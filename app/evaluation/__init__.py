"""Retrieval / chunking evaluation and report rendering."""

from app.evaluation.metrics import (
    DEFAULT_KS,
    MetricAccumulator,
    RankingMetrics,
    collapse_to_unique,
    evaluate_ranking,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.report import write_csv, write_json, write_markdown_table

__all__ = [
    "DEFAULT_KS",
    "MetricAccumulator",
    "RankingMetrics",
    "collapse_to_unique",
    "evaluate_ranking",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "write_csv",
    "write_json",
    "write_markdown_table",
]
