#!/usr/bin/env python
"""Consolidate the measured artefacts into reports/SUMMARY.md.

Reads only what the other scripts actually produced. If an artefact is missing the
corresponding table says so rather than being invented, and any stage that was not
measured renders as ``n/a`` rather than ``0``.

    python scripts/make_summary.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REPORTS_DIR  # noqa: E402
from app.evaluation.report import write_markdown_table  # noqa: E402

# The exact rows required by the final latency table, in order.
LATENCY_ROWS = [
    "STT", "Embedding", "Dense retrieval", "Sparse retrieval", "RRF", "Reranking",
    "Generation TTFT", "Generation E2E", "Output grounding", "Full RAG", "Full voice",
]


def load(name: str) -> dict | None:
    path = REPORTS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    latency = load("latency.json")
    retrieval = load("retrieval_eval.json")
    chunking = load("chunking_eval.json")
    calibration = load("calibration.json")
    scenarios = load("demo_scenarios.json")
    index_report = None
    index_path = Path("data/corpus/msmarco_xi_demo/index_report.json")
    if index_path.exists():
        index_report = json.loads(index_path.read_text(encoding="utf-8"))

    out: list[str] = ["# Measured Results — Summary\n"]
    out.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}._\n")
    out.append(
        "> Every number here was produced by the scripts in this repository on the\n"
        "> hardware stated below. **`n/a` means NOT MEASURED** and is never rendered\n"
        "> as `0`.\n"
    )

    # ---------------------------------------------------------------- environment
    meta = latency or retrieval or {}
    out.append("## Environment\n")
    env_rows = [
        {"setting": "device", "value": meta.get("device", "n/a")},
        {"setting": "torch threads", "value": meta.get("torch_threads", "n/a")},
        {"setting": "int8 reranker", "value": meta.get("int8_reranker", "n/a")},
        {"setting": "fp16", "value": meta.get("fp16", "n/a")},
        {"setting": "embedding model", "value": meta.get("embedding_model", "BAAI/bge-m3")},
        {"setting": "reranker model", "value": meta.get("reranker_model", "BAAI/bge-reranker-v2-m3")},
        {"setting": "generation backend", "value": meta.get("generation_backend", "n/a")},
        {"setting": "generation model", "value": meta.get("generation_model", "n/a")},
        {"setting": "STT configured", "value": meta.get("stt_configured", "n/a")},
        {"setting": "fusion mode", "value": meta.get("fusion_mode", "n/a")},
        {"setting": "rerank_top_k", "value": meta.get("rerank_top_k", "n/a")},
    ]
    out.append(write_markdown_table(env_rows))

    if index_report:
        out.append("\n## Corpus\n")
        out.append(write_markdown_table([
            {"metric": "vectors indexed", "value": index_report["chunks"]},
            {"metric": "dense dim", "value": index_report["dim"]},
            {"metric": "languages", "value": ", ".join(index_report["languages"])},
            {"metric": "split", "value": index_report["split"]},
            {"metric": "rows per language", "value": index_report["max_rows_per_language"]},
            {"metric": "embed throughput (chunks/s)", "value": index_report["embed_chunks_per_s"]},
            {"metric": "total build seconds", "value": index_report["total_seconds"]},
        ]))

    # ------------------------------------------------------------------- latency
    out.append("\n## Latency (ms) — P50 / P70 / P100\n")
    if latency:
        by_stage = {row["stage"]: row for row in latency["table"]}
        rows = []
        for label in LATENCY_ROWS:
            row = by_stage.get(label)
            if row is None:
                rows.append({"stage": label, "n": 0, "P50": None, "P70": None, "P100": None})
                continue
            rows.append({
                "stage": label, "n": row["n"],
                "P50": row["p50_ms"], "P70": row["p70_ms"], "P100": row["p100_ms"],
            })
        out.append(write_markdown_table(rows))
        out.append(
            f"\nQueries: **{latency['queries']}** across "
            f"**{len(latency['queries_by_language'])}** languages "
            f"`{latency['queries_by_language']}`.\n"
        )
        out.append("\n**Caveats**\n")
        out += [f"- {c}" for c in latency.get("caveats", [])]
        outcomes = latency.get("outcomes", {})
        out.append(
            f"\nOutcomes: answered={outcomes.get('answered')}, "
            f"abstained={outcomes.get('abstained')}, grounded={outcomes.get('grounded')}. "
            f"Abstain reasons: `{latency.get('abstain_reasons')}`.\n"
        )
    else:
        out.append("_`reports/latency.json` not found — run `scripts/benchmark_latency.py`._\n")

    # ----------------------------------------------------------------- retrieval
    out.append("\n## Retrieval quality (held-out `test` split)\n")
    if retrieval:
        rows = []
        for row in retrieval["overall"]:
            rows.append({
                "arm": row["arm"], "queries": row["queries"],
                "Recall@1": row.get("recall@1"), "Recall@3": row.get("recall@3"),
                "Recall@5": row.get("recall@5"), "Recall@10": row.get("recall@10"),
                "MRR": row.get("mrr"), "nDCG@10": row.get("ndcg@10"),
                "latency_p50_ms": row.get("latency_p50_ms"),
            })
        out.append(write_markdown_table(rows))
        out.append(f"\nQueries: {retrieval['queries_evaluated']} `{retrieval['queries_by_language']}`\n")

        if retrieval.get("per_language"):
            out.append("\n### Per language\n")
            out.append(write_markdown_table([
                {"language": r["language"], "arm": r["arm"], "Recall@5": r.get("recall@5"),
                 "MRR": r.get("mrr"), "nDCG@10": r.get("ndcg@10")}
                for r in retrieval["per_language"]
            ]))
    else:
        out.append("_`reports/retrieval_eval.json` not found — run `scripts/evaluate_retrieval.py`._\n")

    # ------------------------------------------------------------------ chunking
    out.append("\n## Chunking strategy comparison\n")
    if chunking:
        out.append(write_markdown_table([
            {"Strategy": r["strategy"], "Ranking": r["ranking"], "Recall@5": r["recall@5"],
             "MRR": r["mrr"], "nDCG@10": r["ndcg@10"], "Vectors": r["vectors"],
             "Avg chunk chars": r["avg_chunk_chars"], "Latency p50 ms": r["latency_p50_ms"]}
            for r in chunking["rows"]
        ]))
        out.append(
            f"\nAll arms share {chunking['parents']} identical parent passages and "
            f"{chunking['queries']} identical queries; metrics are computed on unique "
            f"passage content hashes so extra chunks per passage confer no advantage.\n"
        )
        out.append("\n### Index cost\n")
        out.append(write_markdown_table([
            {"strategy": arm, "vectors": d["vectors"],
             "vectors/passage": d["vectors_per_passage"],
             "avg chunk chars": d["avg_chunk_chars"]}
            for arm, d in chunking["details"].items()
        ]))
    else:
        out.append("_`reports/chunking_eval.json` not found — run `scripts/evaluate_chunking.py`._\n")

    # --------------------------------------------------------------- calibration
    out.append("\n## Abstention threshold calibration\n")
    if calibration:
        artefact = calibration["artefact"]
        fitted = artefact["fitted_on"]
        out.append(write_markdown_table([
            {"field": "rerank_abstain_below", "value": artefact["rerank_abstain_below"]},
            {"field": "rerank_margin_min", "value": artefact["rerank_margin_min"]},
            {"field": "objective", "value": artefact["objective"]},
            {"field": "in-corpus queries", "value": fitted["in_corpus_queries"]},
            {"field": "out-of-corpus queries", "value": fitted["out_of_corpus_queries"]},
            {"field": "positives / negatives", "value": f"{fitted['positives']} / {fitted['negatives']}"},
            {"field": "precision at operating point", "value": artefact["chosen_operating_point"]["precision"]},
            {"field": "recall at operating point", "value": artefact["chosen_operating_point"]["recall"]},
            {"field": "int8 quantized", "value": artefact["model_config"]["int8_quantized"]},
        ]))
        out.append(
            "\nFitted on the `calibration` split only; all retrieval metrics above are "
            "reported on the disjoint `test` split.\n"
        )
    else:
        out.append("_`reports/calibration.json` not found — run `scripts/calibrate_thresholds.py`._\n")

    # ----------------------------------------------------------------- scenarios
    out.append("\n## Demo scenarios\n")
    if scenarios:
        out.append(write_markdown_table([
            {"scenario": r["scenario"], "abstained": r["abstained"], "reason": r["reason"],
             "grounded": r["grounded"], "citations": r["citations"], "rag_ms": r["rag_ms"]}
            for r in scenarios["rows"]
        ]))
    else:
        out.append("_`reports/demo_scenarios.json` not found — run `scripts/run_demo_scenarios.py`._\n")

    target = REPORTS_DIR / "SUMMARY.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
