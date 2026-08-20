#!/usr/bin/env python
"""Diff the two candidate-pool arms into reports/rerank_topk_ab.md.

Reads reports/retrieval_eval_topk30.json and retrieval_eval_topk10.json (produced
by scripts/ab_rerank_topk.ps1) and reports the delta per arm and metric.

Interpretation note baked into the output: for the ``hybrid_rerank`` arm the
ranked list is at most ``rerank_top_k`` long, so **Recall@10 is pool-capped** when
rerank_top_k=10 and is not comparable across arms. Recall@5, MRR and nDCG@10 are
comparable, since all fit inside a 10-candidate pool.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REPORTS_DIR  # noqa: E402
from app.evaluation.report import write_markdown_table  # noqa: E402

# Recall@10 excluded from the verdict: pool-capped for rerank_top_k=10.
COMPARABLE = ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@10")
TOLERANCE = 0.02  # "within ~2 points"


def main() -> int:
    a_path = REPORTS_DIR / "retrieval_eval_topk30.json"
    b_path = REPORTS_DIR / "retrieval_eval_topk10.json"
    for path in (a_path, b_path):
        if not path.exists():
            print(f"missing {path}; run scripts/ab_rerank_topk.ps1 first")
            return 1

    a = json.loads(a_path.read_text(encoding="utf-8"))
    b = json.loads(b_path.read_text(encoding="utf-8"))
    a_arms = {row["arm"]: row for row in a["overall"]}
    b_arms = {row["arm"]: row for row in b["overall"]}

    rows: list[dict] = []
    regressions: list[str] = []
    for arm in sorted(set(a_arms) & set(b_arms)):
        for metric in (*COMPARABLE, "recall@10"):
            av, bv = a_arms[arm].get(metric), b_arms[arm].get(metric)
            if av is None or bv is None:
                continue
            delta = bv - av
            capped = metric == "recall@10" and arm == "hybrid_rerank"
            rows.append({
                "arm": arm,
                "metric": metric,
                "pool30_rerank30": round(av, 4),
                "pool15_rerank10": round(bv, 4),
                "delta": round(delta, 4),
                "note": "pool-capped, not comparable" if capped else "",
            })
            if metric in COMPARABLE and delta < -TOLERANCE:
                regressions.append(f"{arm}/{metric}: {av:.4f} -> {bv:.4f} ({delta:+.4f})")

    lat_rows = []
    for arm in sorted(set(a_arms) & set(b_arms)):
        lat_rows.append({
            "arm": arm,
            "p50_ms_pool30": a_arms[arm].get("latency_p50_ms"),
            "p50_ms_pool15": b_arms[arm].get("latency_p50_ms"),
            "speedup_x": (
                round(a_arms[arm]["latency_p50_ms"] / b_arms[arm]["latency_p50_ms"], 2)
                if a_arms[arm].get("latency_p50_ms") and b_arms[arm].get("latency_p50_ms")
                else None
            ),
        })

    verdict = (
        "PASS - every comparable metric is within "
        f"{TOLERANCE:.2f} of the pool-30 baseline."
        if not regressions
        else "REGRESSION - see list below."
    )

    print("\n" + write_markdown_table(rows))
    print(write_markdown_table(lat_rows))
    print(f"\nVERDICT: {verdict}")
    for line in regressions:
        print(f"  - {line}")

    md = [
        "# Candidate-pool A/B: does shrinking the pool cost quality?\n",
        f"- queries: **{a['queries_evaluated']}** (identical set, identical order, both arms)",
        f"- languages: `{a['queries_by_language']}`",
        f"- eval split: `{a['eval_split']}` (held out)",
        f"- device: `{a['device']}`, int8 reranker: `{a['int8_reranker']}`",
        "",
        "**Arm A** dense/sparse/rrf = 30, `rerank_top_k` = 30 (previous default)  ",
        "**Arm B** dense/sparse/rrf = 15, `rerank_top_k` = 10 (new default)",
        "",
        "Motivation: the 30-candidate cross-encoder pass cost ~14 s/query on CPU and",
        "starved the co-located Qdrant container, which killed an evaluation run with",
        "`Server disconnected without sending a response`. Shrinking the pool is only",
        "a good trade if retrieval quality holds.",
        "",
        f"## Verdict: {verdict}\n",
    ]
    if regressions:
        md += [f"- {line}" for line in regressions] + [""]
    md += [
        "## Quality\n",
        write_markdown_table(rows),
        "",
        "> `Recall@10` for `hybrid_rerank` is **pool-capped** when `rerank_top_k=10`:",
        "> the reranked list cannot contain more than 10 candidates, so that single",
        "> cell is a measurement artefact of the pool size and is excluded from the",
        "> verdict. `Recall@1/3/5`, `MRR` and `nDCG@10` all fit inside a 10-candidate",
        "> pool and are directly comparable.",
        "",
        "## Cost\n",
        write_markdown_table(lat_rows),
    ]
    target = REPORTS_DIR / "rerank_topk_ab.md"
    target.write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {target}")
    return 0 if not regressions else 2


if __name__ == "__main__":
    raise SystemExit(main())
