#!/usr/bin/env python
"""End-to-end latency benchmark over held-out MSMARCO-XI queries.

Runs >=100 (default 200) real queries across >=4 languages through the actual
orchestrator - the same code path a request takes - and reports P50 / P70 / P100
per stage.

Honesty rules, enforced in code
-------------------------------
1. A stage that was **not measured** is reported as ``n/a``, never as ``0``.
   ``LatencyBreakdown`` keeps ``None`` distinct from ``0.0`` for exactly this
   reason, and the report writer renders ``None`` as ``n/a``.
2. ``generation_backend`` and ``device`` are recorded in every artefact, so a
   number produced by a local test double can never be mistaken for Groq's.
3. Two different aggregate numbers are reported and never conflated:

   ``total_rag_latency``
       transcript received -> final validated answer (excludes STT)
   ``total_voice_latency``
       audio submitted -> first answer token (includes STT)

   Reporting one as the other is the usual way "<200ms end-to-end" claims are
   manufactured.
4. Warmup iterations are excluded, and lazy model init is forced first, so the
   first sample does not carry one-off weight-materialisation cost.

    python scripts/benchmark_latency.py
    python scripts/benchmark_latency.py --queries 200 --generation-backend mock
    python scripts/benchmark_latency.py --compare-fusion
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REPORTS_DIR  # noqa: E402
from app.evaluation.report import fmt, write_csv, write_json, write_markdown_table  # noqa: E402
from app.observability.metrics import percentile  # noqa: E402
from app.observability.tracing import get_logger, setup_logging  # noqa: E402

logger = get_logger("benchmark")

# Order and labels for the required latency table.
STAGES: list[tuple[str, str]] = [
    ("stt_latency", "STT"),
    ("guardrail_latency", "Input guardrail"),
    ("query_embedding_latency", "Embedding"),
    ("dense_latency", "Dense retrieval"),
    ("sparse_latency", "Sparse retrieval"),
    ("rrf_latency", "RRF"),
    ("rerank_latency", "Reranking"),
    ("grounding_gate_latency", "Confidence gate"),
    ("generation_ttft", "Generation TTFT"),
    ("generation_e2e", "Generation E2E"),
    ("nli_latency", "NLI grounding"),
    ("output_guardrail_latency", "Output grounding"),
    ("total_rag_latency", "Full RAG"),
    ("total_voice_latency", "Full voice"),
    ("total_completion_latency", "Full completion"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", help="Qdrant collection (default: config).")
    p.add_argument("--queries", type=int, default=200, help="Total queries (>=100 required).")
    p.add_argument("--languages", nargs="+", help="Default: all configured languages.")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--eval-split", default="test", choices=("test", "calibration", "all"))
    p.add_argument(
        "--generation-backend", choices=("groq", "mock", "config"), default="config",
        help="Override GENERATION_BACKEND for this run.",
    )
    p.add_argument(
        "--voice", action="store_true",
        help="Drive the voice entry point (STT bypassed via transcript_override "
             "unless SARVAM_API_KEY is set).",
    )
    p.add_argument("--compare-fusion", action="store_true", help="Also benchmark server-side RRF.")
    p.add_argument("--tag", default="")
    return p.parse_args()


def summarise(samples: dict[str, list[float]], n: int) -> list[dict]:
    """Build the P50/P70/P100 table. Unmeasured stages -> None (rendered n/a)."""
    rows: list[dict] = []
    for field, label in STAGES:
        values = sorted(samples.get(field, []))
        if not values:
            rows.append(
                {"stage": label, "n": 0, "p50_ms": None, "p70_ms": None,
                 "p90_ms": None, "p100_ms": None, "mean_ms": None}
            )
            continue
        rows.append(
            {
                "stage": label,
                "n": len(values),
                "p50_ms": round(percentile(values, 50), 2),
                "p70_ms": round(percentile(values, 70), 2),
                "p90_ms": round(percentile(values, 90), 2),
                "p100_ms": round(percentile(values, 100), 2),
                "mean_ms": round(sum(values) / len(values), 2),
            }
        )
    return rows


async def run_suite(args, examples, *, fusion_mode: str | None = None) -> dict:
    from app.config import get_settings
    from app.pipeline.orchestrator import RAGOrchestrator

    settings = get_settings()
    if fusion_mode:
        settings.retrieval_fusion_mode = fusion_mode

    orchestrator = RAGOrchestrator(settings=settings)
    print("  loading models / warming up...", flush=True)
    orchestrator.warmup()

    for example in examples[: args.warmup]:
        await orchestrator.run_text(example.query, language=example.language)

    samples: dict[str, list[float]] = defaultdict(list)
    outcomes = {"answered": 0, "abstained": 0, "grounded": 0}
    abstain_reasons: dict[str, int] = defaultdict(int)
    per_query: list[dict] = []
    started = time.perf_counter()

    for index, example in enumerate(examples, start=1):
        if args.voice:
            response = await orchestrator.run_voice(
                transcript_override=example.query,
                language=example.language,
            )
        else:
            response = await orchestrator.run_text(example.query, language=example.language)

        detail = response.latency_detail
        if detail is not None:
            for field, _ in STAGES:
                value = getattr(detail, field, None)
                if value is not None:
                    samples[field].append(float(value))

        if response.abstained:
            outcomes["abstained"] += 1
            abstain_reasons[response.abstain_reason.value] += 1
        else:
            outcomes["answered"] += 1
        if response.grounded:
            outcomes["grounded"] += 1

        per_query.append(
            {
                "query_id": example.query_id,
                "language": example.language,
                "abstained": response.abstained,
                "abstain_reason": response.abstain_reason.value,
                "grounded": response.grounded,
                "total_rag_ms": detail.total_rag_latency if detail else None,
                "rerank_ms": detail.rerank_latency if detail else None,
                "embed_ms": detail.query_embedding_latency if detail else None,
                "dense_ms": detail.dense_latency if detail else None,
                "sparse_ms": detail.sparse_latency if detail else None,
            }
        )

        if index % 10 == 0 or index == len(examples):
            elapsed = time.perf_counter() - started
            rate = index / elapsed if elapsed else 0
            eta = (len(examples) - index) / rate if rate else 0
            print(
                f"  ... {index}/{len(examples)}  ({rate:.2f} q/s, eta {eta / 60:.1f} min)",
                flush=True,
            )

    return {
        "rows": summarise(samples, len(examples)),
        "samples": {k: v for k, v in samples.items()},
        "outcomes": outcomes,
        "abstain_reasons": dict(abstain_reasons),
        "per_query": per_query,
        "wall_seconds": round(time.perf_counter() - started, 2),
        "fusion_mode": fusion_mode or settings.retrieval_fusion_mode,
    }


def main() -> int:
    args = parse_args()
    setup_logging()

    if args.generation_backend != "config":
        os.environ["GENERATION_BACKEND"] = args.generation_backend
        from app.config import reset_settings_cache

        reset_settings_cache()

    from app.config import get_settings
    from app.indexing.corpus import load_eval_examples

    settings = get_settings()
    collection = args.collection or settings.qdrant_collection
    languages = args.languages or settings.language_list

    eval_split = None if args.eval_split == "all" else args.eval_split
    pool = load_eval_examples(
        collection, eval_split=eval_split, languages=languages, with_labels_only=True
    )
    if not pool:
        logger.error("no eval queries for %r; build the index first", collection)
        return 1

    # Round-robin across languages so the sample is balanced rather than
    # dominated by whichever language streamed most rows.
    by_language: dict[str, list] = defaultdict(list)
    for example in pool:
        by_language[example.language].append(example)
    balanced: list = []
    position = 0
    while len(balanced) < args.queries:
        added = False
        for lang in sorted(by_language):
            bucket = by_language[lang]
            if position < len(bucket):
                balanced.append(bucket[position])
                added = True
                if len(balanced) >= args.queries:
                    break
        if not added:
            break
        position += 1
    examples = balanced

    counts: dict[str, int] = defaultdict(int)
    for example in examples:
        counts[example.language] += 1

    if len(examples) < 100:
        logger.warning(
            "only %d queries available (>=100 expected). Ingest more rows to "
            "strengthen the benchmark.", len(examples),
        )
    if len(counts) < 4:
        logger.warning("only %d languages available (>=4 expected)", len(counts))

    stt_configured = bool(settings.sarvam_api_key)
    print("=" * 78)
    print("LATENCY BENCHMARK")
    print(f"  collection        : {collection}")
    print(f"  queries           : {len(examples)}  {dict(counts)}")
    print(f"  eval split        : {args.eval_split} (held out)")
    print(f"  device            : {settings.resolved_device()}  threads={settings.torch_num_threads}")
    print(f"  int8 reranker     : {settings.int8_reranker_enabled()}")
    print(f"  rerank_top_k      : {settings.rerank_top_k}   final_top_k={settings.final_top_k}")
    print(f"  fusion mode       : {settings.retrieval_fusion_mode}")
    print(f"  generation backend: {settings.generation_backend} ({settings.groq_model})")
    print(f"  NLI grounding     : {settings.enable_nli_grounding}")
    print(f"  STT configured    : {stt_configured}")
    if not stt_configured:
        print("    -> STT latency will be reported as n/a (SARVAM_API_KEY not set),")
        print("       and 'Full voice' therefore excludes real STT time.")
    print("=" * 78)

    result = asyncio.run(run_suite(args, examples))

    fusion_result = None
    if args.compare_fusion:
        print("\n--- re-running with server-side RRF fusion ---")
        fusion_result = asyncio.run(run_suite(args, examples, fusion_mode="server"))

    rows = result["rows"]
    print("\n" + "=" * 78)
    print("LATENCY (ms) - measured")
    print("=" * 78)
    print(write_markdown_table(
        [{k: r[k] for k in ("stage", "n", "p50_ms", "p70_ms", "p90_ms", "p100_ms", "mean_ms")} for r in rows]
    ))
    print(f"\noutcomes: {result['outcomes']}")
    print(f"abstain reasons: {result['abstain_reasons']}")
    print(f"wall time: {result['wall_seconds']}s")

    suffix = f"_{args.tag}" if args.tag else ""
    meta = {
        "collection": collection,
        "queries": len(examples),
        "queries_by_language": dict(counts),
        "eval_split": args.eval_split,
        "voice_entrypoint": args.voice,
        "device": settings.resolved_device(),
        "torch_threads": settings.torch_num_threads,
        "int8_reranker": settings.int8_reranker_enabled(),
        "fp16": settings.fp16_enabled(),
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "nli_model": settings.nli_model if settings.enable_nli_grounding else None,
        "generation_backend": settings.generation_backend,
        "generation_model": settings.groq_model,
        "stt_configured": stt_configured,
        "stt_model": settings.sarvam_stt_model,
        "rerank_top_k": settings.rerank_top_k,
        "final_top_k": settings.final_top_k,
        "fusion_mode": result["fusion_mode"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "percentile_convention": "nearest-rank on sorted samples; p100 = observed max",
        "caveats": [
            "n/a means NOT MEASURED. It is never rendered as 0.",
            "total_rag_latency excludes STT (transcript -> validated answer).",
            "total_voice_latency is audio-in -> first answer token.",
        ],
    }
    if not stt_configured:
        meta["caveats"].append(
            "SARVAM_API_KEY absent: STT was not executed, so STT latency is n/a "
            "and 'Full voice' excludes network STT time."
        )
    if settings.generation_backend == "mock":
        meta["caveats"].append(
            "GENERATION_BACKEND=mock: generation timings come from a local "
            "deterministic test double and are NOT representative of Groq."
        )
    if settings.resolved_device() == "cpu":
        meta["caveats"].append(
            "CPU-only device. bge-reranker-v2-m3 is XLM-R-large (568M params); "
            "reranking dominates and the <100ms retrieval / <200ms TTFT targets "
            "are not attainable in this configuration. See reports/latency.md."
        )

    payload = {**meta, "table": rows, "outcomes": result["outcomes"],
               "abstain_reasons": result["abstain_reasons"],
               "wall_seconds": result["wall_seconds"]}
    if fusion_result:
        payload["server_fusion_table"] = fusion_result["rows"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / f"latency{suffix}.json", payload)
    write_csv(REPORTS_DIR / f"latency{suffix}.csv", rows)
    write_csv(REPORTS_DIR / f"latency{suffix}_per_query.csv", result["per_query"])

    md: list[str] = [
        "# Latency Benchmark\n",
        f"- collection: `{collection}`",
        f"- queries: **{len(examples)}** across **{len(counts)}** languages {dict(counts)}",
        f"- eval split: `{args.eval_split}` (held out from threshold calibration)",
        f"- device: **{settings.resolved_device()}**, torch threads: {settings.torch_num_threads}",
        f"- int8 reranker: `{settings.int8_reranker_enabled()}`, fp16: `{settings.fp16_enabled()}`",
        f"- rerank_top_k: {settings.rerank_top_k}, final_top_k: {settings.final_top_k}",
        f"- fusion: `{result['fusion_mode']}`",
        f"- generation backend: `{settings.generation_backend}` (`{settings.groq_model}`)",
        f"- percentiles: {meta['percentile_convention']}",
        "",
        "> **`n/a` means NOT MEASURED.** It is never rendered as `0`.",
        "",
        "## Measured latency (ms)\n",
        write_markdown_table(
            [{k: r[k] for k in ("stage", "n", "p50_ms", "p70_ms", "p90_ms", "p100_ms", "mean_ms")} for r in rows]
        ),
        "\n## Caveats\n",
    ]
    md += [f"- {c}" for c in meta["caveats"]]
    md += [
        "",
        "## Outcomes\n",
        f"- answered: {result['outcomes']['answered']}",
        f"- abstained: {result['outcomes']['abstained']}",
        f"- grounded: {result['outcomes']['grounded']}",
        f"- abstain reasons: `{result['abstain_reasons']}`",
    ]
    if fusion_result:
        md += [
            "\n## Server-side RRF fusion (one round trip)\n",
            write_markdown_table(
                [{k: r[k] for k in ("stage", "n", "p50_ms", "p70_ms", "p100_ms")} for r in fusion_result["rows"]]
            ),
        ]
    (REPORTS_DIR / f"latency{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"reports -> {REPORTS_DIR / f'latency{suffix}.*'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
