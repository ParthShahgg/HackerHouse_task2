#!/usr/bin/env python
"""Retrieval evaluation on held-out MSMARCO-XI queries.

Compares four arms so the architecture is justified by measurement rather than
assertion:

===================  ==========================================================
arm                  what it isolates
===================  ==========================================================
``dense``            BGE-M3 dense only
``sparse``           BGE-M3 learned-sparse only
``hybrid_rrf``       dense + sparse fused with RRF (does fusion beat both?)
``hybrid_rerank``    RRF + bge-reranker-v2-m3 (does reranking add anything?)
===================  ==========================================================

Ground truth is the dataset's own ``is_selected`` judgement, resolved to passage
content hashes offline. Labels are read from ``data/eval/`` and never exist in the
index.

Metrics are computed over **unique passage content hashes**, not chunk ids, so an
arm that emits several chunks per passage is not rewarded for fragmentation. See
``app/evaluation/metrics.py``.

Only the ``test`` slice of the deterministic query split is scored;
``calibration`` is reserved for threshold fitting, so no threshold is ever tuned
on the queries it is reported against.

    python scripts/evaluate_retrieval.py
    python scripts/evaluate_retrieval.py --limit 150 --languages hi mr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REPORTS_DIR, get_settings  # noqa: E402
from app.evaluation.metrics import DEFAULT_KS, MetricAccumulator, evaluate_ranking  # noqa: E402
from app.evaluation.report import write_csv, write_json, write_markdown_table  # noqa: E402
from app.indexing.corpus import load_eval_examples  # noqa: E402
from app.observability.tracing import get_logger, setup_logging  # noqa: E402
from app.schemas.common import LatencyBreakdown, RetrievalMode, now_ns, ns_to_ms  # noqa: E402

logger = get_logger("evaluate_retrieval")

ARMS = ("dense", "sparse", "hybrid_rrf", "hybrid_rerank")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", help="Qdrant collection (default: config).")
    p.add_argument("--languages", nargs="+", help="Restrict to these languages.")
    p.add_argument("--limit", type=int, default=250, help="Max queries to evaluate.")
    p.add_argument("--eval-split", default="test", choices=("test", "calibration", "all"))
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument(
        "--cross-lingual", action="store_true",
        help="Search all configured languages instead of filtering to the query's language.",
    )
    p.add_argument("--rerank-top-k", type=int, help="Candidates fed to the reranker.")
    p.add_argument("--tag", default="", help="Suffix for the report filenames.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    collection = args.collection or settings.qdrant_collection

    eval_split = None if args.eval_split == "all" else args.eval_split
    examples = load_eval_examples(
        collection, eval_split=eval_split, languages=args.languages, with_labels_only=True
    )
    if not examples:
        logger.error(
            "no labelled queries found for collection %r split %r. Build the index first.",
            collection, args.eval_split,
        )
        return 1
    examples = examples[: args.limit]

    by_language: dict[str, int] = {}
    for ex in examples:
        by_language[ex.language] = by_language.get(ex.language, 0) + 1

    print("=" * 78)
    print(f"  collection : {collection}")
    print(f"  eval split : {args.eval_split}  (held out from threshold calibration)")
    print(f"  queries    : {len(examples)}  {by_language}")
    print(f"  arms       : {', '.join(args.arms)}")
    print(f"  retrieval  : {'cross-lingual' if args.cross_lingual else 'language-filtered'}")
    print(f"  device     : {settings.resolved_device()}")
    print("=" * 78)

    from app.retrieval.service import RetrievalService

    service = RetrievalService(collection=collection)
    service.embedder.load()
    if "hybrid_rerank" in args.arms:
        service.reranker.load()

    rerank_top_k = args.rerank_top_k or settings.rerank_top_k
    accumulators = {arm: MetricAccumulator(arm) for arm in args.arms}
    per_language: dict[str, dict[str, MetricAccumulator]] = {
        lang: {arm: MetricAccumulator(arm) for arm in args.arms} for lang in by_language
    }
    max_k = max(DEFAULT_KS)
    skipped = 0

    for index, example in enumerate(examples, start=1):
        relevant = set(example.relevant_hashes)
        if not relevant:
            skipped += 1
            continue

        languages = None if args.cross_lingual else [example.language]
        lat = LatencyBreakdown()

        try:
            embedding = service.embed_query(example.query, lat)
        except Exception as exc:  # noqa: BLE001
            logger.warning("embed failed for query %s: %s", example.query_id, exc)
            skipped += 1
            continue

        query_filter = service.retriever.build_filter(languages, None)

        # ---- dense-only ----
        if "dense" in args.arms:
            t0 = now_ns()
            hits = service.retriever.search_dense(
                embedding.dense, limit=max(max_k, settings.dense_top_k),
                query_filter=query_filter, collection=collection,
            )
            elapsed = ns_to_ms(now_ns() - t0)
            ranked = [h.payload.get("content_hash", "") for h in hits]
            metrics = evaluate_ranking(ranked, relevant)
            accumulators["dense"].add(metrics, elapsed)
            per_language[example.language]["dense"].add(metrics, elapsed)

        # ---- sparse-only ----
        if "sparse" in args.arms:
            t0 = now_ns()
            hits = service.retriever.search_sparse(
                embedding.sparse_dict(), limit=max(max_k, settings.sparse_top_k),
                query_filter=query_filter, collection=collection,
            )
            elapsed = ns_to_ms(now_ns() - t0)
            ranked = [h.payload.get("content_hash", "") for h in hits]
            metrics = evaluate_ranking(ranked, relevant)
            accumulators["sparse"].add(metrics, elapsed)
            per_language[example.language]["sparse"].add(metrics, elapsed)

        # ---- hybrid RRF (also the candidate source for the rerank arm) ----
        retrieval = None
        if "hybrid_rrf" in args.arms or "hybrid_rerank" in args.arms:
            rrf_lat = LatencyBreakdown()
            t0 = now_ns()
            retrieval = service.retrieve(
                embedding,
                languages=languages,
                mode=RetrievalMode.CROSS_LINGUAL if args.cross_lingual else RetrievalMode.LANGUAGE_FILTERED,
                latency=rrf_lat,
                limit=max(rerank_top_k, max_k),
                collection=collection,
            )
            elapsed = ns_to_ms(now_ns() - t0)
            if "hybrid_rrf" in args.arms:
                ranked = [c.content_hash or "" for c in retrieval.candidates]
                metrics = evaluate_ranking(ranked, relevant)
                accumulators["hybrid_rrf"].add(metrics, elapsed)
                per_language[example.language]["hybrid_rrf"].add(metrics, elapsed)

        # ---- hybrid + reranker ----
        if "hybrid_rerank" in args.arms and retrieval is not None and retrieval.candidates:
            rr_lat = LatencyBreakdown()
            t0 = now_ns()
            rerank = service.rerank(
                example.query, retrieval, latency=rr_lat,
                rerank_top_k=rerank_top_k, final_top_k=max_k, expand_parents=False,
            )
            elapsed = ns_to_ms(now_ns() - t0)
            ranked = [c.content_hash or "" for c in rerank.candidates]
            metrics = evaluate_ranking(ranked, relevant)
            accumulators["hybrid_rerank"].add(metrics, elapsed)
            per_language[example.language]["hybrid_rerank"].add(metrics, elapsed)

        if index % 10 == 0 or index == len(examples):
            print(f"  ... {index}/{len(examples)} queries", flush=True)

    # ------------------------------------------------------------------ report
    results = {arm: acc.finalize() for arm, acc in accumulators.items()}
    rows = [results[arm].to_row() for arm in args.arms if results[arm].queries]

    lang_rows: list[dict] = []
    for lang, arms in per_language.items():
        for arm in args.arms:
            final = arms[arm].finalize()
            if final.queries:
                row = final.to_row()
                row = {"language": lang, **row}
                lang_rows.append(row)

    print("\n" + "=" * 78)
    print("RETRIEVAL EVALUATION (measured)")
    print("=" * 78)
    print(write_markdown_table(rows))
    print("\nPer-language:")
    print(write_markdown_table(lang_rows))

    suffix = f"_{args.tag}" if args.tag else ""
    payload = {
        "collection": collection,
        "eval_split": args.eval_split,
        "queries_evaluated": len(examples) - skipped,
        "queries_skipped": skipped,
        "queries_by_language": by_language,
        "retrieval_mode": "cross_lingual" if args.cross_lingual else "language_filtered",
        "rerank_top_k": rerank_top_k,
        "device": settings.resolved_device(),
        "int8_reranker": settings.int8_reranker_enabled(),
        "torch_threads": settings.torch_num_threads,
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "fusion_mode": settings.retrieval_fusion_mode,
        "rrf_k": settings.rrf_k,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "overall": rows,
        "per_language": lang_rows,
        "notes": (
            "Metrics are macro-averaged over queries and computed on unique "
            "passage content hashes. Recall@k denominator is |relevant|. "
            "Latency columns are wall-clock for that arm on the reported device."
        ),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / f"retrieval_eval{suffix}.json", payload)
    write_csv(REPORTS_DIR / f"retrieval_eval{suffix}.csv", rows + lang_rows)

    md = [
        "# Retrieval Evaluation\n",
        f"- dataset: `{settings.dataset_id}`",
        f"- collection: `{collection}`",
        f"- eval split: `{args.eval_split}` (disjoint from threshold calibration)",
        f"- queries: **{len(examples) - skipped}** {by_language}",
        f"- retrieval: {'cross-lingual' if args.cross_lingual else 'language-filtered'}",
        f"- device: `{settings.resolved_device()}`, int8 reranker: "
        f"`{settings.int8_reranker_enabled()}`, threads: `{settings.torch_num_threads}`",
        f"- embedding: `{settings.embedding_model}`",
        f"- reranker: `{settings.reranker_model}` (top-{rerank_top_k})",
        "",
        "Metrics are macro-averaged over queries, computed on unique passage",
        "content hashes so multi-chunk strategies are not rewarded for",
        "fragmentation. Ground truth is the dataset's `is_selected` label.",
        "",
        "## Overall\n",
        write_markdown_table(rows),
        "\n## Per language\n",
        write_markdown_table(lang_rows),
    ]
    (REPORTS_DIR / f"retrieval_eval{suffix}.md").write_text("\n".join(md), encoding="utf-8")
    print(f"reports -> {REPORTS_DIR / f'retrieval_eval{suffix}.*'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
