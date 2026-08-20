#!/usr/bin/env python
"""Compare chunking / indexing strategies on identical data.

Arms
----
``native``           1 passage = 1 chunk (baseline / control)
``sentence_window``  overlapping sentence-window children
``semantic_split``   embedding-similarity breakpoints within a passage
``fixed_fallback``   fixed-size token windows with ~17.5% overlap
``routed``           the production router: native + at most one child strategy
                     chosen per passage

Fairness
--------
Every arm is built from the **same deduplicated parent passages** and evaluated
on the **same queries**. Each arm gets its own Qdrant collection, so no arm can
see another's vectors. Metrics are computed on unique passage content hashes, so
an arm that emits 6 vectors per passage is not credited for merely occupying more
result slots.

Reported per arm: Recall@5, MRR, nDCG@10, vector count, average chunk length,
retrieval latency and reranking latency - i.e. quality *and* cost, since a
strategy that buys +1% recall for 3x the index is not obviously a win.

Cost note: this builds and embeds one index per arm. It defaults to a smaller
passage set than the main demo corpus for that reason; the comparison stays valid
because all arms share it.

    python scripts/evaluate_chunking.py --max-rows-per-language 40
    python scripts/evaluate_chunking.py --arms native sentence_window --keep
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import ALL_STRATEGIES, ChunkerContext, ChunkingEngine  # noqa: E402
from app.config import REPORTS_DIR, get_settings  # noqa: E402
from app.evaluation.metrics import MetricAccumulator, evaluate_ranking  # noqa: E402
from app.evaluation.report import write_csv, write_json, write_markdown_table  # noqa: E402
from app.indexing.corpus import CorpusBuilder, CorpusPaths, load_eval_examples, load_parents  # noqa: E402
from app.observability.tracing import get_logger, setup_logging  # noqa: E402
from app.schemas.common import LatencyBreakdown, RetrievalMode, now_ns, ns_to_ms  # noqa: E402

logger = get_logger("evaluate_chunking")

ARMS = (*ALL_STRATEGIES, "routed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--split", default=None, choices=("train", "validation"))
    p.add_argument("--max-rows-per-language", type=int, default=40)
    p.add_argument("--limit", type=int, default=120, help="Queries to evaluate per arm.")
    p.add_argument("--eval-split", default="test", choices=("test", "calibration", "all"))
    p.add_argument("--base-name", default="chunkeval")
    p.add_argument("--keep", action="store_true", help="Do not delete the per-arm collections.")
    p.add_argument("--rerank-top-k", type=int, default=20)
    p.add_argument("--reuse", action="store_true", help="Reuse existing per-arm collections.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    languages = args.languages or settings.language_list
    split = args.split or settings.dataset_split
    base = args.base_name

    print("=" * 78)
    print("CHUNKING STRATEGY COMPARISON")
    print(f"  arms        : {', '.join(args.arms)}")
    print(f"  languages   : {', '.join(languages)}")
    print(f"  rows/lang   : {args.max_rows_per_language}")
    print(f"  device      : {settings.resolved_device()}")
    print("=" * 78)

    from app.retrieval.embedder import get_embedder
    from app.retrieval.service import RetrievalService
    from app.retrieval.store import QdrantStore

    embedder = get_embedder()
    embedder.load()

    ctx = ChunkerContext.from_settings(settings)
    ctx.embed_sentences = embedder.embed_sentences
    ctx.count_tokens = embedder.count_tokens
    ctx.encode_tokens = embedder.encode_token_ids
    ctx.decode_tokens = embedder.decode_token_ids
    engine = ChunkingEngine(ctx=ctx, enabled=ALL_STRATEGIES)

    # ---- one shared parent set + one shared label set ----
    paths = CorpusPaths(base)
    if args.reuse and paths.exists():
        print("\nreusing shared parent corpus")
    else:
        print("\nbuilding shared parent corpus (streamed once, reused by all arms)...")
        CorpusBuilder(
            name=base,
            split=split,
            languages=languages,
            max_rows_per_language=args.max_rows_per_language,
            strategies=["native"],
            engine=ChunkingEngine(ctx=ctx, enabled=["native"]),
        ).build()

    parents = load_parents(base)
    eval_split = None if args.eval_split == "all" else args.eval_split
    examples = load_eval_examples(
        base, eval_split=eval_split, languages=languages, with_labels_only=True
    )[: args.limit]
    print(f"  parents: {len(parents):,}   eval queries: {len(examples)}")
    if not examples:
        logger.error("no labelled queries; aborting")
        return 1

    rows: list[dict] = []
    details: dict[str, dict] = {}

    for arm in args.arms:
        collection = f"{base}_{arm}"
        print(f"\n--- arm: {arm} -> collection {collection} ---")

        # ---- chunk ----
        t0 = now_ns()
        chunks = []
        for parent in parents:
            chunks.extend(
                engine.chunk(parent) if arm == "routed" else engine.chunk_forced(parent, arm)
            )
        chunk_ms = ns_to_ms(now_ns() - t0)
        if not chunks:
            logger.warning("arm %s produced no chunks; skipping", arm)
            continue

        avg_len = sum(c.n_chars for c in chunks) / len(chunks)
        strategy_mix: dict[str, int] = {}
        for c in chunks:
            strategy_mix[c.strategy] = strategy_mix.get(c.strategy, 0) + 1
        print(f"  chunks: {len(chunks):,}  avg chars: {avg_len:.0f}  mix: {strategy_mix}")

        store = QdrantStore(collection=collection)
        if args.reuse and store.exists(collection) and store.count(collection) == len(chunks):
            print("  reusing existing collection")
        else:
            t0 = now_ns()
            result = embedder.encode_passages(
                [c.text for c in chunks], show_progress=False
            )
            embed_ms = ns_to_ms(now_ns() - t0)
            store.create_collection(result.dim, recreate=True)
            store.upsert_chunks(chunks, result.dense, result.sparse, collection=collection)
            print(f"  embedded+indexed in {embed_ms / 1000:.1f}s")

        # ---- evaluate ----
        service = RetrievalService(store=store, collection=collection)
        service.reranker.load()
        acc_rrf = MetricAccumulator(f"{arm}:hybrid_rrf")
        acc_rr = MetricAccumulator(f"{arm}:hybrid_rerank")
        retrieval_ms: list[float] = []
        rerank_ms: list[float] = []

        for i, example in enumerate(examples, start=1):
            relevant = set(example.relevant_hashes)
            if not relevant:
                continue
            lat = LatencyBreakdown()
            embedding = service.embed_query(example.query, lat)

            t0 = now_ns()
            retrieval = service.retrieve(
                embedding,
                languages=[example.language],
                mode=RetrievalMode.LANGUAGE_FILTERED,
                latency=lat,
                limit=max(args.rerank_top_k, 10),
                collection=collection,
            )
            r_ms = ns_to_ms(now_ns() - t0)
            retrieval_ms.append(r_ms)
            acc_rrf.add(
                evaluate_ranking([c.content_hash or "" for c in retrieval.candidates], relevant),
                r_ms,
            )

            if retrieval.candidates:
                rl = LatencyBreakdown()
                t0 = now_ns()
                rr = service.rerank(
                    example.query, retrieval, latency=rl,
                    rerank_top_k=args.rerank_top_k, final_top_k=10, expand_parents=False,
                )
                rr_ms = ns_to_ms(now_ns() - t0)
                rerank_ms.append(rr_ms)
                acc_rr.add(
                    evaluate_ranking([c.content_hash or "" for c in rr.candidates], relevant),
                    rr_ms,
                )
            if i % 20 == 0:
                print(f"    ... {i}/{len(examples)}", flush=True)

        info = store.info(collection)
        vectors = store.count(collection)
        rrf_final = acc_rrf.finalize()
        rr_final = acc_rr.finalize()

        for label, final, lat_list in (
            ("hybrid_rrf", rrf_final, retrieval_ms),
            ("hybrid_rerank", rr_final, rerank_ms),
        ):
            if not final.queries:
                continue
            rows.append(
                {
                    "strategy": arm,
                    "ranking": label,
                    "queries": final.queries,
                    "recall@5": round(final.recall.get(5, float("nan")), 4),
                    "mrr": round(final.mrr, 4),
                    "ndcg@10": round(final.ndcg10, 4),
                    "vectors": vectors,
                    "avg_chunk_chars": round(avg_len, 1),
                    "latency_p50_ms": round(final.latency_p50_ms or 0, 2),
                    "latency_mean_ms": round(final.mean_latency_ms or 0, 2),
                }
            )

        details[arm] = {
            "collection": collection,
            "chunks": len(chunks),
            "vectors": vectors,
            "avg_chunk_chars": round(avg_len, 1),
            "chunk_seconds": round(chunk_ms / 1000, 2),
            "strategy_mix": strategy_mix,
            "vectors_per_passage": round(len(chunks) / max(1, len(parents)), 3),
            "qdrant_info": info,
            "hybrid_rrf": rrf_final.to_row(),
            "hybrid_rerank": rr_final.to_row() if rr_final.queries else None,
        }

        if not args.keep:
            store.delete_collection(collection)

    # ------------------------------------------------------------------ report
    print("\n" + "=" * 78)
    print("CHUNKING COMPARISON (measured)")
    print("=" * 78)
    print(write_markdown_table(rows))

    payload = {
        "base_corpus": base,
        "languages": languages,
        "split": split,
        "max_rows_per_language": args.max_rows_per_language,
        "parents": len(parents),
        "queries": len(examples),
        "rerank_top_k": args.rerank_top_k,
        "device": settings.resolved_device(),
        "int8_reranker": settings.int8_reranker_enabled(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rows": rows,
        "details": details,
        "notes": (
            "All arms share one deduplicated parent-passage set and one query "
            "set. Metrics computed on unique passage content hashes so extra "
            "chunks per passage confer no advantage."
        ),
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "chunking_eval.json", payload)
    write_csv(REPORTS_DIR / "chunking_eval.csv", rows)

    md = [
        "# Chunking Strategy Evaluation\n",
        f"- parents (shared by all arms): **{len(parents):,}**",
        f"- queries: **{len(examples)}** ({args.eval_split} split)",
        f"- languages: {', '.join(languages)}",
        f"- device: `{settings.resolved_device()}`, reranker top-k: {args.rerank_top_k}",
        "",
        "All arms are built from the identical parent set and scored on the",
        "identical queries, over unique passage content hashes.",
        "",
        write_markdown_table(rows),
        "\n## Index cost per arm\n",
        write_markdown_table(
            [
                {
                    "strategy": arm,
                    "vectors": d["vectors"],
                    "vectors_per_passage": d["vectors_per_passage"],
                    "avg_chunk_chars": d["avg_chunk_chars"],
                    "chunk_seconds": d["chunk_seconds"],
                }
                for arm, d in details.items()
            ]
        ),
    ]
    (REPORTS_DIR / "chunking_eval.md").write_text("\n".join(md), encoding="utf-8")
    print(f"reports -> {REPORTS_DIR / 'chunking_eval.*'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
