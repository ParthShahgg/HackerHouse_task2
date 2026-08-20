#!/usr/bin/env python
"""Offline ingestion: stream MSMARCO-XI -> dedup -> chunk -> embed -> Qdrant.

Nothing in this script runs on the query path. All chunking and embedding of the
corpus happens here, which is precisely what makes the online path fast enough to
be latency-engineered.

Examples
--------
Default demo corpus (bounded, 4 languages)::

    python scripts/build_index.py --rebuild

Fast local iteration::

    python scripts/build_index.py --mode dev --rebuild

Explicit selection::

    python scripts/build_index.py --languages hi mr ta te --split validation \\
        --max-rows-per-language 1200 --rebuild

Large offline run (see configs/full.yaml before doing this)::

    python scripts/build_index.py --full-mode --languages hi --max-rows-per-language 500000

Spread a small sample across the whole shard rather than taking the head::

    python scripts/build_index.py --sample-mode --max-rows-per-language 300

Scaling the corpus is a flag/config change. No application code is involved.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.chunking import ALL_STRATEGIES, ChunkerContext, ChunkingEngine  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.indexing.corpus import CorpusBuilder, CorpusPaths, verify_no_leakage  # noqa: E402
from app.languages import SUPPORTED_ISO1  # noqa: E402
from app.observability.tracing import get_logger, setup_logging  # noqa: E402

logger = get_logger("build_index")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode", choices=("dev", "demo", "full"),
        help="Corpus profile from configs/<mode>.yaml. Default: INGEST_MODE env.",
    )
    p.add_argument(
        "--languages", nargs="+", metavar="ISO",
        help=f"ISO-639-1 codes. Supported: {', '.join(SUPPORTED_ISO1)}",
    )
    p.add_argument("--split", choices=("train", "validation"), help="Upstream split.")
    p.add_argument(
        "--max-rows-per-language", type=int, metavar="N",
        help="Row cap per language. Omit for no cap (full shard).",
    )
    p.add_argument(
        "--sample-mode", action="store_true",
        help="Spread the sample across the shard (stride) instead of taking the head.",
    )
    p.add_argument(
        "--full-mode", action="store_true",
        help="Shorthand for --mode full (large offline ingestion).",
    )
    p.add_argument(
        "--rebuild", action="store_true",
        help="Delete and recreate the Qdrant collection first.",
    )
    p.add_argument(
        "--strategies", nargs="+", choices=ALL_STRATEGIES, metavar="S",
        help=f"Chunking strategies to index. Default from config. Options: {', '.join(ALL_STRATEGIES)}",
    )
    p.add_argument("--collection", help="Override the Qdrant collection name.")
    p.add_argument(
        "--include-english", action="store_true",
        help="Also index the English source passages as language='en' for cross-lingual fallback.",
    )
    p.add_argument(
        "--no-upload", action="store_true",
        help="Build and persist the corpus locally, skip embedding/upload.",
    )
    p.add_argument(
        "--reuse-corpus", action="store_true",
        help="Skip streaming; re-embed the previously persisted corpus.",
    )
    p.add_argument("--batch-size", type=int, help="Embedding batch size override.")
    return p.parse_args()


def human(n: float) -> str:
    return f"{n:,.0f}"


def print_stats_table(stats: dict) -> None:
    """Print the per-language ingestion report."""
    headers = [
        "language", "rows", "passages", "unique parents", "dupes removed",
        "child chunks", "vectors", "avg chars", "stream s", "chunk s",
    ]
    rows = []
    for lang, st in stats.items():
        data = st.to_json() if hasattr(st, "to_json") else st
        rows.append([
            lang,
            human(data["rows_processed"]),
            human(data["passages_seen"]),
            human(data["unique_parents"]),
            human(data["duplicates_removed"]),
            human(data["child_chunks"]),
            human(data["vectors_generated"]),
            f"{data['avg_chunk_chars']:.0f}",
            f"{data['stream_seconds']:.1f}",
            f"{data['chunk_seconds']:.1f}",
        ])

    try:
        from tabulate import tabulate

        print("\n" + tabulate(rows, headers=headers, tablefmt="github"))
    except ModuleNotFoundError:
        print("\n" + " | ".join(headers))
        for row in rows:
            print(" | ".join(str(c) for c in row))

    print("\nchunks by strategy:")
    for lang, st in stats.items():
        data = st.to_json() if hasattr(st, "to_json") else st
        print(f"  {lang}: {data['chunks_by_strategy']}")


def main() -> int:
    args = parse_args()
    setup_logging()

    import os

    mode = "full" if args.full_mode else args.mode
    if mode:
        # Set before get_settings() is memoised so the YAML profile applies.
        os.environ["INGEST_MODE"] = mode
        from app.config import reset_settings_cache

        reset_settings_cache()

    settings = get_settings()
    collection = args.collection or settings.qdrant_collection
    languages = args.languages or settings.language_list
    split = args.split or settings.dataset_split
    max_rows = (
        args.max_rows_per_language
        if args.max_rows_per_language is not None
        else settings.max_rows_per_language
    )
    strategies = args.strategies or settings.strategy_list

    unknown = set(languages) - set(SUPPORTED_ISO1)
    if unknown:
        logger.error("unsupported languages: %s", sorted(unknown))
        return 2

    print("=" * 78)
    print(f"  dataset      : {settings.dataset_id}")
    print(f"  mode         : {settings.ingest_mode}")
    print(f"  split        : {split}")
    print(f"  languages    : {', '.join(languages)}")
    print(f"  max rows/lang: {max_rows if max_rows is not None else 'ALL (full shard)'}")
    print(f"  strategies   : {', '.join(strategies)}")
    print(f"  collection   : {collection}")
    print(f"  device       : {settings.resolved_device()} (fp16={settings.fp16_enabled()})")
    if max_rows is None:
        print("\n  NOTE: no row cap. MSMARCO-XI train shards are ~3.7GB each and the")
        print("        full corpus is ~11.45M rows / 55.6GB. Streaming keeps memory")
        print("        bounded, but embedding time scales linearly - see README.")
    print("=" * 78)

    t_start = time.perf_counter()

    # ---- wire the embedder into chunking (semantic split + real tokenizer) ----
    embedder = None
    ctx = ChunkerContext.from_settings(settings)
    if not args.no_upload:
        from app.retrieval.embedder import get_embedder

        embedder = get_embedder()
        embedder.load()
        # Semantic splitting needs sentence embeddings; fixed fallback needs the
        # real XLM-R tokenizer so chunk boundaries match what the encoder sees.
        ctx.embed_sentences = embedder.embed_sentences
        ctx.count_tokens = embedder.count_tokens
        ctx.encode_tokens = embedder.encode_token_ids
        ctx.decode_tokens = embedder.decode_token_ids
    else:
        logger.warning(
            "--no-upload: semantic_split will fall back to sentence windows and "
            "fixed_fallback will use character windows (no tokenizer loaded)"
        )

    engine = ChunkingEngine(ctx=ctx, enabled=strategies)

    # ------------------------------- build corpus -----------------------------
    if args.reuse_corpus and CorpusPaths(collection).exists():
        from app.indexing.corpus import load_chunks, load_stats

        logger.info("reusing persisted corpus for %s", collection)
        chunks = load_chunks(collection)
        stats = load_stats(collection).get("per_language", {})
    else:
        builder = CorpusBuilder(
            name=collection,
            split=split,
            languages=languages,
            max_rows_per_language=max_rows,
            sample_stride=7 if args.sample_mode else 1,
            strategies=strategies,
            include_english=args.include_english,
            engine=engine,
        )
        chunks, _examples, stats_objs = builder.build()
        stats = stats_objs

    if not chunks:
        logger.error("no chunks produced; aborting")
        return 1

    print(f"\n  total chunks to index: {human(len(chunks))}")

    if args.no_upload:
        print_stats_table(stats)
        print(f"\n  corpus persisted (no upload). elapsed {time.perf_counter() - t_start:.1f}s")
        return 0

    # --------------------------------- embed ----------------------------------
    from app.retrieval.store import QdrantStore

    assert embedder is not None
    batch_size = args.batch_size or settings.embed_batch_size
    print(f"\n  embedding {human(len(chunks))} chunks (batch={batch_size})...")
    t_embed = time.perf_counter()
    result = embedder.encode_passages(
        [c.text for c in chunks], batch_size=batch_size, show_progress=True
    )
    embed_seconds = time.perf_counter() - t_embed
    rate = len(chunks) / embed_seconds if embed_seconds else 0.0
    print(f"  embedded in {embed_seconds:.1f}s ({rate:.1f} chunks/s), dim={result.dim}")

    # ---------------------------- create + upload -----------------------------
    store = QdrantStore(collection=collection)
    store.create_collection(result.dim, recreate=args.rebuild)

    print(f"  uploading to Qdrant collection '{collection}'...")
    t_upload = time.perf_counter()
    written = store.upsert_chunks(chunks, result.dense, result.sparse, collection=collection)
    upload_seconds = time.perf_counter() - t_upload
    print(f"  uploaded {human(written)} points in {upload_seconds:.1f}s")

    # ---- fold measured timings back into the per-language stats ----
    total_chunks = len(chunks)
    for lang, st in stats.items():
        if hasattr(st, "to_json"):
            share = st.total_chunks / total_chunks if total_chunks else 0
            st.vectors_generated = st.total_chunks
            st.embed_seconds = round(embed_seconds * share, 2)
            st.upload_seconds = round(upload_seconds * share, 2)

    print_stats_table(stats)

    info = store.info(collection)
    print(f"\n  Qdrant collection size: {info}")
    print(f"  exact point count     : {human(store.count(collection))}")

    # ------------------------------ leakage audit ------------------------------
    print("\n  leakage audit (queries/answers must NOT be in the index):")
    try:
        audit = verify_no_leakage(collection)
        for key in ("chunks", "queries_checked", "answers_checked"):
            print(f"    {key:28s}: {audit[key]}")
        # Print COUNTS, not the offending text. The text is multi-script and
        # printing it is both a console-encoding hazard and noise; the JSON
        # report keeps the samples for debugging.
        print(f"    {'query text in index':28s}: {len(audit['query_text_in_index'])}")
        print(f"    {'answer LEAKED into index':28s}: {len(audit['answer_text_in_index'])}")
        print(f"    {'query+answer concatenation':28s}: {len(audit['query_answer_concatenation'])}")
        print(f"    {'forbidden payload fields':28s}: {audit['forbidden_payload_fields']}")
        print(
            f"    {'benign extractive overlap':28s}: {audit['extractive_overlap_benign']}"
            "  (Answer copied verbatim from a gold passage - dataset property)"
        )
        print(f"    {'CLEAN':28s}: {audit['clean']}")
        if not audit["clean"]:
            logger.error("LEAKAGE DETECTED - the index contains query or answer text")
            return 1
    except Exception as exc:  # noqa: BLE001
        logger.warning("leakage audit skipped: %s", exc)

    total = time.perf_counter() - t_start
    print(f"\n  TOTAL indexing time: {total:.1f}s")
    print("=" * 78)

    summary_path = CorpusPaths(collection).corpus_dir / "index_report.json"
    summary_path.write_text(
        json.dumps(
            {
                "collection": collection,
                "mode": settings.ingest_mode,
                "split": split,
                "languages": languages,
                "strategies": strategies,
                "max_rows_per_language": max_rows,
                "chunks": total_chunks,
                "points_written": written,
                "dim": result.dim,
                "embed_seconds": round(embed_seconds, 2),
                "embed_chunks_per_s": round(rate, 2),
                "upload_seconds": round(upload_seconds, 2),
                "total_seconds": round(total, 2),
                "device": settings.resolved_device(),
                "qdrant_info": info,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    print(f"  report -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
