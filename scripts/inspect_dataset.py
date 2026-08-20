#!/usr/bin/env python
"""Inspect the real ai4bharat/MSMARCO-XI shards without downloading them.

Reads only the Parquet footer (a couple of HTTP range requests) to report row
counts, row-group layout and per-column byte sizes, then optionally streams a few
rows to show the actual record shape.

    python scripts/inspect_dataset.py
    python scripts/inspect_dataset.py --languages hi mr --split validation --rows 2

Why this script matters: the dataset README documents a config-based loading API
and a Telugu train file that do not exist in the repo as published. This prints
what is actually there.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import REPORTS_DIR, get_settings  # noqa: E402
from app.indexing.dataset_stream import MSMarcoXIStreamer  # noqa: E402
from app.languages import LANGUAGES, dataset_filename, has_split  # noqa: E402
from app.observability.tracing import setup_logging  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--languages", nargs="+", default=None)
    p.add_argument("--split", choices=("train", "validation"), default="validation")
    p.add_argument("--rows", type=int, default=1, help="Rows to stream per language (0 = none).")
    p.add_argument("--all-shards", action="store_true", help="Footer-inspect every language.")
    p.add_argument("--save", action="store_true", help="Write reports/dataset_inspection.json.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging()
    settings = get_settings()

    print("=" * 78)
    print(f"dataset: {settings.dataset_id}")
    print("=" * 78)

    print("\nUpstream availability (verified against the live repo tree):\n")
    print(f"  {'iso':<5} {'name':<12} {'file code':<10} {'train':<7} {'validation'}")
    print(f"  {'-'*5} {'-'*12} {'-'*10} {'-'*7} {'-'*10}")
    for iso, spec in LANGUAGES.items():
        if iso == "en":
            continue
        print(
            f"  {iso:<5} {spec.name:<12} {spec.file_code:<10} "
            f"{'yes' if spec.has_train else 'NO':<7} {'yes' if spec.has_validation else 'NO'}"
        )
    print(
        "\n  NOTE: Telugu (te) has NO train shard upstream despite the dataset\n"
        "        README listing 'teltrain.jsonl'. It exists only in validation.\n"
        "        The corpus builder falls back to validation for such languages."
    )

    languages = args.languages or (
        [k for k in LANGUAGES if k != "en"] if args.all_shards else settings.language_list
    )

    report: dict = {"dataset_id": settings.dataset_id, "split": args.split, "shards": {}}
    streamer = MSMarcoXIStreamer()

    for lang in languages:
        if not has_split(lang, args.split):
            print(f"\n--- {lang}/{args.split}: NOT AVAILABLE upstream (skipping) ---")
            continue
        print(f"\n--- {lang}/{args.split}: {dataset_filename(lang, args.split)} ---")
        try:
            info = streamer.inspect(lang, args.split)
        except Exception as exc:  # noqa: BLE001
            print(f"  footer read failed: {type(exc).__name__}: {exc}")
            continue

        print(f"  rows        : {info.num_rows:,}")
        print(f"  row groups  : {info.num_row_groups}")
        print(f"  uncompressed: {info.byte_size / 1e9:.2f} GB")
        needed = set(streamer.columns())
        total_needed = 0
        print("  columns (compressed):")
        for name, size in sorted(info.column_bytes.items(), key=lambda kv: -kv[1]):
            mark = ""
            if any(name.startswith(c) for c in needed):
                mark = "  <- read"
                total_needed += size
            if size > 1e6 or mark:
                print(f"    {name:<46} {size / 1e6:9.1f} MB{mark}")
        total_all = sum(info.column_bytes.values())
        print(
            f"  projection saves: {(1 - total_needed / total_all) * 100:.0f}% "
            f"({total_needed / 1e6:.0f} of {total_all / 1e6:.0f} MB)"
        )
        if info.num_row_groups == 1:
            print(
                "  NOTE: single row group. Bounded-memory streaming relies on\n"
                "        pre_buffer=False page-level reads (see dataset_stream.py)."
            )
        report["shards"][lang] = info.to_json()

        if args.rows:
            print(f"  streaming {args.rows} row(s):")
            try:
                for row in streamer.stream(lang, args.split, max_rows=args.rows):
                    print(f"    query_id   : {row.query_id}")
                    print(f"    query_type : {row.query_type}")
                    print(f"    query      : {row.query[:90]}")
                    print(f"    Answer     : {row.answer[:90]}   <- GROUND TRUTH, not indexed")
                    print(f"    passages   : {len(row.passages)}")
                    print(f"    is_selected: {row.is_selected}   <- LABEL, not indexed")
                    lens = [len(p) for p in row.passages]
                    print(f"    lengths    : {lens}")
                    print(f"    passage[0] : {row.passages[0][:110]}")
            except Exception as exc:  # noqa: BLE001
                print(f"    stream failed: {type(exc).__name__}: {exc}")

    if args.save:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        target = REPORTS_DIR / "dataset_inspection.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nsaved -> {target}")

    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
