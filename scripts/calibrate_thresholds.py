#!/usr/bin/env python
"""Calibrate the abstention thresholds empirically.

Why this script exists
----------------------
Picking a number like ``0.5`` for "is this evidence good enough" is guessing.
Reranker logits are unbounded and their useful operating point depends on the
model, the quantization setting, the corpus and the languages. So the thresholds
are *fitted*, and the fitted artefact records the exact configuration it was
fitted under.

Labelling
---------
The target is the question that actually matters at serving time: **will the
top-ranked passage support an answer?**

positive
    Top reranked passage is a gold (``is_selected``) passage for that query.
negative
    It is not - either retrieval failed, or the query is genuinely unanswerable.

Genuine out-of-corpus negatives are essential and are constructed honestly: we
stream *additional* dataset rows beyond those ingested, so their gold passages
were never indexed. Any such query whose gold passage happens to be present
anyway (duplicate passage) is discarded, so negatives are truly unanswerable.
Without this, every query would be answerable and the fitted threshold would be
meaningless.

Data hygiene
------------
Fitting uses only the ``calibration`` slice of the deterministic query split.
``scripts/evaluate_retrieval.py`` reports on the disjoint ``test`` slice, so no
threshold is ever tuned on the queries it is scored against.

    python scripts/calibrate_thresholds.py
    python scripts/calibrate_thresholds.py --target-precision 0.9
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CONFIGS_DIR, REPORTS_DIR, get_settings  # noqa: E402
from app.evaluation.report import write_json, write_markdown_table  # noqa: E402
from app.indexing.corpus import load_eval_examples, load_parents  # noqa: E402
from app.observability.tracing import get_logger, setup_logging  # noqa: E402
from app.schemas.common import LatencyBreakdown, RetrievalMode  # noqa: E402

logger = get_logger("calibrate")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection", help="Qdrant collection (default: config).")
    p.add_argument("--limit", type=int, default=140, help="Max in-corpus calibration queries.")
    p.add_argument("--negatives", type=int, default=90, help="Out-of-corpus negative queries.")
    p.add_argument(
        "--target-precision", type=float, default=0.85,
        help="Minimum precision required of the GENERATE decision; the threshold "
             "is the lowest one meeting it, maximising recall subject to that.",
    )
    p.add_argument("--out", default=None, help="Threshold JSON path (default: configs/thresholds.json).")
    return p.parse_args()


def _sweep(scores: list[tuple[float, bool]]) -> list[dict]:
    """Precision/recall/F1 at every candidate threshold."""
    if not scores:
        return []
    positives = sum(1 for _, label in scores if label)
    candidates = sorted({round(s, 3) for s, _ in scores})
    rows: list[dict] = []
    for th in candidates:
        tp = sum(1 for s, label in scores if s >= th and label)
        fp = sum(1 for s, label in scores if s >= th and not label)
        fn = positives - tp
        precision = tp / (tp + fp) if (tp + fp) else 1.0
        recall = tp / positives if positives else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        # Youden's J = sensitivity + specificity - 1
        negatives = len(scores) - positives
        tn = negatives - fp
        specificity = tn / negatives if negatives else 0.0
        rows.append(
            {
                "threshold": th, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": round(precision, 4), "recall": round(recall, 4),
                "f1": round(f1, 4), "youden_j": round(recall + specificity - 1, 4),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    setup_logging()
    settings = get_settings()
    collection = args.collection or settings.qdrant_collection

    from app.indexing.dataset_stream import MSMarcoXIStreamer
    from app.indexing.normalize import content_hash
    from app.retrieval.service import RetrievalService

    positives_examples = load_eval_examples(
        collection, eval_split="calibration", with_labels_only=True
    )[: args.limit]
    if not positives_examples:
        logger.error("no calibration queries for %r - build the index first", collection)
        return 1

    parents = load_parents(collection)
    indexed_hashes = {p.content_hash for p in parents}
    languages = sorted({p.language for p in parents if p.language != "en"})

    print("=" * 78)
    print("THRESHOLD CALIBRATION")
    print(f"  collection            : {collection}")
    print(f"  in-corpus queries     : {len(positives_examples)} (calibration split)")
    print(f"  indexed passages      : {len(indexed_hashes):,}")
    print(f"  device                : {settings.resolved_device()}")
    print(f"  int8 reranker         : {settings.int8_reranker_enabled()}")
    print(f"  target precision      : {args.target_precision}")
    print("=" * 78)

    service = RetrievalService(collection=collection)
    service.embedder.load()
    service.reranker.load()

    final_top_k = settings.final_top_k
    samples: list[dict] = []

    def probe(query: str, language: str, relevant: set[str], kind: str) -> None:
        lat = LatencyBreakdown()
        try:
            embedding = service.embed_query(query, lat)
            retrieval = service.retrieve(
                embedding, languages=[language],
                mode=RetrievalMode.LANGUAGE_FILTERED, latency=lat,
                collection=collection,
            )
            if not retrieval.candidates:
                samples.append(
                    {"kind": kind, "language": language, "top_score": None,
                     "margin": None, "label": False, "reason": "no_candidates"}
                )
                return
            rerank = service.rerank(
                query, retrieval, latency=lat, final_top_k=final_top_k, expand_parents=False
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("probe failed: %s", exc)
            return
        if not rerank.candidates:
            return
        top = rerank.candidates[0]

        # LABEL: does the context the generator will actually receive contain a
        # gold passage?
        #
        # Labelling on top-1 alone is the wrong target here, for two reasons:
        #   1. generation reads the top FINAL_TOP_K parents, not just rank 1;
        #   2. MS MARCO judgements are sparse - typically 1 of ~10 passages is
        #      marked is_selected - so a genuinely relevant passage ranked first
        #      is frequently unlabelled, making a top-1 label pessimistic by
        #      construction rather than by retrieval quality.
        top_k_hashes = [
            c.content_hash for c in rerank.candidates[:final_top_k] if c.content_hash
        ]
        samples.append(
            {
                "kind": kind,
                "language": language,
                "top_score": float(top.rerank_score),
                "margin": rerank.margin,
                "label": bool(relevant.intersection(top_k_hashes)),
                "top1_is_gold": bool(top.content_hash and top.content_hash in relevant),
            }
        )

    print("\nprobing in-corpus queries...")
    for i, example in enumerate(positives_examples, start=1):
        probe(example.query, example.language, set(example.relevant_hashes), "in_corpus")
        if i % 20 == 0:
            print(f"  ... {i}/{len(positives_examples)}", flush=True)

    # ---------------- genuine out-of-corpus negatives ----------------
    print("\nstreaming out-of-corpus negatives (rows beyond those ingested)...")
    ingested_rows = settings.max_rows_per_language or 0
    per_language = max(1, args.negatives // max(1, len(languages)))
    negative_count = 0
    for language in languages:
        streamer = MSMarcoXIStreamer()
        try:
            for row in streamer.stream(
                language,
                settings.dataset_split,
                max_rows=per_language * 3,
                # Start past the ingested prefix so these passages are NOT indexed.
                skip_rows=ingested_rows + 50,
            ):
                gold = {
                    content_hash(language, text)
                    for text, sel in zip(row.passages, row.is_selected, strict=False)
                    if sel == 1
                }
                if not gold:
                    continue
                # Discard if the gold passage is in the index anyway (duplicate);
                # otherwise this would not be a true negative.
                if gold & indexed_hashes:
                    continue
                probe(row.query, language, gold, "out_of_corpus")
                negative_count += 1
                if negative_count >= (per_language * (languages.index(language) + 1)):
                    break
        except Exception as exc:  # noqa: BLE001
            logger.warning("negative streaming failed for %s: %s", language, exc)
        print(f"  {language}: negatives so far {negative_count}", flush=True)

    scored = [(s["top_score"], s["label"]) for s in samples if s["top_score"] is not None]
    if not scored:
        logger.error("no usable samples")
        return 1

    n_pos = sum(1 for _, label in scored if label)
    n_neg = len(scored) - n_pos
    print(f"\nsamples: {len(scored)}  positive={n_pos}  negative={n_neg}")
    if n_pos == 0 or n_neg == 0:
        logger.error(
            "calibration needs both classes (got pos=%d neg=%d); cannot fit a threshold",
            n_pos, n_neg,
        )
        return 1

    sweep = _sweep(scored)

    # Lowest threshold meeting the precision floor => best recall subject to it.
    # Rationale: a false GENERATE is worse than a false ABSTAIN here, because an
    # abstention is honest whereas answering from irrelevant evidence is not.
    eligible = [r for r in sweep if r["precision"] >= args.target_precision and r["tp"] > 0]
    best_f1 = max(sweep, key=lambda r: r["f1"])
    best_j = max(sweep, key=lambda r: r["youden_j"])

    precision_floor_met = bool(eligible)
    if precision_floor_met:
        chosen = min(eligible, key=lambda r: r["threshold"])
        objective = f"lowest threshold with precision >= {args.target_precision}"
    else:
        # Must be stated, not silently substituted. Reporting the requested
        # objective while actually using a different one would make the artefact
        # claim a precision guarantee the data does not support.
        chosen = best_f1
        objective = (
            f"max F1 (FALLBACK: no threshold reached the requested precision "
            f"floor of {args.target_precision}; best achievable precision was "
            f"{max(r['precision'] for r in sweep if r['tp'] > 0):.3f})"
        )
        logger.warning(
            "precision floor %.2f unreachable on this data; falling back to max-F1 "
            "at precision %.3f. This is recorded in the artefact.",
            args.target_precision, chosen["precision"],
        )

    # Margin threshold: 10th percentile of margins among *correct* generates, so
    # the ambiguity check rarely fires on genuinely confident cases.
    correct_margins = sorted(
        s["margin"] for s in samples
        if s["label"] and s["margin"] is not None and s["top_score"] is not None
        and s["top_score"] >= chosen["threshold"]
    )
    margin_min = round(correct_margins[max(0, int(0.10 * len(correct_margins)) - 1)], 3) if correct_margins else 0.0
    margin_min = max(0.0, min(margin_min, 1.0))

    print("\nthreshold sweep (subset):")
    show = [r for i, r in enumerate(sweep) if i % max(1, len(sweep) // 14) == 0]
    print(write_markdown_table(show))
    print(f"\n  chosen (precision >= {args.target_precision}) : {chosen}")
    print(f"  best F1                            : {best_f1}")
    print(f"  best Youden J                      : {best_j}")
    print(f"  margin_min (p10 of correct)        : {margin_min}")

    artifact = {
        "calibrated": True,
        "source": f"scripts/calibrate_thresholds.py @ {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        "rerank_abstain_below": chosen["threshold"],
        "rerank_margin_min": margin_min,
        "margin_override_score": round(max(chosen["threshold"] + 4.0, 6.0), 3),
        "objective": objective,
        "requested_target_precision": args.target_precision,
        "precision_floor_met": precision_floor_met,
        "label_definition": (
            "a gold (is_selected) passage appears within the top FINAL_TOP_K "
            "reranked candidates - i.e. in the context the generator receives"
        ),
        "chosen_operating_point": chosen,
        "best_f1_operating_point": best_f1,
        "best_youden_operating_point": best_j,
        "fitted_on": {
            "collection": collection,
            "eval_split": "calibration",
            "in_corpus_queries": sum(1 for s in samples if s["kind"] == "in_corpus"),
            "out_of_corpus_queries": sum(1 for s in samples if s["kind"] == "out_of_corpus"),
            "positives": n_pos,
            "negatives": n_neg,
            "languages": languages,
        },
        # Thresholds are only valid for the configuration they were fitted under.
        "model_config": {
            "reranker_model": settings.reranker_model,
            "reranker_revision": settings.reranker_model_revision,
            "int8_quantized": settings.int8_reranker_enabled(),
            "device": settings.resolved_device(),
            "rerank_top_k": settings.rerank_top_k,
            "rerank_max_length": settings.rerank_max_length,
        },
        "warning": (
            "Reranker logits shift with quantization/precision. Re-run this "
            "script if reranker_model, int8_quantized or device changes."
        ),
    }

    out_path = Path(args.out) if args.out else CONFIGS_DIR / "thresholds.json"
    write_json(out_path, artifact)
    print(f"\nthresholds -> {out_path}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(REPORTS_DIR / "calibration.json", {"artifact": artifact, "sweep": sweep, "samples": samples})
    md = [
        "# Abstention Threshold Calibration\n",
        f"- collection: `{collection}`",
        f"- fitted on the **calibration** split only (test split is held out for reporting)",
        f"- in-corpus queries: {artifact['fitted_on']['in_corpus_queries']}",
        f"- out-of-corpus queries: {artifact['fitted_on']['out_of_corpus_queries']}"
        " (gold passages verified absent from the index)",
        f"- positives: {n_pos}, negatives: {n_neg}",
        f"- reranker: `{settings.reranker_model}` int8=`{settings.int8_reranker_enabled()}`"
        f" device=`{settings.resolved_device()}`",
        "",
        "**Label**: a gold (`is_selected`) passage appears within the top "
        f"{final_top_k} reranked candidates - i.e. inside the context the generator",
        "actually receives. Labelling on rank 1 alone would be pessimistic by",
        "construction, because MS MARCO marks only ~1 of ~10 passages as selected,",
        "so a genuinely relevant top hit is frequently unlabelled.",
        "",
        f"**Chosen**: `rerank_abstain_below = {chosen['threshold']}`, "
        f"`rerank_margin_min = {margin_min}`",
        "",
        f"**Objective actually used**: {objective}",
        "",
        (
            f"Requested precision floor `{args.target_precision}` was "
            f"{'MET' if precision_floor_met else '**NOT MET**'}."
            + ("" if precision_floor_met else
               " The artefact records this rather than claiming a guarantee the data"
               " does not support.")
        ),
        "",
        "Rationale for preferring a precision floor: a false GENERATE is worse than",
        "a false ABSTAIN - abstaining is honest, answering from irrelevant evidence",
        "is not.",
        "",
        "## Threshold sweep\n",
        write_markdown_table(show),
    ]
    (REPORTS_DIR / "calibration.md").write_text("\n".join(md), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
