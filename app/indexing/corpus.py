"""Corpus builder: stream -> normalise -> dedup -> chunk -> persist.

Produces two strictly separated artefacts:

``data/corpus/<name>/chunks.jsonl``
    Retrieval documents. Passage text + provenance only.

``data/eval/<name>/queries.jsonl``
    Evaluation labels: the query, the ground-truth ``Answer``, and the
    ``is_selected`` judgements expressed as *content hashes* of the relevant
    passages.

The label file is never read by the serving path - only by
``scripts/evaluate_retrieval.py`` and ``scripts/calibrate_thresholds.py``. This
is the structural guarantee against answer leakage and retrieval-label leakage.

Evaluation split
----------------
Queries are partitioned deterministically into ``calibration`` and ``test`` by
hashing ``query_id``. Thresholds are fitted on ``calibration`` and every
reported metric comes from ``test``, so no threshold is ever tuned on the
queries it is scored against.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.chunking import ChunkingEngine
from app.config import DATA_DIR, get_settings
from app.indexing.dataset_stream import MSMarcoXIStreamer
from app.indexing.deduplicate import PassageDeduplicator
from app.indexing.normalize import content_hash
from app.indexing.records import Chunk, DatasetRow, EvalExample, IngestStats, ParentPassage
from app.languages import has_split
from app.observability.tracing import get_logger

logger = get_logger(__name__)

__all__ = [
    "CorpusBuilder",
    "CorpusPaths",
    "eval_split_of",
    "load_chunks",
    "load_eval_examples",
    "load_parents",
    "load_stats",
]

CALIBRATION_FRACTION = 0.4


def eval_split_of(query_id: int, calibration_fraction: float = CALIBRATION_FRACTION) -> str:
    """Deterministic calibration/test assignment.

    Hash-based rather than random so the split is stable across runs, machines
    and corpus sizes without persisting a seed.
    """
    digest = hashlib.sha256(f"msmarco-xi-split:{query_id}".encode()).digest()
    bucket = int.from_bytes(digest[:4], "big") / 0xFFFFFFFF
    return "calibration" if bucket < calibration_fraction else "test"


class CorpusPaths:
    """Filesystem layout for one named corpus build."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.corpus_dir = DATA_DIR / "corpus" / name
        self.eval_dir = DATA_DIR / "eval" / name

    @property
    def chunks(self) -> Path:
        return self.corpus_dir / "chunks.jsonl"

    @property
    def parents(self) -> Path:
        return self.corpus_dir / "parents.jsonl"

    @property
    def stats(self) -> Path:
        return self.corpus_dir / "stats.json"

    @property
    def queries(self) -> Path:
        return self.eval_dir / "queries.jsonl"

    def ensure(self) -> None:
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.eval_dir.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.chunks.exists() and self.parents.exists()


class CorpusBuilder:
    """Builds the offline corpus for a set of languages."""

    def __init__(
        self,
        *,
        name: str | None = None,
        split: str | None = None,
        languages: Iterable[str] | None = None,
        max_rows_per_language: int | None = None,
        sample_stride: int = 1,
        strategies: Iterable[str] | None = None,
        include_english: bool = False,
        engine: ChunkingEngine | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.name = name or settings.qdrant_collection
        self.split = split or settings.dataset_split
        self.languages = list(languages or settings.language_list)
        self.max_rows_per_language = (
            max_rows_per_language
            if max_rows_per_language is not None
            else settings.max_rows_per_language
        )
        self.sample_stride = max(1, sample_stride)
        self.strategies = list(strategies or settings.strategy_list)
        self.include_english = include_english
        self.engine = engine or ChunkingEngine(enabled=self.strategies)
        self.paths = CorpusPaths(self.name)

    # ------------------------------------------------------------------ helpers
    def resolve_languages(self) -> list[tuple[str, str]]:
        """Pair each language with a split that actually exists upstream.

        Telugu has no ``train`` shard, so when ``--split train`` is requested for
        Telugu we transparently fall back to ``validation`` and say so, rather
        than crashing or silently dropping the language.
        """
        resolved: list[tuple[str, str]] = []
        for lang in self.languages:
            if has_split(lang, self.split):
                resolved.append((lang, self.split))
                continue
            if has_split(lang, "validation"):
                logger.warning(
                    "language %r has no %r shard upstream; falling back to 'validation'",
                    lang, self.split,
                )
                resolved.append((lang, "validation"))
            else:
                logger.error("language %r unavailable in any split; skipping", lang)
        return resolved

    # -------------------------------------------------------------------- build
    def build(self) -> tuple[list[Chunk], list[EvalExample], dict[str, IngestStats]]:
        """Stream, dedup and chunk. Returns chunks, eval labels and per-language stats."""
        all_chunks: list[Chunk] = []
        all_parents: list[ParentPassage] = []
        all_examples: list[EvalExample] = []
        stats_by_lang: dict[str, IngestStats] = {}

        for language, split in self.resolve_languages():
            chunks, parents, examples, stats = self._build_language(language, split)
            all_chunks.extend(chunks)
            all_parents.extend(parents)
            all_examples.extend(examples)
            stats_by_lang[language] = stats

        self._persist(all_chunks, all_parents, all_examples, stats_by_lang)
        return all_chunks, all_examples, stats_by_lang

    def _build_language(
        self, language: str, split: str
    ) -> tuple[list[Chunk], list[ParentPassage], list[EvalExample], IngestStats]:
        stats = IngestStats(language=language, split=split)
        dedup = PassageDeduplicator()
        examples: list[EvalExample] = []

        streamer = MSMarcoXIStreamer(include_english=self.include_english)
        t_stream = time.perf_counter()
        rows: list[DatasetRow] = []
        for row in streamer.stream(
            language,
            split,
            max_rows=self.max_rows_per_language,
            sample_stride=self.sample_stride,
        ):
            rows.append(row)
            if len(rows) % 250 == 0:
                logger.info("  %s: streamed %d rows", language, len(rows))
        stats.stream_seconds = round(time.perf_counter() - t_stream, 2)
        stats.rows_processed = len(rows)

        # --- dedup + label extraction -------------------------------------
        for row in rows:
            hashes: list[str] = []
            relevant: list[str] = []
            for text, selected in zip(row.passages, row.is_selected, strict=False):
                chash, _ = dedup.add(
                    text, language=language, source_split=split, query_id=row.query_id
                )
                if chash is None:
                    continue
                hashes.append(chash)
                if selected == 1:
                    relevant.append(chash)

            examples.append(
                EvalExample(
                    query_id=row.query_id,
                    language=language,
                    split=split,
                    query=row.query,
                    answer=row.answer,
                    query_type=row.query_type,
                    relevant_hashes=sorted(set(relevant)),
                    candidate_hashes=sorted(set(hashes)),
                )
            )

            # Optional English representation for cross-lingual fallback. Indexed
            # as language="en" so it is a separate retrieval unit, never mixed
            # into the target-language namespace.
            if self.include_english and row.english_passages:
                dedup_en_lang = "en"
                for text in row.english_passages:
                    dedup.add(
                        text,
                        language=dedup_en_lang,
                        source_split=split,
                        query_id=row.query_id,
                    )

        stats.passages_seen = dedup.passages_seen
        stats.duplicates_removed = dedup.duplicates_removed
        stats.empty_skipped = dedup.empty_skipped
        stats.unique_parents = dedup.unique_count
        stats.eval_examples = len(examples)
        stats.eval_examples_with_labels = sum(1 for e in examples if e.has_label)

        # --- chunking ------------------------------------------------------
        parents = dedup.parents
        t_chunk = time.perf_counter()
        chunks: list[Chunk] = []
        by_strategy: dict[str, int] = {}
        for parent in parents:
            produced = self.engine.chunk(parent)
            chunks.extend(produced)
            for chunk in produced:
                by_strategy[chunk.strategy] = by_strategy.get(chunk.strategy, 0) + 1
        stats.chunk_seconds = round(time.perf_counter() - t_chunk, 2)
        stats.chunks_by_strategy = by_strategy
        stats.child_chunks = sum(
            count for name, count in by_strategy.items() if name != "native"
        )
        # `unique_parents` counts native chunks; keep the two consistent so
        # total_chunks is exactly len(chunks).
        stats.unique_parents = by_strategy.get("native", len(parents))
        stats.total_chars = sum(c.n_chars for c in chunks)

        logger.info(
            "%s/%s: rows=%d passages=%d unique=%d dupes=%d chunks=%d %s",
            language, split, stats.rows_processed, stats.passages_seen,
            stats.unique_parents, stats.duplicates_removed, len(chunks), by_strategy,
        )
        return chunks, parents, examples, stats

    # ------------------------------------------------------------------ persist
    def _persist(
        self,
        chunks: list[Chunk],
        parents: list[ParentPassage],
        examples: list[EvalExample],
        stats: dict[str, IngestStats],
    ) -> None:
        self.paths.ensure()

        with self.paths.chunks.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

        with self.paths.parents.open("w", encoding="utf-8") as fh:
            for parent in parents:
                fh.write(json.dumps(asdict(parent), ensure_ascii=False) + "\n")

        with self.paths.queries.open("w", encoding="utf-8") as fh:
            for example in examples:
                record = example.to_json()
                record["eval_split"] = eval_split_of(example.query_id)
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

        summary: dict[str, Any] = {
            "name": self.name,
            "dataset_id": self.settings.dataset_id,
            "requested_split": self.split,
            "languages": self.languages,
            "strategies": self.strategies,
            "max_rows_per_language": self.max_rows_per_language,
            "sample_stride": self.sample_stride,
            "include_english": self.include_english,
            "totals": {
                "chunks": len(chunks),
                "parents": len(parents),
                "eval_examples": len(examples),
                "eval_with_labels": sum(1 for e in examples if e.has_label),
            },
            "per_language": {k: v.to_json() for k, v in stats.items()},
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.paths.stats.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "persisted %d chunks / %d eval queries to %s",
            len(chunks), len(examples), self.paths.corpus_dir,
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build the corpus first: python scripts/build_index.py"
        )
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_chunks(name: str) -> list[Chunk]:
    return [Chunk(**record) for record in _iter_jsonl(CorpusPaths(name).chunks)]


def load_parents(name: str) -> list[ParentPassage]:
    return [ParentPassage(**record) for record in _iter_jsonl(CorpusPaths(name).parents)]


def load_eval_examples(
    name: str,
    *,
    eval_split: str | None = None,
    languages: Iterable[str] | None = None,
    with_labels_only: bool = True,
) -> list[EvalExample]:
    """Load evaluation labels, optionally restricted to one split/languages."""
    wanted = set(languages) if languages else None
    out: list[EvalExample] = []
    for record in _iter_jsonl(CorpusPaths(name).queries):
        record_split = record.pop("eval_split", None)
        if eval_split and record_split != eval_split:
            continue
        example = EvalExample.from_json(record)
        if wanted and example.language not in wanted:
            continue
        if with_labels_only and not example.has_label:
            continue
        out.append(example)
    return out


def load_stats(name: str) -> dict[str, Any]:
    path = CorpusPaths(name).stats
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def verify_no_leakage(name: str) -> dict[str, Any]:
    """Assert the indexed corpus was built from passages only.

    The failure mode being guarded against is *constructing* indexed documents
    out of query/answer text (e.g. indexing ``query + answer + passage``), which
    would leak the answer into the retrieval corpus.

    One subtlety that a naive exact-match check gets wrong. MS MARCO answers are
    frequently written by copying the selected passage **verbatim**, so
    ``Answer == passage text`` occurs naturally. When the matching chunk is one of
    that query's own candidate passages, the text is in the index because we
    indexed the *passage* - which is correct and unavoidable. Removing it would
    mean deleting a legitimate gold passage from the corpus and quietly making the
    query unanswerable.

    So answer matches are classified:

    ``extractive_overlap``
        the matching chunk is a candidate passage of that same query -> benign,
        a property of the dataset.
    ``answer_text_in_index``
        the answer text appears somewhere it is *not* a passage of that query ->
        genuine leakage.
    """
    chunks = load_chunks(name)
    by_text: dict[str, list[Any]] = {}
    for chunk in chunks:
        by_text.setdefault(chunk.text, []).append(chunk)

    examples = load_eval_examples(name, with_labels_only=False)

    queries = {e.query for e in examples if e.query}
    query_hits = sorted(set(by_text) & queries)

    real_answer_leaks: list[str] = []
    extractive_overlap = 0
    for example in examples:
        if not example.answer:
            continue
        matches = by_text.get(example.answer)
        if not matches:
            continue
        candidates = set(example.candidate_hashes)
        if any(chunk.content_hash in candidates for chunk in matches):
            extractive_overlap += 1
        else:
            real_answer_leaks.append(example.answer[:80])

    # A chunk must never contain a query AND its answer concatenated.
    concatenation_hits: list[str] = []
    for example in examples:
        if not (example.query and example.answer):
            continue
        for text in by_text:
            if example.query in text and example.answer in text and text != example.answer:
                concatenation_hits.append(f"query_id={example.query_id}")
                break

    forbidden_fields = {"is_selected", "query", "answer", "Answer", "query_type"}
    sample = next(_iter_jsonl(CorpusPaths(name).chunks), {})
    field_hits = sorted(forbidden_fields & set(sample))

    return {
        "chunks": len(by_text),
        "queries_checked": len(queries),
        "answers_checked": sum(1 for e in examples if e.answer),
        "query_text_in_index": query_hits[:5],
        "answer_text_in_index": real_answer_leaks[:5],
        "extractive_overlap_benign": extractive_overlap,
        "query_answer_concatenation": concatenation_hits[:5],
        "forbidden_payload_fields": field_hits,
        "clean": (
            not query_hits
            and not real_answer_leaks
            and not concatenation_hits
            and not field_hits
        ),
    }
