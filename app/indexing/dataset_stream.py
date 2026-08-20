"""Streaming reader for ``ai4bharat/MSMARCO-XI``.

Why this is hand-rolled on pyarrow instead of using ``datasets``
----------------------------------------------------------------
1. MSMARCO-XI ships a legacy loading script (``ms_marco_translations.py``).
   ``datasets>=4`` refuses to execute repo scripts, so the documented
   ``load_dataset("ai4bharat/MSMARCO-XI", "hi")`` config API does not work.
2. The corpus is ~11.45M rows / 55.6GB. It must never be materialised.

Measured layout of the real repo (verified 2026-08)
---------------------------------------------------
Every shard is a **single Parquet row group**:

===========================  =========  ==========  ==================
shard                        rows       compressed  uncompressed rg
===========================  =========  ==========  ==================
``train/hintrain.parquet``     778,638      3.72 GB            9.73 GB
``validation/hinval.parquet``   97,941      0.46 GB            1.20 GB
===========================  =========  ==========  ==================

A single row group defeats the usual "iterate row groups" trick. Two things
make bounded-memory streaming work anyway:

* **Column projection**, including nested struct fields. Reading
  ``passages.Translated_passages`` + ``passages.is_selected`` and skipping
  ``passages.English_passages`` cuts bytes transferred by ~37%.
* **``pre_buffer=False``** with a modest ``buffer_size``, which makes pyarrow
  pull *data pages* on demand rather than materialising the whole column chunk.
  Measured on ``validation/tamval.parquet``: first 64-row batch in ~18.5s
  (connection + first pages), the following 576 rows in ~2.0s. Peak RSS stays
  flat, and ``--max-rows-per-language`` genuinely stops early instead of
  pretending to.

Consequence: bounding rows bounds work. That is what allows a dev/demo corpus
without a 55.6GB download.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from app.config import DATA_DIR, get_settings
from app.indexing.normalize import normalize_text
from app.indexing.records import DatasetRow
from app.languages import dataset_filename, has_split
from app.observability.tracing import get_logger

logger = get_logger(__name__)

__all__ = [
    "MSMarcoXIStreamer",
    "ShardInfo",
    "stream_language",
    "inspect_shard",
    "local_cache_path",
]

HF_DATASETS_PREFIX = "datasets"

# Minimum set of columns. `Answer` and `query` are read because they are needed
# for the *evaluation label store*; they are never written to the live index.
BASE_COLUMNS: tuple[str, ...] = (
    "query_id",
    "query",
    "query_type",
    "Answer",
    "passages.Translated_passages",
    "passages.is_selected",
)
ENGLISH_COLUMNS: tuple[str, ...] = ("passages.English_passages",)

_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection", "reset", "temporarily",
    "503", "502", "504", "429", "incomplete", "ssl", "eof",
)


class ShardInfo:
    """Footer-only metadata for a shard (a couple of HTTP range reads)."""

    def __init__(self, path: str, num_rows: int, num_row_groups: int, byte_size: int,
                 column_bytes: dict[str, int]) -> None:
        self.path = path
        self.num_rows = num_rows
        self.num_row_groups = num_row_groups
        self.byte_size = byte_size
        self.column_bytes = column_bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "num_rows": self.num_rows,
            "num_row_groups": self.num_row_groups,
            "uncompressed_bytes": self.byte_size,
            "compressed_column_bytes": self.column_bytes,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ShardInfo({self.path}, rows={self.num_rows:,}, "
            f"row_groups={self.num_row_groups}, size={self.byte_size / 1e9:.2f}GB)"
        )


def _is_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def local_cache_path(language: str, split: str, dataset_id: str | None = None) -> Path:
    """Where a projected local shard subset is cached.

    Caching the *projected* subset makes repeated evaluation and benchmark runs
    instant instead of re-fetching over the network.
    """
    settings = get_settings()
    slug = (dataset_id or settings.dataset_id).replace("/", "__")
    return DATA_DIR / "raw_cache" / slug / f"{language}_{split}.parquet"


class MSMarcoXIStreamer:
    """Bounded-memory reader over one MSMARCO-XI language shard."""

    def __init__(
        self,
        dataset_id: str | None = None,
        *,
        include_english: bool = False,
        batch_size: int = 64,
        buffer_size: int = 1024 * 1024,
        block_size: int = 4 * 1024 * 1024,
        max_retries: int = 3,
        hf_token: str | None = None,
        use_local_cache: bool = True,
    ) -> None:
        settings = get_settings()
        self.dataset_id = dataset_id or settings.dataset_id
        self.include_english = include_english
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.block_size = block_size
        self.max_retries = max_retries
        self.hf_token = hf_token or settings.hf_token or None
        self.use_local_cache = use_local_cache

    # ------------------------------------------------------------------ paths
    def columns(self) -> list[str]:
        cols = list(BASE_COLUMNS)
        if self.include_english:
            cols.extend(ENGLISH_COLUMNS)
        return cols

    def repo_path(self, language: str, split: str) -> str:
        return f"{HF_DATASETS_PREFIX}/{self.dataset_id}/{dataset_filename(language, split)}"

    def _open_remote(self, language: str, split: str):
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem(token=self.hf_token)
        return fs.open(self.repo_path(language, split), "rb", block_size=self.block_size)

    def _open_parquet(self, language: str, split: str) -> tuple[pq.ParquetFile, str, bool]:
        """Return ``(ParquetFile, source_description, is_local)``."""
        cache = local_cache_path(language, split, self.dataset_id)
        if self.use_local_cache and cache.exists():
            logger.info("using local shard cache %s", cache)
            return pq.ParquetFile(cache, pre_buffer=False, buffer_size=self.buffer_size), str(cache), True

        handle = self._open_remote(language, split)
        # pre_buffer=False is load-bearing: with pre_buffer=True pyarrow fetches
        # the entire column chunk up front, which for these single-row-group
        # shards means downloading the whole (projected) file before the first
        # row appears.
        return (
            pq.ParquetFile(handle, pre_buffer=False, buffer_size=self.buffer_size),
            self.repo_path(language, split),
            False,
        )

    # --------------------------------------------------------------- metadata
    def inspect(self, language: str, split: str) -> ShardInfo:
        pf, path, _ = self._open_parquet(language, split)
        meta = pf.metadata
        column_bytes: dict[str, int] = {}
        if meta.num_row_groups:
            rg = meta.row_group(0)
            for i in range(rg.num_columns):
                col = rg.column(i)
                column_bytes[col.path_in_schema] = col.total_compressed_size
        total_uncompressed = sum(
            meta.row_group(i).total_byte_size for i in range(meta.num_row_groups)
        )
        return ShardInfo(path, meta.num_rows, meta.num_row_groups, total_uncompressed, column_bytes)

    # ---------------------------------------------------------------- reading
    def _iter_batches(self, pf: pq.ParquetFile) -> Iterator[pa.RecordBatch]:
        return pf.iter_batches(batch_size=self.batch_size, columns=self.columns())

    def stream(
        self,
        language: str,
        split: str,
        *,
        max_rows: int | None = None,
        skip_rows: int = 0,
        sample_stride: int = 1,
    ) -> Iterator[DatasetRow]:
        """Yield :class:`DatasetRow` objects with bounded memory.

        Parameters
        ----------
        max_rows:
            Hard cap. ``None`` streams the whole shard.
        skip_rows:
            Rows to discard first - used to carve a disjoint evaluation slice.
        sample_stride:
            Take every Nth row. ``sample_stride > 1`` spreads a small sample
            across the shard instead of taking a contiguous head, which matters
            because MSMARCO rows are not randomly ordered by topic.
        """
        if not has_split(language, split):
            raise FileNotFoundError(
                f"MSMARCO-XI has no {split!r} shard for language {language!r}. "
                "See app/languages.py - Telugu exists only in validation."
            )

        attempt = 0
        emitted = 0
        seen = 0
        while True:
            try:
                pf, source, is_local = self._open_parquet(language, split)
                logger.info(
                    "streaming %s (%s) rows=%s cap=%s",
                    source, "local" if is_local else "remote", f"{pf.metadata.num_rows:,}", max_rows,
                )
                for batch in self._iter_batches(pf):
                    for record in batch.to_pylist():
                        seen += 1
                        if seen <= skip_rows:
                            continue
                        if sample_stride > 1 and (seen - skip_rows - 1) % sample_stride:
                            continue
                        row = self._to_row(record, language, split)
                        if row is None:
                            continue
                        yield row
                        emitted += 1
                        if max_rows is not None and emitted >= max_rows:
                            return
                return
            except (OSError, pa.ArrowException, RuntimeError) as exc:
                # Restarting mid-stream would duplicate rows, so only retry when
                # nothing has been emitted yet. Otherwise surface the failure and
                # let the caller keep the partial corpus.
                attempt += 1
                if emitted:
                    logger.error(
                        "stream failed after %d rows (%s); keeping partial output", emitted, exc
                    )
                    return
                if attempt > self.max_retries or not _is_transient(exc):
                    raise
                backoff = min(2 ** attempt, 20) + random.uniform(0, 0.5)
                logger.warning(
                    "transient stream error (attempt %d/%d): %s - retrying in %.1fs",
                    attempt, self.max_retries, exc, backoff,
                )
                time.sleep(backoff)
                seen = 0

    @staticmethod
    def _to_row(record: dict[str, Any], language: str, split: str) -> DatasetRow | None:
        passages_field = record.get("passages") or {}
        translated: Sequence[str] = passages_field.get("Translated_passages") or []
        selected: Sequence[int] = passages_field.get("is_selected") or []
        english: Sequence[str] = passages_field.get("English_passages") or []

        if not translated:
            return None

        # `is_selected` is occasionally shorter than the passage list upstream.
        # Pad rather than drop the row: the passages are still valid corpus
        # content, they simply have no positive judgement.
        labels = list(selected) + [0] * max(0, len(translated) - len(selected))

        query = normalize_text(record.get("query"))
        if not query:
            return None

        return DatasetRow(
            query_id=int(record.get("query_id") or 0),
            language=language,
            split=split,
            query=query,
            answer=normalize_text(record.get("Answer")),
            query_type=(record.get("query_type") or "").strip(),
            passages=[normalize_text(p) for p in translated],
            is_selected=[int(x) for x in labels[: len(translated)]],
            english_passages=[normalize_text(p) for p in english],
        )

    # ------------------------------------------------------------ local cache
    def write_local_cache(
        self, language: str, split: str, *, max_rows: int | None = None
    ) -> Path:
        """Persist a projected subset locally for fast repeat runs."""
        target = local_cache_path(language, split, self.dataset_id)
        target.parent.mkdir(parents=True, exist_ok=True)

        pf, _, _ = self._open_parquet(language, split)
        writer: pq.ParquetWriter | None = None
        written = 0
        try:
            for batch in self._iter_batches(pf):
                table = pa.Table.from_batches([batch])
                if max_rows is not None and written + table.num_rows > max_rows:
                    table = table.slice(0, max_rows - written)
                if writer is None:
                    writer = pq.ParquetWriter(target, table.schema, compression="zstd")
                writer.write_table(table)
                written += table.num_rows
                if max_rows is not None and written >= max_rows:
                    break
        finally:
            if writer is not None:
                writer.close()
        logger.info("wrote local cache %s (%d rows)", target, written)
        return target


def stream_language(
    language: str,
    split: str = "validation",
    *,
    max_rows: int | None = None,
    skip_rows: int = 0,
    sample_stride: int = 1,
    include_english: bool = False,
) -> Iterator[DatasetRow]:
    """Convenience wrapper around :class:`MSMarcoXIStreamer`."""
    streamer = MSMarcoXIStreamer(include_english=include_english)
    yield from streamer.stream(
        language, split, max_rows=max_rows, skip_rows=skip_rows, sample_stride=sample_stride
    )


def inspect_shard(language: str, split: str) -> ShardInfo:
    return MSMarcoXIStreamer().inspect(language, split)
