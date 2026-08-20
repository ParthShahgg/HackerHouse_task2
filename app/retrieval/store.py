"""Qdrant collection management and upload.

Why Qdrant
----------
It stores dense and learned-sparse vectors in the *same point* under named
vectors, and implements Reciprocal Rank Fusion server-side. That means hybrid
retrieval is one round trip against one consistent snapshot, instead of two
stores that can drift apart. Chroma has no first-class sparse-vector support, so
BGE-M3's lexical branch would have to be bolted on separately - which is why it
is not used here.

Collection layout
-----------------
One collection, two named vectors per point::

    dense  : <dim from model config>, cosine
    sparse : BGE-M3 learned lexical weights (token_id -> weight)

Cosine, not dot: dense vectors are already L2-normalised so the two rank
identically, but declaring cosine makes Qdrant normalise defensively and keeps
the metric correct if an un-normalised vector is ever inserted.

The sparse vector carries **no IDF modifier**. BGE-M3's weights are already
learned term importances; applying Qdrant's IDF on top would re-weight by corpus
frequency a second time and distort the lexical branch. IDF is the right choice
for raw BM25-style counts, which these are not.

Payload indexes are created for ``language``, ``strategy`` and ``parent_id``.
Without them, language filtering degrades into a full scan and the latency
budget is gone.
"""

from __future__ import annotations

import random
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from app.config import get_settings
from app.observability.tracing import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

# Substrings identifying faults that are worth retrying. "Server disconnected
# without sending a response" is the important one: it is what httpx raises when
# the Qdrant container goes away or drops a pooled connection mid-request, and it
# killed a full evaluation run at query 40/92. A dropped connection is not a
# deterministic failure, so retrying is correct - but the pooled socket is dead,
# so the client must also be rebuilt (see `_RECONNECT_MARKERS`).
_RETRYABLE_MARKERS = (
    "server disconnected",
    "remoteprotocolerror",
    "connection reset",
    "connection aborted",
    "connection refused",
    "cannot connect",
    "timed out",
    "timeout",
    "read error",
    "write error",
    "temporarily unavailable",
    "502",
    "503",
    "504",
    "no route to host",
    "broken pipe",
    "incomplete",
)

# Faults where the existing connection pool is unusable and must be discarded.
_RECONNECT_MARKERS = (
    "server disconnected",
    "remoteprotocolerror",
    "connection reset",
    "connection aborted",
    "connection refused",
    "cannot connect",
    "broken pipe",
    "read error",
    "write error",
)


def _classify(exc: BaseException) -> tuple[bool, bool]:
    """Return ``(retryable, needs_reconnect)`` for an exception."""
    text = f"{type(exc).__name__} {exc}".lower()
    retryable = any(marker in text for marker in _RETRYABLE_MARKERS)
    reconnect = any(marker in text for marker in _RECONNECT_MARKERS)
    return retryable, reconnect

__all__ = [
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "QdrantStore",
    "chunk_point_id",
    "get_store",
    "reset_store",
]

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

# Stable namespace so a rebuild regenerates identical point IDs and therefore
# upserts (overwrites) instead of duplicating.
_POINT_NAMESPACE = uuid.UUID("6f9b1e2c-8a3d-4c5e-9f10-2b7d4a6c8e13")


def chunk_point_id(chunk_id: str) -> str:
    """Deterministic UUIDv5 point ID for a chunk.

    Qdrant point IDs must be uint64 or UUID, but our identifiers are readable
    strings like ``hi:0f3a...#sw0_1``. Deriving a UUID5 keeps IDs stable and
    idempotent while the human-readable ``chunk_id`` lives in the payload.
    """
    return str(uuid.uuid5(_POINT_NAMESPACE, chunk_id))


class QdrantStore:
    """Thin wrapper over ``QdrantClient`` with collection lifecycle helpers."""

    def __init__(
        self,
        *,
        url: str | None = None,
        api_key: str | None = None,
        collection: str | None = None,
        timeout: float | None = None,
        prefer_grpc: bool | None = None,
        local_path: str | None = None,
        allow_local_fallback: bool = True,
        max_retries: int | None = None,
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.url = url or settings.qdrant_url
        self.api_key = api_key if api_key is not None else (settings.qdrant_api_key or None)
        self.collection = collection or settings.qdrant_collection
        self.timeout = timeout or settings.qdrant_timeout_s
        self.prefer_grpc = settings.qdrant_prefer_grpc if prefer_grpc is None else prefer_grpc
        self.local_path = local_path or settings.qdrant_local_path
        self.allow_local_fallback = allow_local_fallback
        self.max_retries = (
            settings.qdrant_max_retries if max_retries is None else max_retries
        )
        self._client: Any = None
        self.using_local = False

    # ------------------------------------------------------------------ client
    @property
    def client(self):
        if self._client is None:
            self._client = self._connect()
        return self._client

    def _connect(self):
        from qdrant_client import QdrantClient

        try:
            client = QdrantClient(
                url=self.url,
                api_key=self.api_key,
                timeout=int(self.timeout),
                prefer_grpc=self.prefer_grpc,
            )
            client.get_collections()  # force a real handshake
            logger.info("connected to Qdrant at %s", self.url)
            return client
        except Exception as exc:  # noqa: BLE001
            if not self.allow_local_fallback:
                raise
            # Embedded on-disk mode keeps the demo runnable without Docker. It is
            # a development convenience, never the deployment target - it has no
            # HNSW quantisation, no concurrency and no persistence guarantees.
            path = self.settings.resolve_path(self.local_path)
            logger.warning(
                "Qdrant server unreachable at %s (%s); falling back to embedded "
                "client at %s. NOT for production.",
                self.url, exc, path,
            )
            path.mkdir(parents=True, exist_ok=True)
            self.using_local = True
            return QdrantClient(path=str(path))

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001
                pass
            self._client = None

    # ------------------------------------------------------------------- retry
    def _with_retry(
        self,
        operation: str,
        func: Callable[[], T],
        *,
        attempts: int | None = None,
    ) -> T:
        """Run a Qdrant call with bounded retries and exponential backoff.

        Exists because a single dropped connection should not destroy a
        multi-hour evaluation run. Only *transient* faults are retried;
        deterministic errors (bad collection name, dimension mismatch, malformed
        filter) raise immediately rather than being retried three times and
        delaying the real error by several seconds.

        On a connection-level fault the client is rebuilt, because the pooled
        socket is already dead and retrying on it would fail identically.
        """
        total = attempts if attempts is not None else self.max_retries
        last: BaseException | None = None

        for attempt in range(1, total + 1):
            try:
                return func()
            except Exception as exc:  # noqa: BLE001
                retryable, needs_reconnect = _classify(exc)
                last = exc
                if not retryable or attempt >= total:
                    if not retryable:
                        logger.debug(
                            "qdrant %s failed non-transiently: %s: %s",
                            operation, type(exc).__name__, exc,
                        )
                    break
                if needs_reconnect:
                    logger.warning(
                        "qdrant %s: connection-level fault (%s); rebuilding client",
                        operation, type(exc).__name__,
                    )
                    self.close()
                backoff = min(0.5 * (2 ** (attempt - 1)), 8.0) + random.uniform(0, 0.3)
                logger.warning(
                    "qdrant %s failed (attempt %d/%d): %s: %s - retrying in %.2fs",
                    operation, attempt, total, type(exc).__name__, exc, backoff,
                )
                time.sleep(backoff)

        assert last is not None
        raise last

    # ------------------------------------------------------- guarded operations
    def query_points(self, **kwargs: Any):
        """``client.query_points`` with retry/backoff and reconnect.

        All search paths (dense, sparse, server-side RRF fusion) go through here
        so none of them can bypass the resilience policy.
        """
        return self._with_retry(
            f"query_points({kwargs.get('collection_name')})",
            lambda: self.client.query_points(**kwargs),
        )

    def retrieve_points(self, **kwargs: Any):
        return self._with_retry(
            f"retrieve({kwargs.get('collection_name')})",
            lambda: self.client.retrieve(**kwargs),
        )

    # -------------------------------------------------------------- collection
    def exists(self, collection: str | None = None) -> bool:
        name = collection or self.collection
        try:
            return self.client.collection_exists(name)
        except Exception:  # noqa: BLE001
            return False

    def create_collection(
        self,
        dim: int,
        *,
        collection: str | None = None,
        recreate: bool = False,
        on_disk: bool = False,
    ) -> None:
        from qdrant_client import models

        name = collection or self.collection
        if self.exists(name):
            if not recreate:
                logger.info("collection %s already exists", name)
                return
            logger.warning("deleting existing collection %s (--rebuild)", name)
            self.client.delete_collection(name)

        self.client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                    on_disk=on_disk,
                )
            },
            sparse_vectors_config={
                # No `modifier=IDF`: BGE-M3 weights are already learned
                # importances (see module docstring).
                SPARSE_VECTOR: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=on_disk)
                )
            },
        )
        logger.info("created collection %s (dim=%d)", name, dim)
        self._create_payload_indexes(name)

    def _create_payload_indexes(self, name: str) -> None:
        from qdrant_client import models

        for field in ("language", "strategy", "parent_id", "source_split"):
            try:
                self.client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001
                # Embedded mode does not implement payload indexes; filtering
                # still works there, just without the index.
                logger.debug("payload index %s skipped: %s", field, exc)

    def count(self, collection: str | None = None, *, exact: bool = True) -> int:
        name = collection or self.collection
        try:
            return int(
                self._with_retry(
                    f"count({name})", lambda: self.client.count(name, exact=exact)
                ).count
            )
        except Exception:  # noqa: BLE001
            return 0

    def info(self, collection: str | None = None) -> dict[str, Any]:
        name = collection or self.collection
        try:
            raw = self.client.get_collection(name)
            return {
                "name": name,
                "points": getattr(raw, "points_count", None),
                "vectors": getattr(raw, "vectors_count", None),
                "indexed_vectors": getattr(raw, "indexed_vectors_count", None),
                "status": str(getattr(raw, "status", "")),
            }
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "error": str(exc)}

    # ------------------------------------------------------------------ upload
    def upsert_chunks(
        self,
        chunks: Sequence[Any],
        dense_vectors: Any,
        sparse_vectors: Sequence[dict[int, float]],
        *,
        collection: str | None = None,
        batch_size: int = 128,
        wait: bool = True,
    ) -> int:
        """Upload chunks with both representations. Returns points written."""
        from qdrant_client import models

        name = collection or self.collection
        if len(chunks) != len(sparse_vectors) or len(chunks) != dense_vectors.shape[0]:
            raise ValueError(
                f"length mismatch: chunks={len(chunks)} dense={dense_vectors.shape[0]} "
                f"sparse={len(sparse_vectors)}"
            )

        written = 0
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = []
            for offset, chunk in enumerate(batch):
                i = start + offset
                sparse = sparse_vectors[i]
                points.append(
                    models.PointStruct(
                        id=chunk_point_id(chunk.chunk_id),
                        vector={
                            DENSE_VECTOR: dense_vectors[i].tolist(),
                            SPARSE_VECTOR: models.SparseVector(
                                indices=list(sparse.keys()),
                                values=[float(v) for v in sparse.values()],
                            ),
                        },
                        payload=chunk.to_payload(),
                    )
                )
            self._with_retry(
                f"upsert({name}, {len(points)} pts)",
                lambda: self.client.upsert(
                    collection_name=name, points=points, wait=wait
                ),
            )
            written += len(points)
        return written

    def delete_collection(self, collection: str | None = None) -> None:
        name = collection or self.collection
        if self.exists(name):
            self.client.delete_collection(name)
            logger.info("deleted collection %s", name)

    def fetch_by_chunk_ids(
        self, chunk_ids: Iterable[str], *, collection: str | None = None
    ) -> dict[str, dict[str, Any]]:
        """Retrieve payloads for specific chunk IDs (used by parent expansion)."""
        ids = [chunk_point_id(c) for c in chunk_ids]
        if not ids:
            return {}
        records = self.retrieve_points(
            collection_name=collection or self.collection,
            ids=ids,
            with_payload=True,
            with_vectors=False,
        )
        return {r.payload["chunk_id"]: r.payload for r in records if r.payload}

    def health(self) -> tuple[bool, str]:
        try:
            self._with_retry("get_collections", lambda: self.client.get_collections())
            mode = "embedded" if self.using_local else "server"
            return True, f"{mode} ok"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)


_STORE: QdrantStore | None = None


def get_store(**kwargs: Any) -> QdrantStore:
    global _STORE
    if _STORE is None:
        _STORE = QdrantStore(**kwargs)
    return _STORE


def reset_store() -> None:
    global _STORE
    if _STORE is not None:
        _STORE.close()
    _STORE = None
