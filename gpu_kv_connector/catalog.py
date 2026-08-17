from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from gpu_kv_connector.hashing import OBJECT_ID_BYTES


class SQLiteObjectCatalog:
    """Small shared catalog for scheduler-visible object completeness.

    Physical SSD locations deliberately do not appear here. Each GPU worker
    resolves those through its GPU-resident LSM index. A logical object becomes
    visible only after every tensor-parallel rank has persisted all planes.
    """

    _MAX_SQL_KEYS = 800

    def __init__(self, path: str, *, reset: bool = False) -> None:
        self.path = str(Path(path).expanduser().resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path, timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                object_id BLOB PRIMARY KEY,
                ready_mask INTEGER NOT NULL,
                updated_ns INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        if reset:
            with self._lock, self._connection:
                self._connection.execute("DELETE FROM objects")

    @staticmethod
    def _validate_ids(object_ids: Iterable[bytes]) -> list[bytes]:
        result = list(object_ids)
        if any(len(object_id) != OBJECT_ID_BYTES for object_id in result):
            raise ValueError(f"all object IDs must contain {OBJECT_ID_BYTES} bytes")
        return result

    def ready_set(self, object_ids: Iterable[bytes], full_mask: int) -> set[bytes]:
        ids = self._validate_ids(object_ids)
        if full_mask <= 0:
            raise ValueError("full_mask must contain at least one worker bit")
        ready: set[bytes] = set()
        with self._lock:
            for start in range(0, len(ids), self._MAX_SQL_KEYS):
                chunk = ids[start : start + self._MAX_SQL_KEYS]
                if not chunk:
                    continue
                placeholders = ",".join("?" for _ in chunk)
                rows = self._connection.execute(
                    f"SELECT object_id, ready_mask FROM objects "
                    f"WHERE object_id IN ({placeholders})",
                    chunk,
                )
                ready.update(
                    object_id
                    for object_id, ready_mask in rows
                    if ready_mask & full_mask == full_mask
                )
        return ready

    def longest_ready_prefix(self, object_ids: Sequence[bytes], full_mask: int) -> int:
        ready = self.ready_set(object_ids, full_mask)
        for index, object_id in enumerate(object_ids):
            if object_id not in ready:
                return index
        return len(object_ids)

    def mark_rank_ready(self, object_ids: Iterable[bytes], rank: int) -> None:
        ids = self._validate_ids(object_ids)
        if not 0 <= rank < 63:
            raise ValueError("catalog supports tensor-parallel ranks 0..62")
        if not ids:
            return
        bit = 1 << rank
        now = time.time_ns()
        rows = [(object_id, bit, now) for object_id in ids]
        with self._lock, self._connection:
            self._connection.executemany(
                """
                INSERT INTO objects(object_id, ready_mask, updated_ns)
                VALUES (?, ?, ?)
                ON CONFLICT(object_id) DO UPDATE SET
                    ready_mask = objects.ready_mask | excluded.ready_mask,
                    updated_ns = excluded.updated_ns
                """,
                rows,
            )

    def clear_rank(self, object_ids: Iterable[bytes], rank: int) -> None:
        ids = self._validate_ids(object_ids)
        if not 0 <= rank < 63:
            raise ValueError("catalog supports tensor-parallel ranks 0..62")
        if not ids:
            return
        keep_mask = ~(1 << rank)
        now = time.time_ns()
        with self._lock, self._connection:
            for start in range(0, len(ids), self._MAX_SQL_KEYS):
                chunk = ids[start : start + self._MAX_SQL_KEYS]
                placeholders = ",".join("?" for _ in chunk)
                self._connection.execute(
                    f"UPDATE objects SET ready_mask = ready_mask & ?, "
                    f"updated_ns = ? WHERE object_id IN ({placeholders})",
                    [keep_mask, now, *chunk],
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
