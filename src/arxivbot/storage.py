"""Where the pipeline keeps what it has already paid for.

Three things are cached, and until now all three were files on disk: the
e-print archive arXiv served, the responses the model gave, and the specs that
came out. That is right for a laptop and impossible on a serverless host,
where the filesystem is empty again on the next request and nothing survives.

So the pipeline writes through this interface instead. ``DiskStore`` is the
default and behaves exactly as before. ``SqlStore`` keeps the same bytes in a
database, which is what makes the thing deployable somewhere with no disk.

Keys are opaque strings the caller owns. Nothing here knows what an arXiv id
is, which is why the same store serves archives, model responses and specs.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from arxivbot.ingest.fetch import cache_dir

# Buckets the pipeline uses.
EPRINT = "eprint"
PDF = "pdf"
LLM = "llm"
SPEC = "spec"


class Store(ABC):
    """A namespaced blob store. Implementations must be safe to share."""

    @abstractmethod
    def get(self, bucket: str, key: str) -> bytes | None:
        """The stored bytes, or None when nothing is stored under that key."""

    @abstractmethod
    def put(self, bucket: str, key: str, value: bytes) -> None:
        """Store bytes, replacing anything already under that key."""

    @abstractmethod
    def keys(self, bucket: str) -> list[str]:
        """Every key in a bucket, for listing and for clearing."""

    @abstractmethod
    def delete(self, bucket: str, key: str) -> None: ...

    # ---- text convenience, since most buckets hold JSON ----

    def get_text(self, bucket: str, key: str) -> str | None:
        raw = self.get(bucket, key)
        return raw.decode("utf-8") if raw is not None else None

    def put_text(self, bucket: str, key: str, value: str) -> None:
        self.put(bucket, key, value.encode("utf-8"))

    def size(self, bucket: str) -> int:
        return sum(len(self.get(bucket, k) or b"") for k in self.keys(bucket))


def _safe(key: str) -> str:
    """A key as a filename. Separators are the only real hazard."""
    return key.replace("/", "_").replace("\\", "_")


def _migrate_legacy(root: Path) -> None:
    """Move caches written before buckets existed into their bucket.

    Earlier versions put archives at ``<cache>/1706.03762v7.eprint`` and specs
    under ``<cache>/specs/``. Leaving them there would silently orphan work
    that cost real time - a spec is twenty-odd model calls and an archive is
    megabytes - so they are moved once, on first use, and the marker stops it
    happening again.
    """
    marker = root / ".migrated-to-buckets"
    if marker.exists():
        return

    moves: list[tuple[Path, Path]] = []
    for suffix, bucket in ((".eprint", EPRINT), (".pdf", PDF)):
        for path in root.glob(f"*{suffix}"):
            moves.append((path, root / bucket / path.name[: -len(suffix)]))
    legacy_specs = root / "specs"
    if legacy_specs.is_dir():
        for path in legacy_specs.glob("*.json"):
            moves.append((path, root / SPEC / path.name))

    for source, target in moves:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                source.replace(target)
        except OSError:
            continue

    try:
        root.mkdir(parents=True, exist_ok=True)
        marker.write_text("", encoding="utf-8")
    except OSError:
        pass


class DiskStore(Store):
    """Files under the platform's cache directory. The default."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or cache_dir()
        _migrate_legacy(self.root)

    def _path(self, bucket: str, key: str) -> Path:
        folder = self.root / bucket
        folder.mkdir(parents=True, exist_ok=True)
        return folder / _safe(key)

    def get(self, bucket: str, key: str) -> bytes | None:
        path = self._path(bucket, key)
        try:
            return path.read_bytes()
        except (FileNotFoundError, IsADirectoryError, PermissionError):
            return None

    def put(self, bucket: str, key: str, value: bytes) -> None:
        try:
            self._path(bucket, key).write_bytes(value)
        except OSError:
            # A cache that cannot be written is slow, not broken.
            pass

    def keys(self, bucket: str) -> list[str]:
        folder = self.root / bucket
        if not folder.is_dir():
            return []
        return sorted(p.name for p in folder.iterdir() if p.is_file())

    def delete(self, bucket: str, key: str) -> None:
        self._path(bucket, key).unlink(missing_ok=True)


class SqlStore(Store):
    """The same bytes in SQLite, for hosts with no durable filesystem.

    One table, keyed by bucket and key. SQLite because it needs no server and
    the file can sit on a mounted volume; a hosted Postgres speaks the same
    SQL and can be swapped in by changing the connection.
    """

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._local = threading.local()
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    bucket     TEXT NOT NULL,
                    key        TEXT NOT NULL,
                    value      BLOB NOT NULL,
                    written_at REAL NOT NULL,
                    PRIMARY KEY (bucket, key)
                )
                """
            )

    def _connect(self) -> sqlite3.Connection:
        # A connection per thread: the server answers requests concurrently
        # and sqlite3 objects are not shareable across threads.
        existing = getattr(self._local, "connection", None)
        if existing is None:
            existing = sqlite3.connect(self.path, timeout=30)
            existing.execute("PRAGMA journal_mode = WAL")
            self._local.connection = existing
        return existing

    def get(self, bucket: str, key: str) -> bytes | None:
        row = (
            self._connect()
            .execute(
                "SELECT value FROM blobs WHERE bucket = ? AND key = ?",
                (bucket, _safe(key)),
            )
            .fetchone()
        )
        return bytes(row[0]) if row else None

    def put(self, bucket: str, key: str, value: bytes) -> None:
        import time

        connection = self._connect()
        with connection:
            connection.execute(
                "INSERT INTO blobs (bucket, key, value, written_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(bucket, key) DO UPDATE SET "
                "value = excluded.value, written_at = excluded.written_at",
                (bucket, _safe(key), value, time.time()),
            )

    def keys(self, bucket: str) -> list[str]:
        rows = (
            self._connect()
            .execute("SELECT key FROM blobs WHERE bucket = ? ORDER BY key", (bucket,))
            .fetchall()
        )
        return [row[0] for row in rows]

    def delete(self, bucket: str, key: str) -> None:
        connection = self._connect()
        with connection:
            connection.execute(
                "DELETE FROM blobs WHERE bucket = ? AND key = ?", (bucket, _safe(key))
            )


_store: Store | None = None


def store() -> Store:
    """The store this process uses.

    ``ARXIVBOT_STORE=sqlite`` with ``ARXIVBOT_STORE_PATH`` selects the
    database backend, which is what a serverless deployment sets. Anything
    else keeps files on disk.
    """
    global _store
    if _store is None:
        if os.environ.get("ARXIVBOT_STORE", "").lower() in ("sqlite", "sql", "db"):
            _store = SqlStore(os.environ.get("ARXIVBOT_STORE_PATH", ":memory:"))
        else:
            _store = DiskStore()
    return _store


def use(replacement: Store | None) -> None:
    """Point the pipeline at a different store. Passing None restores the default."""
    global _store
    _store = replacement
