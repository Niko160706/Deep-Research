"""SQLite connection/backup boundary with explicit Python 3.12 transactions."""

from __future__ import annotations

import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from gidr.infrastructure.localfs import (
    LocalLock,
    private_directory,
    regular_fd,
    reject_symlinks,
    sync_directory,
)


class Database:
    def __init__(self, path: Path, *, busy_timeout_ms: int = 5000):
        if not 1 <= busy_timeout_ms <= 60000:
            raise ValueError("INVALID_BUSY_TIMEOUT")
        self.path = private_directory(path.parent) / path.name
        self.busy_timeout_ms = busy_timeout_ms
        self.lifecycle = LocalLock(self.path.with_suffix(self.path.suffix + ".lock"))

    def _connect(self, *, readonly: bool = False) -> sqlite3.Connection:
        reject_symlinks(self.path)
        for suffix in ("-wal", "-shm", "-journal"):
            reject_symlinks(Path(str(self.path) + suffix))
        if readonly:
            conn = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, autocommit=True)
        else:
            fd = regular_fd(self.path, create=True)
            os.close(fd)
            conn = sqlite3.connect(self.path, autocommit=True)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA trusted_schema=OFF")
        conn.execute(f"PRAGMA busy_timeout={self.busy_timeout_ms}")
        if readonly:
            conn.execute("PRAGMA query_only=ON")
        else:
            if conn.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                conn.close()
                raise RuntimeError("WAL_REQUIRED")
            conn.execute("PRAGMA synchronous=FULL")
        return conn

    @contextmanager
    def reader(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect(readonly=True)
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def check(conn: sqlite3.Connection) -> None:
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise RuntimeError("DATABASE_INTEGRITY_FAILURE")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise RuntimeError("DATABASE_FOREIGN_KEY_FAILURE")

    def backup(self, destination: Path) -> None:
        with self.lifecycle.hold(), self.reader() as source:
            self._copy_snapshot(source, destination)

    @staticmethod
    def _copy_snapshot(source: sqlite3.Connection, destination: Path) -> None:
        private_directory(destination.parent)
        reject_symlinks(destination)
        if destination.exists():
            raise FileExistsError("BACKUP_TARGET_EXISTS")
        fd, name = tempfile.mkstemp(prefix=".sqlite-backup-", dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        try:
            target = sqlite3.connect(temporary, autocommit=True)
            try:
                source.backup(target, pages=128)
                Database.check(target)
                target.execute("PRAGMA journal_mode=DELETE")
            finally:
                target.close()
            fd = regular_fd(temporary)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.link(temporary, destination)
            sync_directory(destination.parent)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def restore_new(cls, backup: Path, destination: Path) -> Database:
        reject_symlinks(backup)
        target = cls(destination)
        with target.lifecycle.hold():
            if target.path.exists():
                raise FileExistsError("RESTORE_REQUIRES_NEW_DESTINATION")
            source = sqlite3.connect(
                backup.absolute().as_uri() + "?mode=ro", uri=True, autocommit=True
            )
            try:
                source.enable_load_extension(False)
                source.execute("PRAGMA trusted_schema=OFF")
                cls.check(source)
                cls._copy_snapshot(source, target.path)
            finally:
                source.close()
        return target
