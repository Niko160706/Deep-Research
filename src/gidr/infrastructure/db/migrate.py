"""Ordered checksummed forward migrations, backed up before an upgrade."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from importlib.resources import files

from gidr.infrastructure.db.database import Database


def statements(sql: str) -> Iterator[str]:
    pending = ""
    for line in sql.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            yield pending
            pending = ""
    if pending.strip():
        raise ValueError("INCOMPLETE_MIGRATION_SQL")


def bundled_migrations() -> tuple[tuple[int, str, str], ...]:
    root = files("gidr.infrastructure.db").joinpath("migrations")
    result = []
    for item in sorted(root.iterdir(), key=lambda path: path.name):
        if item.name.endswith(".sql"):
            result.append((int(item.name.split("_", 1)[0]), item.name, item.read_text()))
    return tuple(result)


def migrate(
    database: Database,
    *,
    migrations: tuple[tuple[int, str, str], ...] | None = None,
    target_version: int | None = None,
    fault: Callable[[str], None] = lambda _: None,
) -> int:
    selected = bundled_migrations() if migrations is None else migrations
    if not selected or [item[0] for item in selected] != list(range(1, len(selected) + 1)):
        raise ValueError("MIGRATION_SEQUENCE_INVALID")
    target = selected[-1][0] if target_version is None else target_version
    if target < 1 or target > selected[-1][0]:
        raise ValueError("MIGRATION_TARGET_UNSUPPORTED")
    with database.lifecycle.hold():
        conn = database._connect()
        try:
            exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='schema_migrations'"
            ).fetchone()
            rows = (
                conn.execute(
                    "SELECT version,name,checksum FROM schema_migrations ORDER BY version"
                ).fetchall()
                if exists
                else []
            )
            if [row["version"] for row in rows] != list(range(1, len(rows) + 1)):
                raise RuntimeError("MIGRATION_HISTORY_GAP")
            if len(rows) > target:
                raise RuntimeError("NEWER_DATABASE_SCHEMA_OR_DOWNGRADE")
            for row in rows:
                version, name, sql = selected[row["version"] - 1]
                if (
                    row["name"] != name
                    or row["checksum"] != hashlib.sha256(sql.encode()).hexdigest()
                ):
                    raise RuntimeError("MIGRATION_CHECKSUM_MISMATCH")
            if rows and len(rows) < target:
                stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")
                database.backup(
                    database.path.parent / "backups" / f"before-v{target}-{stamp}.sqlite3"
                )
            for version, name, sql in selected[len(rows) : target]:
                conn.execute("BEGIN IMMEDIATE")
                try:
                    if version == 1:
                        conn.execute(
                            "CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY,name TEXT NOT NULL,checksum TEXT NOT NULL,applied_at_utc TEXT NOT NULL) STRICT"
                        )
                    for statement in statements(sql):

                        def authorize(
                            action: int, a: str | None, b: str | None, d: str | None, t: str | None
                        ) -> int:
                            if action == sqlite3.SQLITE_PRAGMA and (
                                (a == "data_version" and b is None) or a == "quick_check"
                            ):
                                # ALTER TABLE ... CHECK invokes read-only quick_check(table).
                                return sqlite3.SQLITE_OK
                            return (
                                sqlite3.SQLITE_DENY
                                if action
                                in {
                                    sqlite3.SQLITE_TRANSACTION,
                                    sqlite3.SQLITE_SAVEPOINT,
                                    sqlite3.SQLITE_ATTACH,
                                    sqlite3.SQLITE_DETACH,
                                    sqlite3.SQLITE_PRAGMA,
                                }
                                else sqlite3.SQLITE_OK
                            )

                        conn.set_authorizer(authorize)
                        try:
                            conn.execute(statement)
                        finally:
                            conn.set_authorizer(None)
                    fault("migration_before_commit")
                    conn.execute(
                        "INSERT INTO schema_migrations VALUES(?,?,?,?)",
                        (
                            version,
                            name,
                            hashlib.sha256(sql.encode()).hexdigest(),
                            datetime.now(UTC).isoformat(),
                        ),
                    )
                    database.check(conn)
                    conn.execute("COMMIT")
                    fault("migration_after_commit")
                except BaseException:
                    if conn.in_transaction:
                        conn.execute("ROLLBACK")
                    raise
            database.check(conn)
            return target
        finally:
            conn.close()
