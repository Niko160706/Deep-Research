"""SIGKILL at real filesystem/SQLite commit boundaries, no live research."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gidr.infrastructure.artifacts.repository import ArtifactRepository, ReferenceRequest
from gidr.infrastructure.artifacts.store import ArtifactError, ArtifactStore
from gidr.infrastructure.audit.projection import AuditExporter, events
from gidr.infrastructure.db.authority import PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.db.migrate import migrate

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="POSIX SIGKILL exercised on Linux; Windows not claimed"
)
WORKER = Path(__file__).with_name("crash_worker.py")


def support():
    spec = importlib.util.spec_from_file_location("gidr_crash_test_support", WORKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def launch(root, stage):
    p = subprocess.run(
        [sys.executable, "-I", str(WORKER), str(root), stage],
        cwd=root,
        env={"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert p.returncode == -9, (p.returncode, p.stdout, p.stderr)


@pytest.mark.parametrize("stage", ["staged", "published", "before_commit", "after_commit"])
def test_blob_and_reference_reconcile_after_sigkill(tmp_path, stage):
    helper = support()
    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db)
    authority = PersistenceAuthority(db)
    helper.owner(authority)
    launch(tmp_path, stage)
    migrate(db)  # Same schema; actual OS locks are reacquirable despite stale lock files.
    store = ArtifactStore(tmp_path / "artifacts")
    repo = ArtifactRepository(store, authority)
    with db.reader() as conn:
        expected = 1 if stage == "after_commit" else 0
        assert conn.execute("SELECT count(*) FROM artifact_catalog").fetchone()[0] == expected
        assert (
            conn.execute(
                "SELECT count(*) FROM processed_commands WHERE command_id='CMD-reference'"
            ).fetchone()[0]
            == expected
        )
        db.check(conn)
    if stage == "staged":
        assert repo.inventory()["staged"] and not store.published()
    elif stage != "after_commit":
        assert len(repo.inventory()["orphans"]) == 1
    ref = store.publish(helper.CONTENT)
    request = ReferenceRequest(
        reference=ref, owner_type="PROJECT", owner_id="PRJ-crash", role="INPUT"
    )
    repo.reference("CMD-reference", "TRACE-reference", request)
    repo.reference("CMD-reference", "TRACE-reference", request)
    assert repo.read(ref) == helper.CONTENT
    with db.reader() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM artifact_refs WHERE owner_type='PROJECT'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE command_id='CMD-reference'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("stage", ["migration_before_commit", "migration_after_commit"])
def test_migration_kill_preserves_last_committed_version(tmp_path, stage):
    launch(tmp_path, stage)
    db = Database(tmp_path / "state/db.sqlite3")
    assert migrate(db) == 2
    with db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM schema_migrations").fetchone()[0] == 2
        db.check(conn)


@pytest.mark.parametrize("stage", ["before_projection_publish", "after_projection_publish"])
def test_projection_kill_rebuilds_without_duplicate_events(tmp_path, stage):
    helper = support()
    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db)
    authority = PersistenceAuthority(db)
    helper.owner(authority)
    exporter = AuditExporter(authority)
    old = exporter.export(tmp_path / "audit")
    launch(tmp_path, stage)
    if stage == "before_projection_publish":
        assert exporter.current(tmp_path / "audit") == old
    rebuilt = exporter.export(tmp_path / "audit")
    lines = [json.loads(line) for line in (rebuilt / "events.jsonl").read_text().splitlines()]
    assert [x["event_id"] for x in lines] == [e.event_id for e in events(db)]
    assert len(lines) == 2 and len({x["event_id"] for x in lines}) == 2


def test_cleanup_kill_after_unlink_finishes_committed_mark(tmp_path):
    helper = support()
    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db)
    authority = PersistenceAuthority(db)
    helper.owner(authority)
    store = ArtifactStore(tmp_path / "artifacts")
    repo = ArtifactRepository(store, authority)
    ref = store.publish(helper.CONTENT)
    request = ReferenceRequest(
        reference=ref, owner_type="PROJECT", owner_id="PRJ-crash", role="INPUT"
    )
    repo.reference("CMD-initial-ref", "TRACE-initial-ref", request)
    repo.release("CMD-release", "TRACE-release", request)
    launch(tmp_path, "cleanup_after_unlink")
    with db.reader() as conn:
        assert conn.execute("SELECT status FROM artifact_catalog").fetchone()[0] == "DELETING"
    assert not store.path(ref.sha256).exists()
    assert repo.finish_deletions() == 1
    assert repo.finish_deletions() == 0
    with pytest.raises(ArtifactError):
        repo.read(ref)
    with db.reader() as conn:
        assert conn.execute("SELECT status FROM artifact_catalog").fetchone()[0] == "DELETED"
        db.check(conn)
