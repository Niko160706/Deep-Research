"""Real local SQLite/filesystem tests; all records are synthetic, never live."""

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest

from gidr.contracts.common import Contract
from gidr.contracts.identity import Project
from gidr.infrastructure.artifacts.repository import (
    ArtifactRepository,
    CleanupRequest,
    ReferenceRequest,
)
from gidr.infrastructure.artifacts.store import ArtifactError, ArtifactStore
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.audit.projection import AuditExporter, events
from gidr.infrastructure.db.authority import CommitError, IdempotencyConflict, PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.db.index import IndexEntry, LexicalIndex
from gidr.infrastructure.db.migrate import bundled_migrations, migrate

WHEN = datetime(2020, 1, 1, tzinfo=UTC)


def owner(authority, name="fixture"):
    project = Project(
        project_id="PRJ-" + name, title="Synthetic test only", owner="fixture", created_at_utc=WHEN
    )

    def change(ctx):
        ctx.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            (project.project_id, project.title, project.owner, WHEN.isoformat()),
        )
        ctx.emit(
            event_for(
                ctx,
                component="scheduler",
                event_type="project.created",
                aggregate_id=project.project_id,
            )
        )
        return project

    return authority.execute("CMD-owner-" + name, "TRACE-owner-" + name, project, Project, change)


@pytest.fixture
def system(tmp_path):
    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db)
    authority = PersistenceAuthority(db)
    store = ArtifactStore(tmp_path / "artifacts")
    owner(authority)
    return db, authority, store, ArtifactRepository(store, authority)


def source(authority, name="fixture"):
    p = Project(
        project_id="PRJ-" + name, title="Synthetic source", owner="fixture", created_at_utc=WHEN
    )

    def change(ctx):
        ctx.execute(
            "INSERT INTO sessions VALUES(?,?,0,?)", ("SES-" + name, "PRJ-fixture", WHEN.isoformat())
        )
        ctx.execute(
            "INSERT INTO runs(run_id,project_id,session_id,state,created_at_utc,updated_at_utc) VALUES(?,?,?,'CREATED',?,?)",
            ("RUN-" + name, "PRJ-fixture", "SES-" + name, WHEN.isoformat(), WHEN.isoformat()),
        )
        ctx.execute(
            "INSERT INTO source_candidates(source_id,run_id,provider_id,canonical_url,title,discovered_at_utc) VALUES(?,?,?,?,?,?)",
            (
                "SRC-" + name,
                "RUN-" + name,
                "fixture",
                "https://example.org/synthetic",
                "Synthetic",
                WHEN.isoformat(),
            ),
        )
        ctx.emit(
            event_for(
                ctx,
                component="retrieval",
                event_type="source.created",
                aggregate_id="SRC-" + name,
                run_id="RUN-" + name,
            )
        )
        return p

    authority.execute("CMD-source-" + name, "TRACE-source-" + name, p, Project, change)


def test_inventory_pragmas_and_reader_is_read_only(system):
    db, *_ = system
    required = [
        "schema_migrations",
        "projects",
        "sessions",
        "runs",
        "active_run_gate",
        "plans",
        "plan_versions",
        "questions",
        "grants",
        "budget_ledger",
        "tasks",
        "task_dependencies",
        "task_attempts",
        "leads",
        "lead_edges",
        "frontier_scores",
        "source_candidates",
        "source_observations",
        "source_aliases",
        "source_dependencies",
        "documents",
        "chunks",
        "translations",
        "claims",
        "claim_relations",
        "evidence",
        "assessments",
        "assessment_evidence",
        "critic_proposals",
        "plan_change_proposals",
        "model_executions",
        "model_attempts",
        "provider_health",
        "provider_cooldowns",
        "artifact_catalog",
        "artifact_refs",
        "artifact_tombstones",
        "audit_events",
        "processed_commands",
        "coverage_observations",
        "report_snapshots",
        "runtime_lock_observations",
        "query_cache",
        "dedup_fingerprints",
        "source_fts",
    ]
    with db.reader() as conn:
        assert set(required) <= {
            r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA trusted_schema").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM projects")
        db.check(conn)
    conn = db._connect()
    try:
        assert conn.autocommit is True and conn.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        conn.close()


def test_same_command_multiple_events_and_concurrent_replay(system):
    db, authority, *_ = system
    req = Project(project_id="PRJ-replay", title="Synthetic", owner="fixture", created_at_utc=WHEN)
    calls = []

    def change(ctx):
        calls.append(1)
        ctx.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            (req.project_id, req.title, req.owner, WHEN.isoformat()),
        )
        for kind in ["project.created", "project.reviewed"]:
            ctx.emit(
                event_for(ctx, component="scheduler", event_type=kind, aggregate_id=req.project_id)
            )
        return req

    def submit(_):
        return authority.execute("CMD-repeat", "TRACE-repeat", req, Project, change)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(x == req for x in pool.map(submit, range(12)))
    assert len(calls) == 1
    with db.reader() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE command_id='CMD-repeat'"
            ).fetchone()[0]
            == 2
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM processed_commands WHERE command_id='CMD-repeat'"
            ).fetchone()[0]
            == 1
        )
    with pytest.raises(IdempotencyConflict):
        authority.execute(
            "CMD-repeat",
            "TRACE-repeat",
            Project.model_validate({**req.model_dump(), "title": "Different"}),
            Project,
            change,
        )


@pytest.mark.parametrize(
    "failure", ["exception", "foreign_key", "missing_event", "hidden_commit", "bad_result"]
)
def test_atomic_rollback_and_sanitized_failures(system, failure):
    db, authority, *_ = system
    req = Project(
        project_id="PRJ-rollback", title="Synthetic", owner="fixture", created_at_utc=WHEN
    )

    def change(ctx):
        ctx.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            (req.project_id, req.title, req.owner, WHEN.isoformat()),
        )
        if failure == "exception":
            raise RuntimeError("SYNTHETIC-SECRET-CANARY")
        if failure == "foreign_key":
            ctx.execute(
                "INSERT INTO sessions VALUES(?,?,0,?)", ("SES-bad", "PRJ-missing", WHEN.isoformat())
            )
        if failure == "hidden_commit":
            ctx.execute("-- hidden\nCOMMIT")
        if failure != "missing_event":
            ctx.emit(
                event_for(
                    ctx,
                    component="scheduler",
                    event_type="test.created",
                    aggregate_id=req.project_id,
                )
            )
        return object() if failure == "bad_result" else req

    with pytest.raises(CommitError) as caught:
        authority.execute("CMD-rollback", "TRACE-rollback", req, Project, change)
    assert "SYNTHETIC-SECRET-CANARY" not in str(caught.value)
    with db.reader() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM projects WHERE project_id='PRJ-rollback'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE command_id='CMD-rollback'"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM processed_commands WHERE command_id='CMD-rollback'"
            ).fetchone()[0]
            == 0
        )


def test_immutable_events_processed_receipts_and_expired_context(system):
    db, authority, *_ = system
    req = Contract()
    saved = []

    def change(ctx):
        saved.append(ctx)
        ctx.emit(
            event_for(
                ctx, component="scheduler", event_type="test.marker", aggregate_id="TEST-marker"
            )
        )
        return req

    authority.execute("CMD-immutable", "TRACE-immutable", req, Contract, change)
    with pytest.raises(CommitError, match="EXPIRED"):
        saved[0].execute("SELECT 1")
    for statement in [
        "UPDATE audit_events SET component='other'",
        "DELETE FROM audit_events",
        "UPDATE processed_commands SET result_type='other'",
        "DELETE FROM processed_commands",
    ]:
        with pytest.raises(CommitError):
            authority.execute(
                "CMD-tamper",
                "TRACE-tamper",
                req,
                Contract,
                lambda ctx, statement=statement: (ctx.execute(statement), req)[1],
            )
    with db.reader() as conn:
        db.check(conn)


def test_upgrade_checksum_future_schema_and_rollback(tmp_path):
    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db, target_version=1)
    owner(PersistenceAuthority(db))
    assert migrate(db) == 2
    assert list((db.path.parent / "backups").glob("*.sqlite3"))
    original = bundled_migrations()
    with pytest.raises(RuntimeError, match="CHECKSUM"):
        migrate(db, migrations=((1, original[0][1], original[0][2] + "\n-- changed"), original[1]))
    with pytest.raises(RuntimeError, match="NEWER"):
        migrate(db, target_version=1)
    broken = (
        *original,
        (
            3,
            "0003_broken.sql",
            "CREATE TABLE transient_test(id INTEGER);\nSELECT missing_column FROM projects;\n",
        ),
    )
    with pytest.raises(sqlite3.OperationalError):
        migrate(db, migrations=broken)
    with db.reader() as conn:
        assert (
            conn.execute("SELECT 1 FROM sqlite_master WHERE name='transient_test'").fetchone()
            is None
        )
        assert conn.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM projects").fetchone()[0] == 1
        db.check(conn)


def test_online_backup_includes_active_wal_and_restore_requires_new_target(system, tmp_path):
    db, authority, *_ = system
    with db.reader() as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM projects").fetchall()
        owner(authority, "second")
        assert db.path.with_name(db.path.name + "-wal").stat().st_size > 0
        backup = tmp_path / "backups/snapshot.db"
        db.backup(backup)
        restored = Database.restore_new(backup, tmp_path / "restored/new.db")
        with restored.reader() as conn:
            assert conn.execute("SELECT count(*) FROM projects").fetchone()[0] == 2
            restored.check(conn)
        with pytest.raises(FileExistsError):
            Database.restore_new(backup, restored.path)


def test_artifact_publish_dedup_references_gc_and_tombstone(system):
    _db, authority, store, repo = system
    one = store.publish(b"Synthetic immutable bytes")
    two = store.publish(b"Synthetic immutable bytes")
    assert one == two and len(store.published()) == 1
    owner(authority, "second")
    first = ReferenceRequest(
        reference=one, owner_type="PROJECT", owner_id="PRJ-fixture", role="INPUT"
    )
    second = ReferenceRequest(
        reference=one, owner_type="PROJECT", owner_id="PRJ-second", role="INPUT"
    )
    repo.reference("CMD-ref1", "TRACE-ref1", first)
    repo.reference("CMD-ref2", "TRACE-ref2", second)
    repo.release("CMD-release1", "TRACE-release1", first)
    assert (
        repo.cleanup("CMD-gc1", "TRACE-gc1", CleanupRequest(reference=one, confirmed=True)).status
        == "PROTECTED"
    )
    assert repo.read(one) == b"Synthetic immutable bytes"
    repo.release("CMD-release2", "TRACE-release2", second)
    assert (
        repo.cleanup("CMD-gc2", "TRACE-gc2", CleanupRequest(reference=one, confirmed=True)).status
        == "DELETED"
    )
    assert repo.finish_deletions() == 0
    with pytest.raises(ArtifactError):
        repo.read(one)
    store.publish(b"Synthetic immutable bytes")
    with pytest.raises(CommitError):
        repo.reference("CMD-resurrect", "TRACE-resurrect", first)
    assert repo.inventory()["tombstoned_bytes"] == (one.artifact_id,)


def test_corruption_quarantines_and_never_overwrites_history(system):
    db, _, store, repo = system
    ref = store.publish(b"Synthetic original bytes")
    repo.reference(
        "CMD-ref",
        "TRACE-ref",
        ReferenceRequest(reference=ref, owner_type="PROJECT", owner_id="PRJ-fixture", role="INPUT"),
    )
    store.path(ref.sha256).write_bytes(b"Corrupted bytes")
    with pytest.raises(ArtifactError, match="INTEGRITY"):
        repo.read(ref)
    assert list(store.quarantine.iterdir())
    with db.reader() as conn:
        assert (
            conn.execute(
                "SELECT status FROM artifact_catalog WHERE artifact_id=?", (ref.artifact_id,)
            ).fetchone()[0]
            == "QUARANTINED"
        )
    assert any(e.event_type == "artifact.integrity_failed" for e in events(db))


def test_orphans_are_retained_and_failed_reference_never_succeeds(system):
    db, _, store, repo = system
    ref = store.publish(b"Synthetic orphan")
    assert repo.inventory()["orphans"] == (ref.artifact_id,)
    with pytest.raises(CommitError):
        repo.reference(
            "CMD-bad-owner",
            "TRACE-bad-owner",
            ReferenceRequest(
                reference=ref, owner_type="PROJECT", owner_id="PRJ-missing", role="OUTPUT"
            ),
        )
    with db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM artifact_catalog").fetchone()[0] == 0
    assert store.read(ref) == b"Synthetic orphan"


def test_fts_upsert_delete_rebuild_and_unicode(system):
    db, authority, *_ = system
    source(authority)
    index = LexicalIndex(authority)
    index.upsert(
        "CMD-index1",
        "TRACE-index1",
        IndexEntry(source_id="SRC-fixture", title="日本語", extracted_text="provenance oldterm"),
    )
    assert index.search("provenance")[0]["title"] == "日本語"
    index.upsert(
        "CMD-index-markdown",
        "TRACE-index-markdown",
        IndexEntry(source_id="SRC-fixture", extracted_text="[citation] provenance oldterm"),
    )
    with db.reader() as conn:
        assert (
            conn.execute("SELECT extracted_text FROM source_index").fetchone()[0]
            == "[citation] provenance oldterm"
        )
    index.upsert(
        "CMD-index2",
        "TRACE-index2",
        IndexEntry(source_id="SRC-fixture", title="日本語", extracted_text="provenance newterm"),
    )
    assert not index.search("oldterm") and index.search("newterm")

    def corrupt(ctx):
        ctx.execute("INSERT INTO source_fts(source_fts) VALUES('delete-all')")
        ctx.emit(
            event_for(
                ctx,
                component="indexing",
                event_type="test.index_cleared",
                aggregate_id="INDEX-source",
            )
        )
        return Contract()

    authority.execute("CMD-index-corrupt", "TRACE-index-corrupt", Contract(), Contract, corrupt)
    assert not index.search("provenance")
    index.rebuild("CMD-rebuild", "TRACE-rebuild")
    assert index.search("provenance")
    index.remove("CMD-remove", "TRACE-remove", IndexEntry(source_id="SRC-fixture"))
    assert not index.search("provenance")
    with db.reader() as conn:
        db.check(conn)


@pytest.mark.parametrize("stage", ["before_projection_publish", "after_projection_publish"])
def test_projection_rebuild_after_interruption(system, tmp_path, stage):
    db, authority, *_ = system
    exporter = AuditExporter(authority)
    root = tmp_path / "audit"
    previous = exporter.export(root)
    owner(authority, "second")

    def fail(at):
        if at == stage:
            raise RuntimeError("Synthetic interruption")

    with pytest.raises(RuntimeError):
        exporter.export(root, fault=fail)
    if stage == "before_projection_publish":
        assert exporter.current(root) == previous
    current = exporter.export(root)
    assert exporter.current(root) == current
    lines = [json.loads(line) for line in (current / "events.jsonl").read_text().splitlines()]
    assert [x["event_id"] for x in lines] == [e.event_id for e in events(db)]
    assert len(lines) == len({x["event_id"] for x in lines})
    assert len(list(current.glob("*.jsonl"))) == 9
    (current / "events.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        exporter.current(root)
    rebuilt = exporter.export(root)
    assert exporter.current(root) == rebuilt


def test_secret_canaries_blocked_before_commit_and_export(system, tmp_path):
    db, *_ = system
    authority = PersistenceAuthority(db, SecretPolicy(("SYNTHETIC-SECRET-CANARY",)))
    request = Project(
        project_id="PRJ-secret",
        title="SYNTHETIC-SECRET-CANARY",
        owner="fixture",
        created_at_utc=WHEN,
    )
    with pytest.raises(ValueError, match="SECRET"):
        authority.execute("CMD-secret", "TRACE-secret", request, Project, lambda _: request)
    assert b"SYNTHETIC-SECRET-CANARY" not in db.path.read_bytes()
    root = tmp_path / "audit"
    exporter = AuditExporter(authority)
    exporter.export(root)
    assert all(
        "SYNTHETIC-SECRET-CANARY" not in p.read_text()
        for p in exporter.current(root).glob("*.jsonl")
    )


@pytest.mark.parametrize("rollback", [False, True])
def test_domain_and_artifact_reference_share_one_transaction(system, rollback):
    db, authority, store, repo = system
    ref = store.publish(b"Synthetic composed commit")
    request = ReferenceRequest(
        reference=ref, owner_type="PROJECT", owner_id="PRJ-composed", role="INPUT"
    )

    def change(ctx):
        ctx.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            ("PRJ-composed", "Synthetic", "fixture", WHEN.isoformat()),
        )
        repo.attach(ctx, request)
        if rollback:
            raise RuntimeError("Synthetic rollback after attachment")
        return Contract()

    with repo.prepared((ref,)):
        if rollback:
            with pytest.raises(CommitError):
                authority.execute("CMD-composed", "TRACE-composed", request, Contract, change)
        else:
            authority.execute("CMD-composed", "TRACE-composed", request, Contract, change)
    with db.reader() as conn:
        for query in (
            "SELECT count(*) FROM projects WHERE project_id='PRJ-composed'",
            "SELECT count(*) FROM artifact_catalog",
            "SELECT count(*) FROM artifact_refs",
            "SELECT count(*) FROM audit_events WHERE command_id='CMD-composed'",
            "SELECT count(*) FROM processed_commands WHERE command_id='CMD-composed'",
        ):
            assert conn.execute(query).fetchone()[0] == int(not rollback)
    assert store.read(ref) == b"Synthetic composed commit"


@pytest.mark.parametrize("protection", ["structural_fk", "event_reference"])
def test_cleanup_protects_structural_and_audit_references(system, protection):
    db, authority, store, repo = system
    source(authority)
    ref = store.publish(b"Synthetic protected metadata")
    request = ReferenceRequest(
        reference=ref, owner_type="PROJECT", owner_id="PRJ-fixture", role="INPUT"
    )
    repo.reference("CMD-protect-ref", "TRACE-protect-ref", request)

    def change(ctx):
        event = event_for(
            ctx, component="artifacts", event_type="test.protect", aggregate_id=ref.artifact_id
        )
        if protection == "structural_fk":
            ctx.execute(
                "UPDATE source_candidates SET metadata_artifact_id=? WHERE source_id=?",
                (ref.artifact_id, "SRC-fixture"),
            )
        else:
            event = type(event).model_validate({**event.model_dump(), "output_refs": (ref,)})
        ctx.emit(event)
        return Contract()

    authority.execute("CMD-protection", "TRACE-protection", Contract(), Contract, change)
    repo.release("CMD-drop-explicit", "TRACE-drop-explicit", request)
    assert (
        repo.cleanup(
            "CMD-protected-gc", "TRACE-protected-gc", CleanupRequest(reference=ref, confirmed=True)
        ).status
        == "PROTECTED"
    )
    assert repo.read(ref) == b"Synthetic protected metadata"
    with db.reader() as conn:
        db.check(conn)


def test_migration_cannot_hide_early_commit_in_comments(tmp_path):
    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db)
    hidden = (
        3,
        "0003_hidden.sql",
        "CREATE TABLE transient_test(id INTEGER);\n-- bypass attempt\nCOMMIT;\n",
    )
    with pytest.raises(sqlite3.DatabaseError):
        migrate(db, migrations=(*bundled_migrations(), hidden))
    with db.reader() as conn:
        assert (
            conn.execute("SELECT 1 FROM sqlite_master WHERE name='transient_test'").fetchone()
            is None
        )
        assert conn.execute("SELECT max(version) FROM schema_migrations").fetchone()[0] == 2


def test_secret_sql_parameters_rejected_before_sqlite_sees_them(system):
    db, *_ = system
    authority = PersistenceAuthority(db, SecretPolicy(("SYNTHETIC-SECRET-CANARY",)))

    def change(ctx):
        ctx.execute("UPDATE projects SET title=?", ("SYNTHETIC-SECRET-CANARY",))
        return Contract()

    with pytest.raises(CommitError, match="ValueError"):
        authority.execute("CMD-secret-sql", "TRACE-secret-sql", Contract(), Contract, change)
    for path in db.path.parent.glob("db.sqlite3*"):
        assert b"SYNTHETIC-SECRET-CANARY" not in path.read_bytes()
