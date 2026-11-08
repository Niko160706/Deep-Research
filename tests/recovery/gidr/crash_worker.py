"""Trusted synthetic child process killed at an explicitly selected boundary."""

import os
import signal
import sys
from datetime import UTC, datetime
from pathlib import Path

from gidr.contracts.identity import Project
from gidr.infrastructure.artifacts.repository import (
    ArtifactRepository,
    CleanupRequest,
    ReferenceRequest,
)
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.audit.projection import AuditExporter
from gidr.infrastructure.db.authority import PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.db.migrate import migrate

CONTENT = b"Synthetic recovery content"
WHEN = datetime(2020, 1, 1, tzinfo=UTC)


def owner(authority, name="crash"):
    p = Project(
        project_id="PRJ-" + name,
        title="Synthetic crash fixture",
        owner="fixture",
        created_at_utc=WHEN,
    )

    def change(ctx):
        ctx.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            (p.project_id, p.title, p.owner, WHEN.isoformat()),
        )
        ctx.emit(
            event_for(
                ctx, component="scheduler", event_type="project.created", aggregate_id=p.project_id
            )
        )
        return p

    return authority.execute("CMD-owner-" + name, "TRACE-owner-" + name, p, Project, change)


def main():
    root = Path(sys.argv[1])
    stage = sys.argv[2]

    def kill(at):
        if at == stage:
            os.kill(os.getpid(), signal.SIGKILL)

    db = Database(root / "state/db.sqlite3")
    if stage.startswith("migration_"):
        migrate(db, fault=kill)
        raise RuntimeError("Selected crash boundary was not reached")
    migrate(db)
    authority = PersistenceAuthority(db)
    owner(authority)
    store = ArtifactStore(root / "artifacts")
    repo = ArtifactRepository(store, authority)
    if stage.startswith("projection_") or stage in {
        "before_projection_publish",
        "after_projection_publish",
    }:
        owner(authority, "second")
        AuditExporter(authority).export(root / "audit", fault=kill)
    elif stage.startswith("cleanup_"):
        ref = store.publish(CONTENT)
        finish = repo.finish_deletions
        repo.finish_deletions = lambda: finish(fault=kill)
        repo.cleanup("CMD-cleanup", "TRACE-cleanup", CleanupRequest(reference=ref, confirmed=True))
    else:
        ref = store.publish_chunks((CONTENT,), fault=kill)
        repo.reference(
            "CMD-reference",
            "TRACE-reference",
            ReferenceRequest(
                reference=ref, owner_type="PROJECT", owner_id="PRJ-crash", role="INPUT"
            ),
            fault=kill,
        )
    raise RuntimeError("Selected crash boundary was not reached")


if __name__ == "__main__":
    main()
