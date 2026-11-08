"""Metadata references commit through the same authority as the canonical event."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Literal

from gidr.contracts.common import ArtifactRef, Contract, Text
from gidr.infrastructure.artifacts.store import ArtifactError, ArtifactStore
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.db.authority import PersistenceAuthority, WriteContext
from gidr.infrastructure.localfs import sync_directory


class ReferenceRequest(Contract):
    reference: ArtifactRef
    owner_type: Literal[
        "PROJECT",
        "RUN",
        "SOURCE",
        "DOCUMENT",
        "CHUNK",
        "CLAIM",
        "EVIDENCE",
        "ASSESSMENT",
        "MODEL_EXECUTION",
        "REPORT",
        "TRANSLATION",
        "PLAN_VERSION",
    ]
    owner_id: Text
    role: Text


class CleanupRequest(Contract):
    reference: ArtifactRef
    confirmed: Literal[True]


class StorageResult(Contract):
    status: Literal["REFERENCED", "RELEASED", "DELETING", "DELETED", "PROTECTED", "QUARANTINED"]
    artifact_id: str


OWNERS = {
    "PROJECT": ("projects", "project_id"),
    "RUN": ("runs", "run_id"),
    "SOURCE": ("source_candidates", "source_id"),
    "DOCUMENT": ("documents", "document_id"),
    "CHUNK": ("chunks", "chunk_id"),
    "CLAIM": ("claims", "claim_id"),
    "EVIDENCE": ("evidence", "evidence_id"),
    "ASSESSMENT": ("assessments", "assessment_id"),
    "MODEL_EXECUTION": ("model_executions", "model_execution_id"),
    "REPORT": ("report_snapshots", "report_id"),
    "TRANSLATION": ("translations", "translation_id"),
    "PLAN_VERSION": ("plan_versions", "plan_version_id"),
}
# These structural FKs are a second safety check, not a substitute for artifact_refs.
CONTENT_FKS = {
    "plan_versions": ("snapshot_artifact_id",),
    "source_candidates": ("metadata_artifact_id",),
    "source_observations": ("artifact_id",),
    "task_attempts": ("output_artifact_id",),
    "documents": ("original_artifact_id",),
    "chunks": ("text_artifact_id",),
    "translations": ("translated_artifact_id",),
    "claims": ("proposition_artifact_id",),
    "evidence": ("excerpt_artifact_id",),
    "assessments": ("snapshot_artifact_id",),
    "critic_proposals": ("snapshot_artifact_id",),
    "plan_change_proposals": ("snapshot_artifact_id",),
    "model_executions": ("input_artifact_id", "output_artifact_id"),
    "model_attempts": ("output_artifact_id",),
    "provider_health": ("verification_artifact_id",),
    "report_snapshots": ("snapshot_artifact_id",),
    "query_cache": ("result_artifact_id",),
}


def has_references(context: WriteContext, artifact_id: str) -> bool:
    if context.execute(
        "SELECT 1 FROM artifact_refs WHERE artifact_id=? LIMIT 1", (artifact_id,)
    ).fetchone():
        return True
    return any(
        context.execute(
            f'SELECT 1 FROM "{table}" WHERE "{column}"=? LIMIT 1', (artifact_id,)
        ).fetchone()
        is not None
        for table, columns in CONTENT_FKS.items()
        for column in columns
    )


class ArtifactRepository:
    """Lock order: artifact OS lock -> DB lifecycle lock -> SQLite transaction."""

    def __init__(self, store: ArtifactStore, authority: PersistenceAuthority):
        self.store = store
        self.authority = authority
        self._prepared = threading.local()

    @contextmanager
    def prepared(self, references: tuple[ArtifactRef, ...]) -> Iterator[None]:
        """Verify bytes before BEGIN; hold the artifact lock through the domain commit."""
        with self.store.lock.hold():
            for reference in references:
                self.store.read(reference)
            previous: frozenset[ArtifactRef] = getattr(self._prepared, "references", frozenset())
            self._prepared.references = previous | frozenset(references)
            try:
                yield
            finally:
                self._prepared.references = previous

    def attach(self, context: WriteContext, request: ReferenceRequest) -> StorageResult:
        """Reusable inside the SAME transaction as a task/result/domain update."""
        ready: frozenset[ArtifactRef] = getattr(self._prepared, "references", frozenset())
        if request.reference not in ready:
            raise ValueError("ARTIFACT_NOT_PREPARED_OUTSIDE_TRANSACTION")
        ref = request.reference
        table, key = OWNERS[request.owner_type]
        if (
            context.execute(
                f'SELECT 1 FROM "{table}" WHERE "{key}"=?', (request.owner_id,)
            ).fetchone()
            is None
        ):
            raise ValueError("ARTIFACT_OWNER_MISSING")
        prior = context.execute(
            "SELECT sha256,size_bytes,status FROM artifact_catalog WHERE artifact_id=?",
            (ref.artifact_id,),
        ).fetchone()
        if prior is not None:
            if (
                prior["status"] != "AVAILABLE"
                or prior["sha256"] != ref.sha256
                or prior["size_bytes"] != ref.size_bytes
            ):
                raise ValueError("ARTIFACT_TOMBSTONED_OR_METADATA_MISMATCH")
        else:
            context.execute(
                "INSERT INTO artifact_catalog(artifact_id,sha256,size_bytes,category,status,created_at_utc) VALUES(?,?,?,'CONTENT','AVAILABLE',?)",
                (ref.artifact_id, ref.sha256, ref.size_bytes, datetime.now(UTC).isoformat()),
            )
        context.execute(
            "INSERT OR IGNORE INTO artifact_refs VALUES(?,?,?,?,?)",
            (
                ref.artifact_id,
                request.owner_type,
                request.owner_id,
                request.role,
                datetime.now(UTC).isoformat(),
            ),
        )
        context.emit(
            event_for(
                context,
                component="artifacts",
                event_type="artifact.referenced",
                aggregate_id=ref.artifact_id,
            )
        )
        return StorageResult(status="REFERENCED", artifact_id=ref.artifact_id)

    def reference(
        self,
        command_id: str,
        trace_id: str,
        request: ReferenceRequest,
        *,
        fault: Callable[[str], None] = lambda _: None,
    ) -> StorageResult:
        with self.prepared((request.reference,)):
            return self.authority.execute(
                command_id,
                trace_id,
                request,
                StorageResult,
                lambda context: self.attach(context, request),
                fault=fault,
            )

    def release(self, command_id: str, trace_id: str, request: ReferenceRequest) -> StorageResult:
        def change(context: WriteContext) -> StorageResult:
            context.execute(
                "DELETE FROM artifact_refs WHERE artifact_id=? AND owner_type=? AND owner_id=? AND role=?",
                (request.reference.artifact_id, request.owner_type, request.owner_id, request.role),
            )
            context.emit(
                event_for(
                    context,
                    component="artifacts",
                    event_type="artifact.reference_released",
                    aggregate_id=request.reference.artifact_id,
                )
            )
            return StorageResult(status="RELEASED", artifact_id=request.reference.artifact_id)

        with self.store.lock.hold():
            return self.authority.execute(command_id, trace_id, request, StorageResult, change)

    def cleanup(self, command_id: str, trace_id: str, request: CleanupRequest) -> StorageResult:
        """Storage primitive only; confirmation/retention policy is caller-owned (B12)."""
        with self.store.lock.hold():

            def change(context: WriteContext) -> StorageResult:
                ref = request.reference
                prior = context.execute(
                    "SELECT status,sha256,size_bytes FROM artifact_catalog WHERE artifact_id=?",
                    (ref.artifact_id,),
                ).fetchone()
                if (
                    prior is None
                    or prior["sha256"] != ref.sha256
                    or prior["size_bytes"] != ref.size_bytes
                ):
                    raise ValueError("CLEANUP_REQUIRES_CATALOGUED_ARTIFACT")
                if has_references(context, ref.artifact_id):
                    context.emit(
                        event_for(
                            context,
                            component="artifacts",
                            event_type="artifact.cleanup_protected",
                            aggregate_id=ref.artifact_id,
                        )
                    )
                    return StorageResult(status="PROTECTED", artifact_id=ref.artifact_id)
                context.execute(
                    "UPDATE artifact_catalog SET status='DELETING',deletion_epoch=deletion_epoch+1 WHERE artifact_id=? AND status<>'DELETED'",
                    (ref.artifact_id,),
                )
                context.execute(
                    "INSERT OR IGNORE INTO artifact_tombstones SELECT artifact_id,deletion_epoch,'USER_CLEANUP',? FROM artifact_catalog WHERE artifact_id=?",
                    (datetime.now(UTC).isoformat(), ref.artifact_id),
                )
                context.emit(
                    event_for(
                        context,
                        component="artifacts",
                        event_type="artifact.cleanup_marked",
                        aggregate_id=ref.artifact_id,
                    )
                )
                return StorageResult(status="DELETING", artifact_id=ref.artifact_id)

            result = self.authority.execute(command_id, trace_id, request, StorageResult, change)
            if result.status == "DELETING":
                self.finish_deletions()
                return StorageResult(status="DELETED", artifact_id=result.artifact_id)
            return result

    def finish_deletions(self, *, fault: Callable[[str], None] = lambda _: None) -> int:
        """Idempotent recovery of previously committed marks; never guesses consent."""
        count = 0
        with self.store.lock.hold():
            with self.authority.database.reader() as conn:
                rows = conn.execute(
                    "SELECT artifact_id,sha256,size_bytes FROM artifact_catalog WHERE status='DELETING'"
                ).fetchall()
            for row in rows:
                reference = ArtifactRef(
                    artifact_id=row["artifact_id"],
                    sha256=row["sha256"],
                    size_bytes=row["size_bytes"],
                )

                def change(
                    context: WriteContext, reference: ArtifactRef = reference
                ) -> StorageResult:
                    if has_references(context, reference.artifact_id):
                        raise ValueError("DELETING_ARTIFACT_ACQUIRED_REFERENCE")
                    # This local unlink is short and recoverable; no network/parser is called.
                    fault("cleanup_before_unlink")
                    self.store.path(reference.sha256).unlink(missing_ok=True)
                    fault("cleanup_after_unlink")
                    sync_directory(self.store.path(reference.sha256).parent)
                    context.execute(
                        "UPDATE artifact_catalog SET status='DELETED' WHERE artifact_id=?",
                        (reference.artifact_id,),
                    )
                    context.emit(
                        event_for(
                            context,
                            component="artifacts",
                            event_type="artifact.deleted",
                            aggregate_id=reference.artifact_id,
                        )
                    )
                    return StorageResult(status="DELETED", artifact_id=reference.artifact_id)

                self.authority.execute(
                    "CMD-cleanup-" + reference.sha256,
                    "TRACE-cleanup-" + reference.sha256,
                    reference,
                    StorageResult,
                    change,
                )
                count += 1
        return count

    def inventory(self) -> dict[str, tuple[str, ...]]:
        """Orphans are retained, not treated as succeeded work or silently deleted."""
        with self.store.lock.hold(), self.authority.database.reader() as conn:
            known = {
                row["artifact_id"]: row["status"]
                for row in conn.execute("SELECT artifact_id,status FROM artifact_catalog")
            }
            present = {ref.artifact_id for ref in self.store.published()}
            return {
                "orphans": tuple(sorted(present - set(known))),
                "tombstoned_bytes": tuple(
                    sorted(key for key in present if known.get(key) in {"DELETING", "DELETED"})
                ),
                "missing": tuple(
                    sorted(
                        key
                        for key, status in known.items()
                        if status == "AVAILABLE" and key not in present
                    )
                ),
                "staged": tuple(
                    sorted(path.name for path in self.store.staging.iterdir() if path.is_file())
                ),
            }

    def read(self, reference: ArtifactRef) -> bytes:
        with self.store.lock.hold():
            with self.authority.database.reader() as conn:
                row = conn.execute(
                    "SELECT status FROM artifact_catalog WHERE artifact_id=? AND sha256=? AND size_bytes=?",
                    (reference.artifact_id, reference.sha256, reference.size_bytes),
                ).fetchone()
            if row is None or row["status"] != "AVAILABLE":
                raise ArtifactError("CATALOGUED_ARTIFACT_UNAVAILABLE")
            try:
                return self.store.read(reference)
            except ArtifactError:

                def change(context: WriteContext) -> StorageResult:
                    context.execute(
                        "UPDATE artifact_catalog SET status='QUARANTINED' WHERE artifact_id=?",
                        (reference.artifact_id,),
                    )
                    context.execute(
                        "UPDATE documents SET availability='QUARANTINED' WHERE original_artifact_id=?",
                        (reference.artifact_id,),
                    )
                    context.emit(
                        event_for(
                            context,
                            component="security",
                            event_type="artifact.integrity_failed",
                            aggregate_id=reference.artifact_id,
                        )
                    )
                    return StorageResult(status="QUARANTINED", artifact_id=reference.artifact_id)

                self.authority.execute(
                    "CMD-integrity-" + reference.sha256,
                    "TRACE-integrity-" + reference.sha256,
                    reference,
                    StorageResult,
                    change,
                )
                raise
