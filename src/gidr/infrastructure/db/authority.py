"""The single transactional commit primitive; domain command policy is B03."""

from __future__ import annotations

import hashlib
import inspect
import json
import sqlite3
from collections.abc import Callable, Sequence
from datetime import UTC, datetime

from pydantic import TypeAdapter

from gidr.contracts.common import CommandId, Contract, TraceId, parse_contract
from gidr.contracts.control import AuditEvent
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.db.database import Database

type SQLValue = str | int | float | bytes | None


class CommitError(RuntimeError):
    pass


class IdempotencyConflict(CommitError):
    pass


class WriteContext:
    def __init__(
        self, conn: sqlite3.Connection, command_id: str, trace_id: str, policy: SecretPolicy
    ):
        self._conn = conn
        self.command_id = command_id
        self.trace_id = trace_id
        self._policy = policy
        self.events = 0
        self.active = True

    def execute(self, sql: str, parameters: Sequence[SQLValue] = ()) -> sqlite3.Cursor:
        """Trusted repository SQL only, never provider/LLM SQL. No transaction control."""
        if not self.active or not self._conn.in_transaction:
            raise CommitError("WRITE_CONTEXT_EXPIRED")
        self._policy.validate_text(sql)
        for value in parameters:
            if isinstance(value, (str, bytes)):
                text = value if isinstance(value, str) else value.decode("utf-8", errors="replace")
                self._policy.validate_text(text)
                if text.lstrip().startswith(("{", "[")):
                    try:
                        json.loads(text)
                    except ValueError:
                        pass  # Original source text may begin with Markdown or braces.
                    else:
                        self._policy.validate_json(text)
        return self._conn.execute(sql, parameters)

    def emit(self, event: AuditEvent) -> None:
        event = AuditEvent.model_validate(event)
        if (
            event.command_id != self.command_id
            or event.trace_id != self.trace_id
            or event.seq is not None
            or event.committed_at_utc is not None
        ):
            raise CommitError("EVENT_ENVELOPE_MISMATCH")
        payload = event.model_dump_json(exclude={"seq", "committed_at_utc"})
        if len(payload.encode()) > 65536:
            raise CommitError("EVENT_TOO_LARGE_USE_ARTIFACT_REF")
        self._policy.validate_json(payload)
        self.execute(
            "INSERT INTO audit_events(event_id,command_id,run_id,trace_id,aggregate_type,aggregate_id,revision_before,revision_after,component,event_type,event_payload_json,committed_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                event.event_id,
                self.command_id,
                event.run_id,
                self.trace_id,
                event.aggregate_type,
                event.aggregate_id,
                event.revision_before,
                event.revision_after,
                event.component,
                event.event_type,
                payload,
                datetime.now(UTC).isoformat(),
            ),
        )
        self.events += 1
        for role, references in (("INPUT", event.input_refs), ("OUTPUT", event.output_refs)):
            for reference in references:
                row = self.execute(
                    "SELECT sha256,size_bytes,status FROM artifact_catalog WHERE artifact_id=?",
                    (reference.artifact_id,),
                ).fetchone()
                if (
                    row is None
                    or row["sha256"] != reference.sha256
                    or row["size_bytes"] != reference.size_bytes
                    or row["status"] != "AVAILABLE"
                ):
                    raise CommitError("EVENT_ARTIFACT_REFERENCE_INVALID")
                self.execute(
                    "INSERT OR IGNORE INTO artifact_refs VALUES(?,?,?,?,?)",
                    (
                        reference.artifact_id,
                        "AUDIT_EVENT",
                        event.event_id,
                        role,
                        datetime.now(UTC).isoformat(),
                    ),
                )


class PersistenceAuthority:
    """Only this boundary opens data write transactions; no network or async work."""

    def __init__(self, database: Database, policy: SecretPolicy | None = None):
        self.database = database
        self.policy = policy if policy is not None else SecretPolicy()

    def replay[T: Contract](
        self, command_id: str, request: Contract, result_type: type[T]
    ) -> T | None:
        """Replay before reopening referenced bytes; execute still rechecks in transaction."""
        TypeAdapter(CommandId).validate_python(command_id, strict=True)
        encoded = json.dumps(
            {
                "type": type(request).__module__ + "." + type(request).__qualname__,
                "payload": request.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        self.policy.validate_json(encoded)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self.database.reader() as conn:
            row = conn.execute(
                "SELECT request_sha256,result_type,result_json FROM processed_commands WHERE command_id=?",
                (command_id,),
            ).fetchone()
        if row is None:
            return None
        if (
            row["request_sha256"] != digest
            or row["result_type"] != result_type.__module__ + "." + result_type.__qualname__
        ):
            raise IdempotencyConflict("COMMAND_ID_REUSED_WITH_DIFFERENT_REQUEST")
        return parse_contract(result_type, row["result_json"])

    def execute[T: Contract](
        self,
        command_id: str,
        trace_id: str,
        request: Contract,
        result_type: type[T],
        operation: Callable[[WriteContext], T],
        *,
        fault: Callable[[str], None] = lambda _: None,
    ) -> T:
        TypeAdapter(CommandId).validate_python(command_id, strict=True)
        TypeAdapter(TraceId).validate_python(trace_id, strict=True)
        if inspect.iscoroutinefunction(operation):
            raise CommitError("ASYNC_OPERATION_FORBIDDEN")
        request_json = json.dumps(
            {
                "type": type(request).__module__ + "." + type(request).__qualname__,
                "payload": request.model_dump(mode="json"),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(request_json.encode()) > 2 * 1024 * 1024:
            raise CommitError("COMMAND_METADATA_TOO_LARGE_USE_ARTIFACT_REFS")
        self.policy.validate_json(request_json)
        digest = hashlib.sha256(request_json.encode()).hexdigest()
        result_name = result_type.__module__ + "." + result_type.__qualname__
        with self.database.lifecycle.hold():
            conn = self.database._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                prior = conn.execute(
                    "SELECT request_sha256,result_type,result_json FROM processed_commands WHERE command_id=?",
                    (command_id,),
                ).fetchone()
                if prior is not None:
                    if prior["request_sha256"] != digest or prior["result_type"] != result_name:
                        raise IdempotencyConflict("COMMAND_ID_REUSED_WITH_DIFFERENT_REQUEST")
                    result = parse_contract(result_type, prior["result_json"])
                    conn.execute("COMMIT")
                    return result
                context = WriteContext(conn, command_id, trace_id, self.policy)

                # Deny hidden COMMIT/SAVEPOINT/ATTACH even behind SQL comments.
                def authorize(
                    action: int,
                    arg1: str | None,
                    arg2: str | None,
                    db: str | None,
                    trigger: str | None,
                ) -> int:
                    if action == sqlite3.SQLITE_PRAGMA and arg1 == "data_version" and arg2 is None:
                        return sqlite3.SQLITE_OK  # FTS5 reads this, never mutates a pragma.
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
                    result = result_type.model_validate(operation(context))
                    if context.events < 1:
                        raise CommitError("COMMAND_REQUIRES_CANONICAL_EVENT")
                finally:
                    context.active = False
                    conn.set_authorizer(None)
                encoded = result.model_dump_json()
                self.policy.validate_json(encoded)
                if len(encoded.encode()) > 131072:
                    raise CommitError("RESULT_TOO_LARGE_USE_ARTIFACT_REF")
                conn.execute(
                    "INSERT INTO processed_commands VALUES(?,?,?,?,?)",
                    (command_id, digest, result_name, encoded, datetime.now(UTC).isoformat()),
                )
                fault("before_commit")
                conn.execute("COMMIT")
                fault("after_commit")
                return result
            except BaseException as error:
                conn.set_authorizer(None)
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                if isinstance(error, CommitError) or not isinstance(error, Exception):
                    raise
                raise CommitError(
                    getattr(error, "sqlite_errorname", type(error).__name__)
                ) from None
            finally:
                conn.close()

    def checkpoint_projection(
        self, projection_id: str, last_seq: int, generation: str, manifest_sha256: str
    ) -> None:
        """Derived cursor only; does not recursively create a domain audit event."""
        from gidr.contracts.common import Digest, Name

        TypeAdapter(Name).validate_python(projection_id, strict=True)
        TypeAdapter(Digest).validate_python(manifest_sha256, strict=True)
        if (
            last_seq < 0
            or len(generation) != 32
            or any(c not in "0123456789abcdef" for c in generation)
        ):
            raise CommitError("INVALID_PROJECTION_CURSOR")
        with self.database.lifecycle.hold():
            conn = self.database._connect()
            try:
                conn.execute("BEGIN IMMEDIATE")
                maximum = conn.execute("SELECT coalesce(max(seq),0) FROM audit_events").fetchone()[
                    0
                ]
                if last_seq > maximum:
                    raise CommitError("PROJECTION_AHEAD_OF_CANONICAL_EVENTS")
                conn.execute(
                    "INSERT INTO projection_cursors VALUES(?,?,?,?,?) ON CONFLICT(projection_id) DO UPDATE SET last_seq=excluded.last_seq,generation=excluded.generation,manifest_sha256=excluded.manifest_sha256,updated_at_utc=excluded.updated_at_utc WHERE excluded.last_seq>=projection_cursors.last_seq",
                    (
                        projection_id,
                        last_seq,
                        generation,
                        manifest_sha256,
                        datetime.now(UTC).isoformat(),
                    ),
                )
                conn.execute("COMMIT")
            finally:
                if conn.in_transaction:
                    conn.execute("ROLLBACK")
                conn.close()
