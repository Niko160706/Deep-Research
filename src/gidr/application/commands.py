"""Single synchronous state coordinator; typed ports never execute external work."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from typing import Literal, Self
from uuid import uuid4

from gidr.application import accounting
from gidr.contracts import commands as dto
from gidr.contracts.common import ArtifactRef, Contract, Name, new_id, parse_contract
from gidr.contracts.control import AuditEvent, ResearchPlanProposal
from gidr.domain.transitions import allowed
from gidr.infrastructure.artifacts.repository import ArtifactRepository, ReferenceRequest
from gidr.infrastructure.artifacts.store import ArtifactError
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.db.authority import CommitError, PersistenceAuthority, WriteContext
from gidr.infrastructure.localfs import LocalLock


class Rejected(CommitError):
    """Stable nonsensitive reason, never untrusted provider text."""


class DeniedPrincipal(Contract):
    attempted_command_id: str
    command_type: Name
    reason: Name


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise Rejected(reason)


def stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def time_value(value: str) -> datetime:
    return datetime.fromisoformat(value)


ACTIVE = {"RUNNING", "PAUSING", "RECOVERING", "COMPLETING"}
PERMISSIONS: dict[str, frozenset[str]] = {
    "CREATE_RUN": frozenset({"USER"}),
    "PUBLISH_PLAN": frozenset({"USER"}),
    "START_RUN": frozenset({"USER"}),
    "RESUME_RUN": frozenset({"USER"}),
    "CHANGE_RUN": frozenset({"USER", "SYSTEM"}),
    "RENEW_RUN": frozenset({"SYSTEM"}),
    "SWITCH_SESSION": frozenset({"USER"}),
    "OPEN_SESSION": frozenset({"USER"}),
    "REVOKE_GRANT": frozenset({"USER"}),
    "APPROVE_GRANT": frozenset({"USER"}),
    "RELEASE_RESERVATION": frozenset({"SCHEDULER", "SYSTEM"}),
    "ADD_TASKS": frozenset({"USER", "SYSTEM"}),
    "CHANGE_TASK": frozenset({"USER", "SYSTEM", "SCHEDULER"}),
    "RESERVE_TASK": frozenset({"SCHEDULER"}),
    "START_ATTEMPT": frozenset({"WORKER"}),
    "RENEW_ATTEMPT": frozenset({"WORKER"}),
    "FINISH_ATTEMPT": frozenset({"WORKER"}),
    "RECONCILE_USAGE": frozenset({"USER", "SYSTEM"}),
    "RECOVER_RUN": frozenset({"SYSTEM"}),
    "CREATE_WORKFLOW": frozenset({"USER", "SYSTEM"}),
    "CHANGE_WORKFLOW": frozenset({"USER", "SYSTEM"}),
}


class CommandPort:
    """Internal role binding; this is not authentication of downloaded content."""

    def __init__(self, authority: StateAuthority, actor: dto.Actor, worker_id: str | None = None):
        self._authority = authority
        self.actor = actor
        self.worker_id = worker_id

    def handle(self, command: dto.StateCommand) -> dto.CommandResult:
        return self._authority.handle(command, actor=self.actor, worker_id=self.worker_id)


class StateAuthority:
    def __init__(
        self,
        storage: PersistenceAuthority,
        artifacts: ArtifactRepository,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        max_inflight: int = 3,
    ):
        if artifacts.authority is not storage:
            raise ValueError("SINGLE_PERSISTENCE_AUTHORITY_REQUIRED")
        self.storage = storage
        self.artifacts = artifacts
        if not 1 <= max_inflight <= 32:
            raise ValueError("INVALID_INFLIGHT_LIMIT")
        self.max_inflight = max_inflight
        self.clock = clock
        self.nonce = "instance-" + uuid4().hex
        self._ownership = LocalLock(
            storage.database.path.with_suffix(".instance.lock"), timeout=0.1
        )
        self._instance: AbstractContextManager[None] | None = None
        self._startup = False
        self._owned = False
        self._closing = False
        self._calls = 0
        self._condition = threading.Condition()
        self._startup_thread = 0

    def __enter__(self) -> Self:
        require(self._instance is None, "INSTANCE_ALREADY_OPEN")
        self._instance = self._ownership.hold()
        try:
            self._instance.__enter__()
        except BaseException:
            self._instance = None
            raise
        try:
            self._owned = True
            self._closing = False
            self._startup_thread = threading.get_ident()
            self._startup = True
            self._recover_startup()
            self._startup = False
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args: object) -> None:
        with self._condition:
            self._closing = True
            self._owned = False
            self._condition.wait_for(lambda: self._calls == 0)
        context, self._instance = self._instance, None
        if context is not None:
            context.__exit__(None, None, None)

    @contextmanager
    def model_operation(self) -> Iterator[None]:
        """Hold the single authority lifetime while a trusted model commit runs."""
        with self._condition:
            require(self._owned and not self._closing, "INSTANCE_NOT_OWNED")
            require(not self._startup, "STARTUP_RECOVERY_IN_PROGRESS")
            self._calls += 1
        try:
            yield
        finally:
            with self._condition:
                self._calls -= 1
                self._condition.notify_all()

    def port(self, actor: dto.Actor, worker_id: str | None = None) -> CommandPort:
        return CommandPort(self, actor, worker_id)

    def handle(
        self,
        command: dto.StateCommand,
        *,
        actor: dto.Actor,
        worker_id: str | None = None,
        fault: Callable[[str], None] = lambda _: None,
    ) -> dto.CommandResult:
        with self._condition:
            require(self._owned and not self._closing, "INSTANCE_NOT_OWNED")
            require(
                not self._startup or threading.get_ident() == self._startup_thread,
                "STARTUP_RECOVERY_IN_PROGRESS",
            )
            self._calls += 1
        try:
            return self._handle(command, actor=actor, worker_id=worker_id, fault=fault)
        finally:
            with self._condition:
                self._calls -= 1
                self._condition.notify_all()

    def _handle(
        self,
        command: dto.StateCommand,
        *,
        actor: dto.Actor,
        worker_id: str | None,
        fault: Callable[[str], None],
    ) -> dto.CommandResult:
        command = dto.StateCommand.model_validate(command)
        require(self._instance is not None, "INSTANCE_NOT_OWNED")
        # Do not allow a replay issued through a different role or worker port.
        if command.actor != actor or actor not in PERMISSIONS[command.command_type]:
            self._deny_principal(command, actor, "ACTOR_NOT_AUTHORIZED")
        if isinstance(command.payload, (dto.StartAttempt, dto.RenewAttempt, dto.FinishAttempt)):
            if worker_id != command.payload.token.holder:
                self._deny_principal(command, actor, "WORKER_ID_MISMATCH")
        prior = self.storage.replay(command.command_id, command, dto.CommandResult)
        if prior is not None:
            return prior
        try:
            with self.storage.database.reader() as reader:
                logical_prior = reader.execute(
                    "SELECT 1 FROM state_command_keys WHERE run_id=? AND idempotency_key=?",
                    (command.run_id, command.idempotency_key),
                ).fetchone()
            references = () if logical_prior is not None else self._references(command)
            for reference in references:
                self._read_reference(reference)
            with self.artifacts.prepared(references) if references else nullcontext():
                content = {ref.artifact_id: self.artifacts.store.read(ref) for ref in references}
                return self.storage.execute(
                    command.command_id,
                    command.trace_id,
                    command,
                    dto.CommandResult,
                    lambda ctx: self._operate(ctx, command, content),
                    fault=fault,
                )
        except Rejected as error:
            reason = str(error)
        except CommitError:
            reason = "PERSISTENCE_CONSTRAINT_FAILURE"
        except (ArtifactError, ValueError):
            reason = "ARTIFACT_OR_SCHEMA_INVALID"

        # A separate rejection transaction follows full rollback; no half-applied state.
        def rejection(context: WriteContext) -> dto.CommandResult:
            exists = context.execute(
                "SELECT 1 FROM runs WHERE run_id=?", (command.run_id,)
            ).fetchone()
            event = event_for(
                context,
                component="scheduler",
                event_type="command.rejected",
                aggregate_id=command.aggregate_id,
                run_id=command.run_id if exists else None,
            )
            context.emit(
                AuditEvent.model_validate(
                    {
                        **event.model_dump(),
                        "actor": command.actor,
                        "cause": command.command_type,
                        "reason_code": reason,
                    }
                )
            )
            return dto.CommandResult(accepted=False, code=reason)

        return self.storage.execute(
            command.command_id, command.trace_id, command, dto.CommandResult, rejection
        )

    def _deny_principal(self, command: dto.StateCommand, actor: dto.Actor, reason: str) -> None:
        request = DeniedPrincipal(
            attempted_command_id=command.command_id,
            command_type=command.command_type,
            reason=reason,
        )

        def change(ctx: WriteContext) -> dto.CommandResult:
            exists = ctx.execute("SELECT 1 FROM runs WHERE run_id=?", (command.run_id,)).fetchone()
            event = event_for(
                ctx,
                component="security",
                event_type="command.principal_denied",
                aggregate_id=command.command_id,
                run_id=command.run_id if exists else None,
            )
            ctx.emit(
                AuditEvent.model_validate(
                    {
                        **event.model_dump(),
                        "actor": actor,
                        "cause": command.command_type,
                        "reason_code": reason,
                    }
                )
            )
            return dto.CommandResult(accepted=False, code=reason)

        self.storage.execute(new_id("CMD"), command.trace_id, request, dto.CommandResult, change)
        raise Rejected(reason)

    def _references(self, command: dto.StateCommand) -> tuple[ArtifactRef, ...]:
        payload = command.payload
        refs: list[ArtifactRef] = []
        if isinstance(payload, dto.PublishPlan):
            refs.append(payload.snapshot_ref)
            plan = parse_contract(ResearchPlanProposal, self._read_reference(payload.snapshot_ref))
            refs.append(plan.budget_policy_ref)
        if isinstance(payload, dto.ChangeRun) and payload.checkpoint_ref:
            refs.append(payload.checkpoint_ref)
        if isinstance(payload, dto.ChangeTask) and payload.condition_change_ref:
            refs.append(payload.condition_change_ref)
        if isinstance(payload, dto.FinishAttempt) and payload.result_ref:
            refs.append(payload.result_ref)
        if isinstance(payload, dto.ReconcileUsage):
            refs.append(payload.observation_ref)
        if isinstance(payload, dto.CreateWorkflow) and payload.snapshot_ref:
            refs.append(payload.snapshot_ref)
        if isinstance(payload, dto.ChangeWorkflow) and payload.condition_change_ref:
            refs.append(payload.condition_change_ref)
        if isinstance(payload, dto.ChangeWorkflow) and payload.kind == "PROPOSAL":
            with self.storage.database.reader() as conn:
                row = conn.execute(
                    "SELECT a.* FROM critic_proposals p JOIN artifact_catalog a ON a.artifact_id=p.snapshot_artifact_id WHERE p.proposal_id=? AND p.run_id=?",
                    (payload.entity_id, command.run_id),
                ).fetchone()
            if row is not None:
                refs.append(
                    ArtifactRef(
                        artifact_id=row["artifact_id"],
                        sha256=row["sha256"],
                        size_bytes=row["size_bytes"],
                    )
                )
        if isinstance(payload, dto.AddTasks):
            for task in payload.tasks:
                refs.extend(task.input_refs)
        return tuple(dict.fromkeys(refs))

    def _read_reference(self, reference: ArtifactRef) -> bytes:
        # Known corruption must invalidate the catalog as well as quarantine bytes.
        with self.storage.database.reader() as reader:
            row = reader.execute(
                "SELECT status FROM artifact_catalog WHERE artifact_id=?", (reference.artifact_id,)
            ).fetchone()
        if row is not None and row[0] == "AVAILABLE":
            return self.artifacts.read(reference)
        return self.artifacts.store.read(reference)

    def _run(
        self, ctx: WriteContext, cmd: dto.StateCommand, *, revision: bool = True
    ) -> sqlite3.Row:
        row = ctx.execute(
            "SELECT r.*,coalesce(p.version,0) AS plan_version FROM runs r LEFT JOIN plan_versions p ON p.plan_version_id=r.active_plan_version_id WHERE r.run_id=?",
            (cmd.run_id,),
        ).fetchone()
        require(row is not None, "RUN_NOT_FOUND")
        assert isinstance(row, sqlite3.Row)
        if revision:
            require(row["revision"] == cmd.expected_revision, "CONFLICT_STALE_REVISION")
        require(cmd.plan_version == (row["plan_version"] or 1), "CONFLICT_STALE_PLAN_VERSION")
        return row

    def _emit(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        entity: str,
        before: str | None,
        after: str | None,
        revision: int,
        *,
        reason: str = "VALIDATED_COMMAND",
        task_id: str | None = None,
        attempt_id: str | None = None,
    ) -> None:
        revision_after = revision
        aggregate_type = "RUN"
        for table, key in (
            ("runs", "run_id"),
            ("tasks", "task_id"),
            ("grants", "grant_id"),
            ("task_attempts", "attempt_id"),
            ("leads", "lead_id"),
            ("critic_proposals", "proposal_id"),
            ("plan_change_proposals", "proposal_id"),
            ("assessments", "assessment_id"),
            ("sessions", "session_id"),
        ):
            current = ctx.execute(
                f"SELECT revision FROM {table} WHERE {key}=?", (entity,)
            ).fetchone()
            if current is not None:
                revision_after = int(current[0])
                aggregate_type = {
                    "runs": "RUN",
                    "tasks": "TASK",
                    "grants": "GRANT",
                    "task_attempts": "ATTEMPT",
                    "leads": "LEAD",
                    "critic_proposals": "PROPOSAL",
                    "plan_change_proposals": "PLAN_CHANGE",
                    "assessments": "ASSESSMENT",
                    "sessions": "SESSION",
                }[table]
                break
        current_plan = ctx.execute(
            "SELECT p.version FROM runs r LEFT JOIN plan_versions p ON p.plan_version_id=r.active_plan_version_id WHERE r.run_id=?",
            (cmd.run_id,),
        ).fetchone()
        if task_id is not None:
            current_plan = ctx.execute(
                "SELECT p.version FROM tasks t JOIN plan_versions p ON p.plan_version_id=t.plan_version_id WHERE t.task_id=?",
                (task_id,),
            ).fetchone()
        grant_id = cmd.grant_id
        if attempt_id is not None:
            attempt = ctx.execute(
                "SELECT grant_id FROM task_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
            if attempt is not None:
                grant_id = attempt[0]
        actual_grant = ctx.execute(
            "SELECT grant_id FROM grants WHERE grant_id=? AND run_id=?", (grant_id, cmd.run_id)
        ).fetchone()
        event = event_for(
            ctx,
            component="scheduler",
            event_type=cmd.command_type.lower() + ".committed",
            aggregate_id=entity,
            run_id=cmd.run_id,
        )
        ctx.emit(
            AuditEvent.model_validate(
                {
                    **event.model_dump(),
                    "actor": cmd.actor,
                    "aggregate_type": aggregate_type,
                    "session_id": run["session_id"],
                    "task_id": task_id,
                    "attempt_id": attempt_id,
                    "plan_version": None if current_plan is None else current_plan[0],
                    "grant_id": None if actual_grant is None else actual_grant[0],
                    "revision_before": None
                    if (before is None and after is not None)
                    or isinstance(cmd.payload, (dto.CreateRun, dto.CreateWorkflow))
                    else revision,
                    "revision_after": revision_after,
                    "old_state": before,
                    "new_state": after,
                    "cause": cmd.command_type,
                    "reason_code": reason,
                }
            )
        )

    def _move_run(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row, target: str, reason: str
    ) -> dto.CommandResult:
        require(allowed("RUN", run["state"], target), "ILLEGAL_RUN_TRANSITION")
        ctx.execute(
            "UPDATE runs SET state=?,revision=revision+1,updated_at_utc=?,stop_reason=? WHERE run_id=? AND revision=?",
            (target, stamp(self.clock()), reason, cmd.run_id, run["revision"]),
        )
        self._emit(
            ctx,
            cmd,
            run,
            cmd.run_id,
            str(run["state"]),
            target,
            int(run["revision"]),
            reason=reason,
        )
        return dto.CommandResult(
            accepted=True, code="COMMITTED", state=target, revision=int(run["revision"]) + 1
        )

    def _task(self, ctx: WriteContext, cmd: dto.StateCommand, task_id: str) -> sqlite3.Row:
        require(cmd.aggregate_id == task_id, "AGGREGATE_MISMATCH")
        row = ctx.execute(
            "SELECT * FROM tasks WHERE task_id=? AND run_id=?", (task_id, cmd.run_id)
        ).fetchone()
        require(row is not None, "TASK_NOT_FOUND")
        assert isinstance(row, sqlite3.Row)
        require(row["revision"] == cmd.expected_revision, "CONFLICT_STALE_REVISION")
        run = self._run(ctx, cmd, revision=False)
        require(
            row["plan_version_id"] == run["active_plan_version_id"], "CONFLICT_STALE_PLAN_VERSION"
        )
        return row

    def _move_task(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        task: sqlite3.Row,
        target: str,
        reason: str,
        attempt_id: str | None = None,
    ) -> dto.CommandResult:
        require(allowed("TASK", task["state"], target), "ILLEGAL_TASK_TRANSITION")
        ctx.execute(
            "UPDATE tasks SET state=?,revision=revision+1,updated_at_utc=? WHERE task_id=? AND revision=?",
            (target, stamp(self.clock()), task["task_id"], task["revision"]),
        )
        self._emit(
            ctx,
            cmd,
            run,
            str(task["task_id"]),
            str(task["state"]),
            target,
            int(task["revision"]),
            reason=reason,
            task_id=str(task["task_id"]),
            attempt_id=attempt_id,
        )
        return dto.CommandResult(
            accepted=True, code="COMMITTED", state=target, revision=int(task["revision"]) + 1
        )

    def _gate(self, ctx: WriteContext, cmd: dto.StateCommand, *, fresh: bool = True) -> sqlite3.Row:
        gate = ctx.execute("SELECT * FROM active_run_gate WHERE singleton=1").fetchone()
        assert isinstance(gate, sqlite3.Row)
        require(gate["run_id"] == cmd.run_id and gate["holder"] == self.nonce, "RUN_GATE_NOT_OWNED")
        if fresh:
            require(time_value(gate["expires_at_utc"]) > self.clock(), "RUN_LEASE_EXPIRED")
        return gate

    def _acquire(self, ctx: WriteContext, cmd: dto.StateCommand, ttl: int) -> None:
        gate = ctx.execute("SELECT * FROM active_run_gate WHERE singleton=1").fetchone()
        assert isinstance(gate, sqlite3.Row)
        require(gate["run_id"] is None, "GLOBAL_RUN_BUSY")
        ctx.execute(
            "UPDATE active_run_gate SET run_id=?,generation=generation+1,holder=?,expires_at_utc=?,updated_at_utc=? WHERE singleton=1",
            (
                cmd.run_id,
                self.nonce,
                stamp(self.clock() + timedelta(seconds=ttl)),
                stamp(self.clock()),
            ),
        )

    def _grant(self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row) -> dto.GrantPolicy:
        row = ctx.execute("SELECT * FROM grants WHERE grant_id=?", (cmd.grant_id,)).fetchone()
        require(row is not None, "GRANT_REQUIRED")
        assert isinstance(row, sqlite3.Row)
        require(
            row["run_id"] == cmd.run_id and row["plan_version_id"] == run["active_plan_version_id"],
            "GRANT_SCOPE_MISMATCH",
        )
        require(not row["revoked"], "GRANT_REVOKED")
        policy = parse_contract(dto.GrantPolicy, row["scope_json"])
        require(not dto.expiry(self.clock(), policy.expires_at_utc), "GRANT_EXPIRED")
        return policy

    def _save_grant(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        policy: dto.GrantPolicy,
        version_id: str,
    ) -> None:
        require(cmd.actor == "USER" and cmd.grant_id == policy.grant_id, "USER_APPROVAL_REQUIRED")
        require(not dto.expiry(self.clock(), policy.expires_at_utc), "GRANT_EXPIRED")
        require(accounting.fits(ctx, cmd.run_id, policy), "GRANT_BELOW_ACCOUNTED_USAGE")
        control = ctx.execute(
            "SELECT categories_json FROM plan_control WHERE plan_version_id=?", (version_id,)
        ).fetchone()
        if control is not None:
            require(
                set(policy.categories) <= set(json.loads(control[0])), "GRANT_CATEGORY_OUTSIDE_PLAN"
            )
        ctx.execute(
            "INSERT INTO grants(grant_id,run_id,plan_version_id,scope_json,budget_json,approval_source,expires_at_utc,created_at_utc) VALUES(?,?,?,?,?,'USER_APPROVED',?,?)",
            (
                policy.grant_id,
                cmd.run_id,
                version_id,
                policy.model_dump_json(),
                policy.model_dump_json(),
                None if policy.expires_at_utc is None else stamp(policy.expires_at_utc),
                stamp(self.clock()),
            ),
        )

    def _attach(
        self,
        ctx: WriteContext,
        ref: ArtifactRef,
        owner: Literal["RUN", "PLAN_VERSION"],
        identity: str,
        role: str,
    ) -> None:
        self.artifacts.attach(
            ctx, ReferenceRequest(reference=ref, owner_type=owner, owner_id=identity, role=role)
        )

    @staticmethod
    def _logical_hash(cmd: dto.StateCommand) -> str:
        body = cmd.model_dump(mode="json", exclude={"command_id", "trace_id", "created_at_utc"})
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _operate(
        self, ctx: WriteContext, cmd: dto.StateCommand, content: dict[str, bytes]
    ) -> dto.CommandResult:
        prior = ctx.execute(
            "SELECT request_sha256,result_json FROM state_command_keys WHERE run_id=? AND idempotency_key=?",
            (cmd.run_id, cmd.idempotency_key),
        ).fetchone()
        digest = self._logical_hash(cmd)
        if prior is not None:
            require(prior["request_sha256"] == digest, "IDEMPOTENCY_KEY_CONFLICT")
            result = parse_contract(dto.CommandResult, prior["result_json"])
            event = event_for(
                ctx,
                component="scheduler",
                event_type="command.logical_replay",
                aggregate_id=cmd.aggregate_id,
                run_id=cmd.run_id,
            )
            ctx.emit(
                AuditEvent.model_validate(
                    {
                        **event.model_dump(),
                        "actor": cmd.actor,
                        "cause": cmd.command_type,
                        "reason_code": "LOGICAL_KEY_REPLAY",
                    }
                )
            )
            return result
        result = self._dispatch(ctx, cmd, content)
        ctx.execute(
            "INSERT INTO state_command_keys VALUES(?,?,?,?,?)",
            (cmd.run_id, cmd.idempotency_key, digest, result.model_dump_json(), cmd.command_id),
        )
        return result

    def _dispatch(
        self, ctx: WriteContext, cmd: dto.StateCommand, content: dict[str, bytes]
    ) -> dto.CommandResult:
        p = cmd.payload
        if isinstance(p, dto.ApproveGrant):
            run = self._run(ctx, cmd)
            require(
                run["state"] in {"RUNNING", "PAUSED", "PLAN_READY"}, "GRANT_APPROVAL_STATE_INVALID"
            )
            self._save_grant(ctx, cmd, run, p.grant, str(run["active_plan_version_id"]))
            ctx.execute("UPDATE runs SET revision=revision+1 WHERE run_id=?", (cmd.run_id,))
            self._emit(
                ctx,
                cmd,
                run,
                cmd.run_id,
                str(run["state"]),
                str(run["state"]),
                int(run["revision"]),
                reason="USER_GRANT_APPROVED",
            )
            return dto.CommandResult(
                accepted=True, code="GRANT_APPROVED", revision=int(run["revision"]) + 1
            )
        if isinstance(p, dto.ReleaseReservation):
            run = self._run(ctx, cmd, revision=False)
            task = self._task(ctx, cmd, p.token.task_id)
            gate = self._gate(ctx, cmd, fresh=False)
            row = ctx.execute(
                "SELECT * FROM task_attempts WHERE attempt_id=?", (p.token.attempt_id,)
            ).fetchone()
            require(
                row is not None
                and row["task_id"] == p.token.task_id
                and row["state"] == "RESERVED"
                and task["state"] == "RESERVED",
                "UNISSUED_RESERVATION_REQUIRED",
            )
            assert isinstance(row, sqlite3.Row)
            require(
                row["lease_generation"] == p.token.generation
                and row["holder"] == p.token.holder
                and row["gate_generation"] == p.token.gate_generation == gate["generation"],
                "STALE_FENCING_TOKEN",
            )
            ctx.execute(
                "UPDATE task_attempts SET state='ROLLED_BACK',usage_state='KNOWN',revision=revision+1 WHERE attempt_id=?",
                (p.token.attempt_id,),
            )
            ctx.execute(
                "UPDATE attempt_accounts SET status='RELEASED',cost_usd='0',tokens=0,requests=0 WHERE attempt_id=?",
                (p.token.attempt_id,),
            )
            from decimal import Decimal

            account = ctx.execute(
                "SELECT max_cost_usd FROM attempt_accounts WHERE attempt_id=?",
                (p.token.attempt_id,),
            ).fetchone()
            assert account is not None
            accounting.ledger(
                ctx,
                cmd.run_id,
                p.token.task_id,
                p.token.attempt_id,
                "RELEASE",
                Decimal(account[0]),
                "KNOWN",
                stamp(self.clock()),
            )
            return self._move_task(ctx, cmd, run, task, "READY", p.reason, p.token.attempt_id)
        if isinstance(p, dto.CreateRun):
            return self._create(ctx, cmd, p)
        if isinstance(p, dto.OpenSession):
            return self._open_session(ctx, cmd, p)
        run = self._run(
            ctx,
            cmd,
            revision=isinstance(
                p,
                (
                    dto.ChangeRun,
                    dto.PublishPlan,
                    dto.StartRun,
                    dto.ResumeRun,
                    dto.RenewRun,
                    dto.SwitchSession,
                    dto.AddTasks,
                    dto.RecoverRun,
                ),
            ),
        )
        if isinstance(p, dto.PublishPlan):
            return self._publish_plan(ctx, cmd, run, p, content)
        if isinstance(p, dto.StartRun):
            require(run["state"] == "PLAN_READY", "ILLEGAL_RUN_TRANSITION")
            self._save_grant(ctx, cmd, run, p.grant, str(run["active_plan_version_id"]))
            self._acquire(ctx, cmd, p.lease_ttl_seconds)
            return self._move_run(ctx, cmd, run, "RUNNING", "USER_START_APPROVED")
        if isinstance(p, dto.ResumeRun):
            require(run["state"] == "PAUSED", "EXPLICIT_PAUSED_RESUME_REQUIRED")
            self._grant(ctx, cmd, run)
            require(not self._inflight(ctx, cmd.run_id), "RECONCILIATION_REQUIRED")
            self._acquire(ctx, cmd, p.lease_ttl_seconds)
            self._move_run(ctx, cmd, run, "RECOVERING", "EXPLICIT_USER_RESUME")
            return self._move_run(
                ctx, cmd, self._run(ctx, cmd, revision=False), "RUNNING", "RECONCILED_USER_RESUME"
            )
        if isinstance(p, dto.RenewRun):
            gate = self._gate(ctx, cmd)
            require(gate["generation"] == p.generation, "STALE_FENCING_TOKEN")
            ctx.execute(
                "UPDATE active_run_gate SET expires_at_utc=?,updated_at_utc=? WHERE singleton=1",
                (stamp(self.clock() + timedelta(seconds=p.lease_ttl_seconds)), stamp(self.clock())),
            )
            ctx.execute("UPDATE runs SET revision=revision+1 WHERE run_id=?", (cmd.run_id,))
            self._emit(
                ctx,
                cmd,
                run,
                cmd.run_id,
                str(run["state"]),
                str(run["state"]),
                int(run["revision"]),
            )
            return dto.CommandResult(
                accepted=True,
                code="COMMITTED",
                revision=int(run["revision"]) + 1,
                state=run["state"],
            )
        if isinstance(p, dto.SwitchSession):
            require(
                ctx.execute("SELECT 1 FROM sessions WHERE session_id=?", (p.session_id,)).fetchone()
                is not None,
                "SESSION_NOT_FOUND",
            )
            require(run["state"] == "RUNNING", "RUN_NOT_RUNNING")
            self._gate(ctx, cmd)
            ctx.execute(
                "UPDATE navigation_state SET pending_session_id=? WHERE singleton=1",
                (p.session_id,),
            )
            return self._move_run(ctx, cmd, run, "PAUSING", "SESSION_SWITCH")
        if isinstance(p, dto.ChangeRun):
            return self._change_run(ctx, cmd, run, p, content)
        if isinstance(p, dto.RevokeGrant):
            row = ctx.execute(
                "SELECT * FROM grants WHERE grant_id=? AND run_id=?", (cmd.grant_id, cmd.run_id)
            ).fetchone()
            require(row is not None, "GRANT_REQUIRED")
            assert isinstance(row, sqlite3.Row)
            require(
                cmd.aggregate_id == cmd.grant_id and row["revision"] == cmd.expected_revision,
                "CONFLICT_STALE_REVISION",
            )
            ctx.execute(
                "UPDATE grants SET revoked=1,revision=revision+1 WHERE grant_id=?", (cmd.grant_id,)
            )
            self._emit(
                ctx,
                cmd,
                run,
                str(cmd.grant_id),
                "VALID",
                "REVOKED",
                int(row["revision"]),
                reason=p.reason,
            )
            return dto.CommandResult(
                accepted=True, code="COMMITTED", revision=int(row["revision"]) + 1
            )
        if isinstance(p, dto.AddTasks):
            return self._add_tasks(ctx, cmd, run, p)
        if isinstance(p, dto.ChangeTask):
            return self._change_task(ctx, cmd, run, p)
        if isinstance(p, dto.ReserveTask):
            return self._reserve(ctx, cmd, run, p)
        if isinstance(p, (dto.StartAttempt, dto.RenewAttempt, dto.FinishAttempt)):
            return self._attempt(ctx, cmd, run, p)
        if isinstance(p, dto.ReconcileUsage):
            return self._usage(ctx, cmd, run, p)
        if isinstance(p, dto.RecoverRun):
            return self._recover(ctx, cmd, run)
        if isinstance(p, (dto.CreateWorkflow, dto.ChangeWorkflow)):
            return self._workflow(ctx, cmd, run, p, content)
        raise Rejected("COMMAND_UNSUPPORTED")

    def _create(
        self, ctx: WriteContext, cmd: dto.StateCommand, p: dto.CreateRun
    ) -> dto.CommandResult:
        require(cmd.expected_revision == 0 and cmd.aggregate_id == cmd.run_id, "AGGREGATE_MISMATCH")
        require(
            ctx.execute("SELECT 1 FROM runs WHERE run_id=?", (cmd.run_id,)).fetchone() is None,
            "RUN_ALREADY_EXISTS",
        )
        ctx.execute(
            "INSERT OR IGNORE INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            (p.project_id, p.title, p.owner, stamp(self.clock())),
        )
        existing = ctx.execute(
            "SELECT project_id FROM sessions WHERE session_id=?", (p.session_id,)
        ).fetchone()
        require(existing is None or existing[0] == p.project_id, "SESSION_PROJECT_MISMATCH")
        ctx.execute(
            "INSERT OR IGNORE INTO sessions(session_id,project_id,created_at_utc) VALUES(?,?,?)",
            (p.session_id, p.project_id, stamp(self.clock())),
        )
        ctx.execute(
            "INSERT INTO runs(run_id,project_id,session_id,state,created_at_utc,updated_at_utc) VALUES(?,?,?,'CREATED',?,?)",
            (cmd.run_id, p.project_id, p.session_id, stamp(self.clock()), stamp(self.clock())),
        )
        run = self._run(ctx, cmd)
        self._emit(ctx, cmd, run, cmd.run_id, None, "CREATED", 0)
        return dto.CommandResult(accepted=True, code="COMMITTED", state="CREATED", revision=0)

    def _open_session(
        self, ctx: WriteContext, cmd: dto.StateCommand, p: dto.OpenSession
    ) -> dto.CommandResult:
        run = self._run(ctx, cmd, revision=False)
        require(run["session_id"] == p.session_id, "SESSION_RUN_MISMATCH")
        require(
            ctx.execute("SELECT 1 FROM sessions WHERE session_id=?", (p.session_id,)).fetchone()
            is not None,
            "SESSION_NOT_FOUND",
        )
        gate = ctx.execute("SELECT run_id FROM active_run_gate WHERE singleton=1").fetchone()
        assert isinstance(gate, sqlite3.Row)
        if gate[0] is not None:
            active = ctx.execute(
                "SELECT session_id FROM runs WHERE run_id=?", (gate[0],)
            ).fetchone()
            require(
                active is not None and active[0] == p.session_id,
                "SESSION_SWITCH_REQUIRES_CHECKPOINT",
            )
        ctx.execute("UPDATE navigation_state SET session_id=? WHERE singleton=1", (p.session_id,))
        run = self._run(ctx, cmd, revision=False)
        self._emit(
            ctx,
            cmd,
            run,
            p.session_id,
            None,
            None,
            int(
                ctx.execute(
                    "SELECT revision FROM sessions WHERE session_id=?", (p.session_id,)
                ).fetchone()[0]
            ),
            reason="NAVIGATION_ONLY_NO_RESUME",
        )
        return dto.CommandResult(
            accepted=True,
            code="SESSION_OPENED_NO_RESUME",
            state=str(run["state"]),
            revision=int(run["revision"]),
        )

    def _publish_plan(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        p: dto.PublishPlan,
        content: dict[str, bytes],
    ) -> dto.CommandResult:
        plan = parse_contract(ResearchPlanProposal, content[p.snapshot_ref.artifact_id])
        require(len(plan.questions) <= 1000, "PLAN_QUESTION_LIMIT")
        require(
            plan.run_id == cmd.run_id
            and plan.project_id == run["project_id"]
            and plan.session_id == run["session_id"],
            "PLAN_LINK_MISMATCH",
        )
        require(plan.expected_plan_version == run["plan_version"], "CONFLICT_STALE_PLAN_VERSION")
        require(
            run["state"] in {"PLANNING", "PLAN_READY", "RUNNING", "PAUSED"},
            "PLAN_PUBLICATION_STATE_INVALID",
        )
        if run["plan_version"]:
            change = ctx.execute(
                "SELECT * FROM plan_change_proposals WHERE proposal_id=? AND run_id=?",
                (p.change_proposal_id, cmd.run_id),
            ).fetchone()
            require(
                change is not None
                and change["status"] == "REVIEWED"
                and change["expected_version"] == run["plan_version"]
                and change["snapshot_artifact_id"] == p.snapshot_ref.artifact_id,
                "REVIEWED_PLAN_CHANGE_REQUIRED",
            )
            require(p.grant is not None, "NEW_VERSION_GRANT_REQUIRED")
            ctx.execute(
                "UPDATE plan_change_proposals SET status='APPROVED',revision=revision+1 WHERE proposal_id=?",
                (p.change_proposal_id,),
            )
        else:
            require(
                run["state"] == "PLANNING" and p.change_proposal_id is None,
                "INITIAL_PLAN_REQUIRES_PLANNING",
            )
        version = int(run["plan_version"]) + 1
        ctx.execute(
            "INSERT OR IGNORE INTO plans VALUES(?,?,?)",
            (p.plan_id, cmd.run_id, stamp(self.clock())),
        )
        self.artifacts.register(ctx, p.snapshot_ref)
        # Catalog + owner FK are deferred within this same transaction, never a second writer.
        ctx.execute(
            "INSERT INTO plan_versions VALUES(?,?,?,?,?,?,?,?)",
            (
                p.plan_version_id,
                p.plan_id,
                cmd.run_id,
                version,
                run["active_plan_version_id"],
                p.snapshot_ref.artifact_id,
                stamp(self.clock()),
                "USER_PUBLICATION",
            ),
        )
        self._attach(ctx, p.snapshot_ref, "PLAN_VERSION", p.plan_version_id, "SNAPSHOT")
        self._attach(
            ctx, plan.budget_policy_ref, "PLAN_VERSION", p.plan_version_id, "BUDGET_POLICY"
        )
        for question in plan.questions:
            ctx.execute(
                "INSERT INTO questions VALUES(?,?,?,? ,'UNEXPLORED',?)",
                (
                    question.question_id,
                    p.plan_version_id,
                    None,
                    question.methodology.value,
                    question.model_dump_json(),
                ),
            )
        if p.grant:
            require(
                set(p.grant.categories) <= set(plan.source_categories),
                "GRANT_CATEGORY_OUTSIDE_PLAN",
            )
            self._save_grant(ctx, cmd, run, p.grant, p.plan_version_id)
        # Category allowlist is part of immutable plan policy; no data/provider inference.
        ctx.execute(
            "INSERT INTO plan_control VALUES(?,?)",
            (p.plan_version_id, json.dumps(list(plan.source_categories))),
        )
        ctx.execute(
            "UPDATE runs SET active_plan_version_id=?,revision=revision+1,updated_at_utc=? WHERE run_id=?",
            (p.plan_version_id, stamp(self.clock()), cmd.run_id),
        )
        self._emit(
            ctx,
            cmd,
            run,
            cmd.run_id,
            str(run["state"]),
            str(run["state"]),
            int(run["revision"]),
            reason="IMMUTABLE_PLAN_PUBLICATION",
        )
        return dto.CommandResult(
            accepted=True,
            code="PLAN_PUBLISHED",
            revision=int(run["revision"]) + 1,
            state=str(run["state"]),
        )

    @staticmethod
    def _inflight(ctx: WriteContext, run_id: str) -> int:
        return int(
            ctx.execute(
                "SELECT count(*) FROM task_attempts a JOIN tasks t USING(task_id) WHERE t.run_id=? AND a.state IN ('RESERVED','RUNNING')",
                (run_id,),
            ).fetchone()[0]
        )

    def checkpoint(self, run_id: str) -> ArtifactRef:
        require(self._instance is not None, "INSTANCE_NOT_OWNED")
        with self.storage.database.reader() as conn:
            conn.execute("BEGIN")
            run = conn.execute(
                "SELECT r.revision,coalesce(p.version,0) FROM runs r LEFT JOIN plan_versions p ON p.plan_version_id=r.active_plan_version_id WHERE r.run_id=?",
                (run_id,),
            ).fetchone()
            require(run is not None, "RUN_NOT_FOUND")
            assert run is not None
            sequence = int(
                conn.execute("SELECT coalesce(max(seq),0) FROM audit_events").fetchone()[0]
            )
            generation = int(
                conn.execute("SELECT generation FROM active_run_gate WHERE singleton=1").fetchone()[
                    0
                ]
            )
            value = dto.StateCheckpoint(
                run_id=run_id,
                revision=run[0],
                event_seq=sequence,
                gate_generation=generation,
                plan_version=run[1],
                created_at_utc=self.clock(),
            )
        return self.artifacts.store.publish(value.model_dump_json().encode())

    def _change_run(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        p: dto.ChangeRun,
        content: dict[str, bytes],
    ) -> dto.CommandResult:
        require(cmd.aggregate_id == cmd.run_id, "AGGREGATE_MISMATCH")
        require(allowed("RUN", run["state"], p.target), "ILLEGAL_RUN_TRANSITION")
        require(p.target not in {"RUNNING", "RECOVERING"}, "SPECIALIZED_COMMAND_REQUIRED")
        if p.target == "COMPLETED":
            raise Rejected("FINAL_REPORT_VALIDATION_REQUIRED_B13")
        if p.target == "PLAN_READY":
            require(run["active_plan_version_id"] is not None, "VALIDATED_PLAN_REQUIRED")
        if run["state"] in ACTIVE or run["state"] == "INTERRUPTED":
            self._gate(ctx, cmd, fresh=False)
        if p.target in {"COMPLETING", "PAUSED", "FAILED", "CANCELLED"}:
            require(not self._inflight(ctx, cmd.run_id), "INFLIGHT_RECONCILIATION_REQUIRED")
        if p.target == "COMPLETING":
            require(cmd.actor == "SYSTEM", "SYSTEM_REQUIRED")
            require(
                ctx.execute(
                    "SELECT 1 FROM tasks WHERE run_id=? AND state NOT IN ('SUCCEEDED','FAILED','CANCELLED') LIMIT 1",
                    (cmd.run_id,),
                ).fetchone()
                is None,
                "UNRESOLVED_TASKS_REQUIRE_PARTIAL_PAUSE",
            )
        if p.target == "PAUSED":
            require(p.checkpoint_ref is not None, "DURABLE_CHECKPOINT_REQUIRED")
            assert p.checkpoint_ref
            snap = parse_contract(dto.StateCheckpoint, content[p.checkpoint_ref.artifact_id])
            gate = self._gate(ctx, cmd, fresh=False)
            sequence = int(
                ctx.execute("SELECT coalesce(max(seq),0) FROM audit_events").fetchone()[0]
            )
            require(
                snap.run_id == cmd.run_id
                and snap.revision == run["revision"]
                and snap.event_seq == sequence
                and snap.gate_generation == gate["generation"]
                and snap.plan_version == run["plan_version"],
                "CHECKPOINT_STALE_OR_MISMATCH",
            )
            self._attach(ctx, p.checkpoint_ref, "RUN", cmd.run_id, "CHECKPOINT")
            ctx.execute(
                "INSERT INTO run_checkpoints VALUES(?,?,?,?,?)",
                (
                    new_id("ART"),
                    cmd.run_id,
                    int(run["revision"]) + 1,
                    p.checkpoint_ref.artifact_id,
                    stamp(self.clock()),
                ),
            )
            ctx.execute(
                "UPDATE navigation_state SET session_id=coalesce(pending_session_id,session_id),pending_session_id=NULL WHERE singleton=1"
            )
        result = self._move_run(ctx, cmd, run, p.target, p.reason)
        if p.target in {"PAUSED", "FAILED", "CANCELLED"}:
            ctx.execute(
                "UPDATE active_run_gate SET run_id=NULL,holder=NULL,expires_at_utc=NULL,generation=generation+1,updated_at_utc=? WHERE singleton=1 AND run_id=?",
                (stamp(self.clock()), cmd.run_id),
            )
        return result

    def _eligible(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row, task: sqlite3.Row
    ) -> tuple[dto.GrantPolicy, dto.ScheduledTask]:
        require(run["state"] == "RUNNING", "DISPATCH_STOPPED")
        self._gate(ctx, cmd)
        grant = self._grant(ctx, cmd, run)
        require(task["policy_json"] is not None, "TASK_POLICY_REQUIRED")
        spec = parse_contract(dto.ScheduledTask, task["policy_json"])
        require(
            spec.provider_id in grant.providers
            and spec.category in grant.categories
            and spec.task_type in grant.task_types,
            "TASK_OUTSIDE_GRANT",
        )
        categories = ctx.execute(
            "SELECT categories_json FROM plan_control WHERE plan_version_id=?",
            (run["active_plan_version_id"],),
        ).fetchone()
        require(
            categories is not None and spec.category in json.loads(categories[0]),
            "TASK_OUTSIDE_PLAN",
        )
        require(
            ctx.execute(
                "SELECT 1 FROM task_dependencies d JOIN tasks t ON t.task_id=d.dependency_task_id WHERE d.task_id=? AND t.state<>'SUCCEEDED' LIMIT 1",
                (task["task_id"],),
            ).fetchone()
            is None,
            "DEPENDENCIES_NOT_SATISFIED",
        )
        if task["not_before_utc"]:
            require(time_value(task["not_before_utc"]) <= self.clock(), "TASK_COOLDOWN_ACTIVE")
        if spec.work_kind == "PROVIDER":
            if spec.category != "model" or spec.provider_id not in {
                "jev",
                "gemini",
                "groq",
                "siliconflow",
            }:
                raise Rejected("EXTERNAL_SECURITY_GATE_UNVERIFIED_B04")
        return grant, spec

    def _add_tasks(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row, p: dto.AddTasks
    ) -> dto.CommandResult:
        require(run["state"] in {"PLAN_READY", "RUNNING", "PAUSED"}, "TASK_CREATION_STATE_INVALID")
        require(run["active_plan_version_id"] is not None, "VALIDATED_PLAN_REQUIRED")
        require(len({t.task_id for t in p.tasks}) == len(p.tasks), "DUPLICATE_TASK_ID")
        require(
            int(
                ctx.execute("SELECT count(*) FROM tasks WHERE run_id=?", (cmd.run_id,)).fetchone()[
                    0
                ]
            )
            + len(p.tasks)
            <= 10000,
            "TASK_LIMIT",
        )
        for spec in p.tasks:
            require(
                ctx.execute(
                    "SELECT 1 FROM questions WHERE question_id=? AND plan_version_id=?",
                    (spec.question_id, run["active_plan_version_id"]),
                ).fetchone()
                is not None,
                "TASK_QUESTION_OUTSIDE_PLAN",
            )
            fingerprint = hashlib.sha256(
                spec.model_dump_json(exclude={"task_id", "reason_code"}).encode()
            ).hexdigest()
            require(
                ctx.execute(
                    "SELECT 1 FROM tasks WHERE run_id=? AND plan_version_id=? AND fingerprint=?",
                    (cmd.run_id, run["active_plan_version_id"], fingerprint),
                ).fetchone()
                is None,
                "DUPLICATE_TASK_FINGERPRINT",
            )
            ctx.execute(
                "INSERT INTO tasks(task_id,run_id,plan_version_id,type,state,fingerprint,created_at_utc,updated_at_utc,policy_json) VALUES(?,?,?,?,'PENDING',?,?,?,?)",
                (
                    spec.task_id,
                    cmd.run_id,
                    run["active_plan_version_id"],
                    spec.task_type,
                    fingerprint,
                    stamp(self.clock()),
                    stamp(self.clock()),
                    spec.model_dump_json(),
                ),
            )
            for ref in spec.input_refs:
                self._attach(ctx, ref, "RUN", cmd.run_id, "TASK_INPUT_" + spec.task_id)
            self._emit(
                ctx,
                cmd,
                run,
                spec.task_id,
                None,
                "PENDING",
                0,
                reason=spec.reason_code,
                task_id=spec.task_id,
            )
        for spec in p.tasks:
            for dependency in spec.dependencies:
                row = ctx.execute(
                    "SELECT plan_version_id FROM tasks WHERE task_id=? AND run_id=?",
                    (dependency, cmd.run_id),
                ).fetchone()
                require(
                    row is not None and row[0] == run["active_plan_version_id"],
                    "DEPENDENCY_OUTSIDE_PLAN",
                )
                ctx.execute("INSERT INTO task_dependencies VALUES(?,?)", (spec.task_id, dependency))
        cycle = ctx.execute(
            "WITH RECURSIVE reach(a,b) AS (SELECT task_id,dependency_task_id FROM task_dependencies UNION SELECT r.a,d.dependency_task_id FROM reach r JOIN task_dependencies d ON r.b=d.task_id) SELECT 1 FROM reach WHERE a=b LIMIT 1"
        ).fetchone()
        require(cycle is None, "DEPENDENCY_CYCLE")
        ctx.execute("UPDATE runs SET revision=revision+1 WHERE run_id=?", (cmd.run_id,))
        self._emit(
            ctx,
            cmd,
            run,
            cmd.run_id,
            None,
            None,
            int(run["revision"]),
            reason="TASK_PROPOSALS_COMMITTED",
        )
        return dto.CommandResult(accepted=True, code="COMMITTED", revision=int(run["revision"]) + 1)

    def _change_task(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row, p: dto.ChangeTask
    ) -> dto.CommandResult:
        task = self._task(ctx, cmd, p.task_id)
        require(
            p.target not in {"RESERVED", "RUNNING", "SUCCEEDED"}, "SPECIALIZED_COMMAND_REQUIRED"
        )
        require(task["state"] not in {"RESERVED", "RUNNING"}, "ATTEMPT_RECONCILIATION_REQUIRED")
        if p.target == "READY":
            self._eligible(ctx, cmd, run, task)
            if task["state"] in {"FAILED", "AWAITING_APPROVAL"}:
                require(cmd.actor == "USER", "EXPLICIT_REPAIR_APPROVAL_REQUIRED")
            if task["state"] == "BLOCKED":
                require(p.condition_change_ref is not None, "VALIDATED_CONDITION_CHANGE_REQUIRED")
            uncertain = ctx.execute(
                "SELECT 1 FROM attempt_accounts a JOIN task_attempts t USING(attempt_id) WHERE t.task_id=? AND a.status='UNCERTAIN'",
                (p.task_id,),
            ).fetchone()
            if uncertain:
                require(cmd.actor == "USER", "UNCERTAIN_RETRY_REQUIRES_USER_APPROVAL")
        if p.condition_change_ref:
            self._attach(ctx, p.condition_change_ref, "RUN", cmd.run_id, "CONDITION_CHANGE")
        result = self._move_task(ctx, cmd, run, task, p.target, p.reason)
        ctx.execute(
            "UPDATE tasks SET not_before_utc=? WHERE task_id=?",
            (None if p.not_before_utc is None else stamp(p.not_before_utc), p.task_id),
        )
        return result

    def _reserve(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row, p: dto.ReserveTask
    ) -> dto.CommandResult:
        task = self._task(ctx, cmd, p.task_id)
        require(task["state"] == "READY", "TASK_NOT_READY")
        require(self._inflight(ctx, cmd.run_id) < self.max_inflight, "CONCURRENCY_LIMIT")
        grant, spec = self._eligible(ctx, cmd, run, task)
        count = int(
            ctx.execute(
                "SELECT count(*) FROM task_attempts WHERE task_id=?", (p.task_id,)
            ).fetchone()[0]
        )
        require(count < min(grant.max_attempts, spec.max_attempts), "ATTEMPT_LIMIT")
        require(accounting.fits(ctx, cmd.run_id, grant, spec), "BUDGET_EXCEEDED")
        gate = self._gate(ctx, cmd)
        expiry = min(
            self.clock() + timedelta(seconds=p.lease_ttl_seconds),
            time_value(gate["expires_at_utc"]),
        )
        token = dto.AttemptToken(
            task_id=p.task_id,
            attempt_id=p.attempt_id,
            holder=p.worker_id,
            generation=count + 1,
            gate_generation=gate["generation"],
        )
        ctx.execute(
            "INSERT INTO task_attempts(attempt_id,task_id,attempt_no,lease_generation,holder,expires_at_utc,state,gate_generation,plan_version_id,grant_id) VALUES(?,?,?,?,?,?,'RESERVED',?,?,?)",
            (
                p.attempt_id,
                p.task_id,
                count + 1,
                count + 1,
                p.worker_id,
                stamp(expiry),
                gate["generation"],
                run["active_plan_version_id"],
                cmd.grant_id,
            ),
        )
        ctx.execute(
            "INSERT INTO attempt_accounts(attempt_id,status,max_cost_usd,max_tokens,max_requests) VALUES(?,'HELD',?,?,?)",
            (p.attempt_id, str(spec.max_cost_usd), spec.max_tokens, spec.max_requests),
        )
        accounting.ledger(
            ctx,
            cmd.run_id,
            p.task_id,
            p.attempt_id,
            "RESERVE",
            spec.max_cost_usd,
            "ESTIMATED",
            stamp(self.clock()),
        )
        result = self._move_task(
            ctx, cmd, run, task, "RESERVED", "ATOMIC_BUDGET_LEASE", p.attempt_id
        )
        return dto.CommandResult.model_validate({**result.model_dump(), "token": token})

    def _attempt(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        p: dto.StartAttempt | dto.RenewAttempt | dto.FinishAttempt,
    ) -> dto.CommandResult:
        token = p.token
        task = self._task(ctx, cmd, token.task_id)
        gate = self._gate(ctx, cmd)
        row = ctx.execute(
            "SELECT * FROM task_attempts WHERE attempt_id=? AND task_id=?",
            (token.attempt_id, token.task_id),
        ).fetchone()
        require(row is not None, "ATTEMPT_NOT_FOUND")
        assert isinstance(row, sqlite3.Row)
        require(
            row["holder"] == token.holder
            and row["lease_generation"] == token.generation
            and row["gate_generation"] == token.gate_generation == gate["generation"],
            "STALE_FENCING_TOKEN",
        )
        require(
            row["plan_version_id"] == run["active_plan_version_id"]
            and row["grant_id"] == cmd.grant_id,
            "STALE_ATTEMPT_PLAN_OR_GRANT",
        )
        require(
            row["state"] in {"RESERVED", "RUNNING"}
            and time_value(row["expires_at_utc"]) > self.clock(),
            "ATTEMPT_LEASE_EXPIRED_OR_FINISHED",
        )
        if isinstance(p, dto.RenewAttempt):
            require(run["state"] in {"RUNNING", "PAUSING"}, "DISPATCH_STOPPED")
            self._grant(ctx, cmd, run)
            expires = min(
                self.clock() + timedelta(seconds=p.lease_ttl_seconds),
                time_value(gate["expires_at_utc"]),
            )
            ctx.execute(
                "UPDATE task_attempts SET expires_at_utc=?,revision=revision+1 WHERE attempt_id=?",
                (stamp(expires), token.attempt_id),
            )
            ctx.execute("UPDATE tasks SET revision=revision+1 WHERE task_id=?", (token.task_id,))
            self._emit(
                ctx,
                cmd,
                run,
                token.task_id,
                str(task["state"]),
                str(task["state"]),
                int(task["revision"]),
                task_id=token.task_id,
                attempt_id=token.attempt_id,
            )
            return dto.CommandResult(
                accepted=True, code="COMMITTED", revision=int(task["revision"]) + 1, token=token
            )
        if isinstance(p, dto.StartAttempt):
            require(run["state"] == "RUNNING", "DISPATCH_STOPPED")
            self._eligible(ctx, cmd, run, task)
            require(row["state"] == "RESERVED", "ATTEMPT_ALREADY_STARTED")
            ctx.execute(
                "UPDATE task_attempts SET state='RUNNING',revision=revision+1 WHERE attempt_id=?",
                (token.attempt_id,),
            )
            return self._move_task(
                ctx, cmd, run, task, "RUNNING", "WORKER_ISSUE_RECORDED", token.attempt_id
            )
        require(
            row["state"] == "RUNNING" and run["state"] in {"RUNNING", "PAUSING"},
            "ATTEMPT_NOT_ISSUED",
        )
        if p.target == "SUCCEEDED":
            self._grant(ctx, cmd, run)
            assert p.result_ref is not None
            self._attach(ctx, p.result_ref, "RUN", cmd.run_id, "RESULT_" + token.attempt_id)
        if p.target == "CANCELLED":
            require(p.safe_cancellation, "UNSAFE_CANCELLATION")
        ctx.execute(
            "UPDATE task_attempts SET state=?,output_artifact_id=?,revision=revision+1 WHERE attempt_id=?",
            (
                p.target,
                None if p.result_ref is None else p.result_ref.artifact_id,
                token.attempt_id,
            ),
        )
        accounting.settle(
            ctx, cmd.run_id, token.task_id, token.attempt_id, p.usage, stamp(self.clock())
        )
        result = self._move_task(ctx, cmd, run, task, p.target, p.reason, token.attempt_id)
        ctx.execute(
            "UPDATE tasks SET not_before_utc=? WHERE task_id=?",
            (None if p.not_before_utc is None else stamp(p.not_before_utc), token.task_id),
        )
        policy = self._grant_for_accounting(ctx, cmd.grant_id)
        if (
            policy is not None
            and not accounting.fits(ctx, cmd.run_id, policy)
            and run["state"] == "RUNNING"
        ):
            self._move_run(ctx, cmd, run, "PAUSING", "ACCOUNTED_BUDGET_OVERRUN")
        return result

    @staticmethod
    def _grant_for_accounting(ctx: WriteContext, grant_id: str | None) -> dto.GrantPolicy | None:
        row = ctx.execute("SELECT scope_json FROM grants WHERE grant_id=?", (grant_id,)).fetchone()
        return None if row is None else parse_contract(dto.GrantPolicy, row[0])

    def _usage(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row, p: dto.ReconcileUsage
    ) -> dto.CommandResult:
        row = ctx.execute(
            "SELECT a.*,t.run_id FROM task_attempts a JOIN tasks t USING(task_id) WHERE attempt_id=?",
            (p.attempt_id,),
        ).fetchone()
        require(row is not None and row["run_id"] == cmd.run_id, "ATTEMPT_NOT_FOUND")
        assert isinstance(row, sqlite3.Row)
        require(
            cmd.aggregate_id == p.attempt_id and row["revision"] == cmd.expected_revision,
            "CONFLICT_STALE_REVISION",
        )
        require(row["state"] not in {"RESERVED", "RUNNING"}, "ATTEMPT_STILL_ACTIVE")
        ctx.execute(
            "UPDATE task_attempts SET revision=revision+1 WHERE attempt_id=?", (p.attempt_id,)
        )
        require(p.usage.state == "KNOWN", "RECONCILIATION_REQUIRES_KNOWN_USAGE")
        self._attach(ctx, p.observation_ref, "RUN", cmd.run_id, "BILLING_RECONCILIATION")
        accounting.settle(
            ctx, cmd.run_id, str(row["task_id"]), p.attempt_id, p.usage, stamp(self.clock())
        )
        self._emit(
            ctx,
            cmd,
            run,
            p.attempt_id,
            None,
            None,
            int(row["revision"]),
            reason="OBSERVED_USAGE_RECONCILED",
            task_id=str(row["task_id"]),
            attempt_id=p.attempt_id,
        )
        policy = self._grant_for_accounting(ctx, row["grant_id"])
        if policy and not accounting.fits(ctx, cmd.run_id, policy) and run["state"] == "RUNNING":
            self._move_run(ctx, cmd, run, "PAUSING", "ACCOUNTED_BUDGET_OVERRUN")
        return dto.CommandResult(accepted=True, code="USAGE_RECONCILED")

    def _recover(
        self, ctx: WriteContext, cmd: dto.StateCommand, run: sqlite3.Row
    ) -> dto.CommandResult:
        require(run["state"] in ACTIVE | {"INTERRUPTED"}, "RECOVERY_STATE_INVALID")
        gate = ctx.execute("SELECT * FROM active_run_gate WHERE singleton=1").fetchone()
        assert isinstance(gate, sqlite3.Row)
        require(gate["run_id"] == cmd.run_id, "RECOVERY_GATE_MISMATCH")
        # Exclusive lifetime OS ownership is already held; TTL/PID never proves death.
        if run["state"] in {"RUNNING", "PAUSING"}:
            self._move_run(ctx, cmd, run, "INTERRUPTED", "LOCAL_INTERRUPTION_RECONCILIATION")
            run = self._run(ctx, cmd, revision=False)
        if run["state"] == "INTERRUPTED":
            self._move_run(ctx, cmd, run, "RECOVERING", "EXCLUSIVE_INSTANCE_RECOVERY")
            run = self._run(ctx, cmd, revision=False)
        ctx.execute(
            "UPDATE active_run_gate SET generation=generation+1,holder=?,expires_at_utc=?,updated_at_utc=? WHERE singleton=1",
            (self.nonce, stamp(self.clock() + timedelta(seconds=120)), stamp(self.clock())),
        )
        attempts = ctx.execute(
            "SELECT a.* FROM task_attempts a JOIN tasks t USING(task_id) WHERE t.run_id=? AND a.state IN ('RESERVED','RUNNING') ORDER BY attempt_id",
            (cmd.run_id,),
        ).fetchall()
        for row in attempts:
            task = ctx.execute("SELECT * FROM tasks WHERE task_id=?", (row["task_id"],)).fetchone()
            assert task is not None
            ctx.execute(
                "UPDATE task_attempts SET state='EXPIRED',revision=revision+1 WHERE attempt_id=?",
                (row["attempt_id"],),
            )
            if row["state"] == "RESERVED":
                ctx.execute(
                    "UPDATE task_attempts SET usage_state='KNOWN' WHERE attempt_id=?",
                    (row["attempt_id"],),
                )
                ctx.execute(
                    "UPDATE attempt_accounts SET status='RELEASED',cost_usd='0',tokens=0,requests=0 WHERE attempt_id=?",
                    (row["attempt_id"],),
                )
                account = ctx.execute(
                    "SELECT max_cost_usd FROM attempt_accounts WHERE attempt_id=?",
                    (row["attempt_id"],),
                ).fetchone()
                assert account is not None
                from decimal import Decimal

                accounting.ledger(
                    ctx,
                    cmd.run_id,
                    str(row["task_id"]),
                    str(row["attempt_id"]),
                    "RELEASE",
                    Decimal(account[0]),
                    "KNOWN",
                    stamp(self.clock()),
                )
                target = "READY"
            else:
                accounting.settle(
                    ctx,
                    cmd.run_id,
                    str(row["task_id"]),
                    str(row["attempt_id"]),
                    dto.Usage(state="UNKNOWN_USAGE"),
                    stamp(self.clock()),
                )
                target = "RETRY_WAIT"
            self._move_task(
                ctx, cmd, run, task, target, "INTERRUPTED_USAGE_RECONCILED", str(row["attempt_id"])
            )
        ctx.execute("UPDATE runs SET revision=revision+1 WHERE run_id=?", (cmd.run_id,))
        self._emit(
            ctx,
            cmd,
            run,
            cmd.run_id,
            str(run["state"]),
            str(run["state"]),
            int(run["revision"]),
            reason="NO_AUTOMATIC_REMOTE_RESUME",
        )
        return dto.CommandResult(
            accepted=True,
            code="RECOVERY_RECONCILED",
            state=str(run["state"]),
            revision=int(run["revision"]) + 1,
        )

    def _recover_startup(self) -> None:
        with self.storage.database.reader() as conn:
            gate = conn.execute("SELECT run_id FROM active_run_gate WHERE singleton=1").fetchone()
            assert isinstance(gate, sqlite3.Row)
            active = conn.execute(
                "SELECT run_id FROM runs WHERE state IN ('RUNNING','PAUSING','RECOVERING','COMPLETING','INTERRUPTED')"
            ).fetchall()
            require(
                not active or (len(active) == 1 and active[0][0] == gate[0]),
                "INCONSISTENT_ACTIVE_GATE",
            )
            run_id = gate[0]
        if run_id is None:
            return

        def make(payload: dto.RecoverRun | dto.ChangeRun) -> dto.StateCommand:
            with self.storage.database.reader() as conn:
                row = conn.execute(
                    "SELECT r.*,p.version FROM runs r LEFT JOIN plan_versions p ON p.plan_version_id=r.active_plan_version_id WHERE r.run_id=?",
                    (run_id,),
                ).fetchone()
                assert isinstance(row, sqlite3.Row)
            return dto.StateCommand(
                command_id=new_id("CMD"),
                trace_id=new_id("TRACE"),
                run_id=run_id,
                actor="SYSTEM",
                command_type=payload.action,
                aggregate_id=run_id,
                expected_revision=row["revision"],
                plan_version=row["version"] or 1,
                grant_id=None,
                idempotency_key=self.nonce + payload.action,
                payload=payload,
                created_at_utc=self.clock(),
            )

        first = self.handle(make(dto.RecoverRun()), actor="SYSTEM")
        require(first.accepted, "STARTUP_RECOVERY_FAILED")
        checkpoint = self.checkpoint(str(run_id))
        final = self.handle(
            make(
                dto.ChangeRun(
                    target="PAUSED",
                    reason="STARTUP_RECONCILED_NO_AUTO_RESUME",
                    checkpoint_ref=checkpoint,
                )
            ),
            actor="SYSTEM",
        )
        require(final.accepted, "STARTUP_CHECKPOINT_FAILED")

    def _workflow(
        self,
        ctx: WriteContext,
        cmd: dto.StateCommand,
        run: sqlite3.Row,
        p: dto.CreateWorkflow | dto.ChangeWorkflow,
        content: dict[str, bytes],
    ) -> dto.CommandResult:
        require(
            run["state"] not in {"COMPLETED", "FAILED", "CANCELLED"}, "HISTORICAL_RUN_PROTECTED"
        )
        tables = {
            "LEAD": ("leads", "lead_id", "state"),
            "PROPOSAL": ("critic_proposals", "proposal_id", "status"),
            "PLAN_CHANGE": ("plan_change_proposals", "proposal_id", "status"),
            "ASSESSMENT": ("assessments", "assessment_id", "lifecycle"),
        }
        table, key, state = tables[p.kind]
        if isinstance(p, dto.CreateWorkflow):
            require(
                cmd.expected_revision == 0 and cmd.aggregate_id == p.entity_id, "AGGREGATE_MISMATCH"
            )
            require(run["active_plan_version_id"] is not None, "VALIDATED_PLAN_REQUIRED")
            if p.kind == "LEAD":
                require(p.entity_id.startswith("LEAD-"), "LEAD_ID_INVALID")
                if p.question_id:
                    require(
                        ctx.execute(
                            "SELECT 1 FROM questions WHERE question_id=? AND plan_version_id=?",
                            (p.question_id, run["active_plan_version_id"]),
                        ).fetchone()
                        is not None,
                        "LEAD_QUESTION_OUTSIDE_PLAN",
                    )
                ctx.execute(
                    "INSERT INTO leads(lead_id,run_id,question_id,state,reason_code,plan_version_id) VALUES(?,?,?,'DISCOVERED',?,?)",
                    (
                        p.entity_id,
                        cmd.run_id,
                        p.question_id,
                        p.reason,
                        run["active_plan_version_id"],
                    ),
                )
                initial = "DISCOVERED"
            else:
                require(p.snapshot_ref is not None, "WORKFLOW_SNAPSHOT_REQUIRED")
                assert p.snapshot_ref
                self.artifacts.register(ctx, p.snapshot_ref)
                if p.kind == "PROPOSAL":
                    parse_contract(dto.ProposalScope, content[p.snapshot_ref.artifact_id])
                    duplicate = ctx.execute(
                        "SELECT proposal_id FROM critic_proposals WHERE run_id=? AND fingerprint=? AND material_version=?",
                        (cmd.run_id, p.fingerprint, p.material_version),
                    ).fetchone()
                    require(duplicate is None, "DUPLICATE_PROPOSAL_SUPPRESSED")
                    ctx.execute(
                        "INSERT INTO critic_proposals(proposal_id,run_id,fingerprint,material_version,snapshot_artifact_id,status,plan_version_id) VALUES(?,?,?,?,?,'PROPOSED',?)",
                        (
                            p.entity_id,
                            cmd.run_id,
                            p.fingerprint,
                            p.material_version,
                            p.snapshot_ref.artifact_id,
                            run["active_plan_version_id"],
                        ),
                    )
                else:
                    parse_contract(ResearchPlanProposal, content[p.snapshot_ref.artifact_id])
                    ctx.execute(
                        "INSERT INTO plan_change_proposals(proposal_id,run_id,expected_version,status,snapshot_artifact_id) VALUES(?,?,?,'PROPOSED',?)",
                        (p.entity_id, cmd.run_id, run["plan_version"], p.snapshot_ref.artifact_id),
                    )
                self._attach(ctx, p.snapshot_ref, "RUN", cmd.run_id, "WORKFLOW_SNAPSHOT")
                initial = "PROPOSED"
            self._emit(ctx, cmd, run, p.entity_id, None, initial, 0, reason=p.reason)
            return dto.CommandResult(accepted=True, code="COMMITTED", state=initial, revision=0)
        require(cmd.aggregate_id == p.entity_id, "AGGREGATE_MISMATCH")
        row = ctx.execute(f"SELECT * FROM {table} WHERE {key}=?", (p.entity_id,)).fetchone()
        require(row is not None, "WORKFLOW_NOT_FOUND")
        assert isinstance(row, sqlite3.Row)
        if p.kind == "ASSESSMENT":
            claim = ctx.execute(
                "SELECT run_id FROM claims WHERE claim_id=?", (row["claim_id"],)
            ).fetchone()
            require(claim is not None and claim[0] == cmd.run_id, "WORKFLOW_RUN_MISMATCH")
        else:
            require(row["run_id"] == cmd.run_id, "WORKFLOW_RUN_MISMATCH")
        require(row["revision"] == cmd.expected_revision, "CONFLICT_STALE_REVISION")
        require(allowed(p.kind, row[state], p.target), "ILLEGAL_WORKFLOW_TRANSITION")
        if p.kind == "ASSESSMENT":
            require(p.target != "COMMITTED", "ASSESSMENT_ELIGIBILITY_REQUIRES_B10")
            if p.target == "VALIDATING":
                require(row["validation_attempts"] < 3, "VALIDATION_ATTEMPT_LIMIT")
                ctx.execute(
                    "UPDATE assessments SET validation_attempts=validation_attempts+1 WHERE assessment_id=?",
                    (p.entity_id,),
                )
            if row[state] == "CONFLICT_REVIEW":
                require(p.condition_change_ref is not None, "EXPLICIT_REVIEW_ROUTE_REQUIRED")
        if p.kind == "PROPOSAL" and p.target == "APPLIED":
            raise Rejected("PROPOSAL_APPLICATION_REQUIRES_B11")
        if p.kind == "PLAN_CHANGE":
            require(p.target != "APPROVED", "PLAN_PUBLICATION_COMMAND_REQUIRED")
        if p.target in {"USER_APPROVED", "USER_REJECTED"}:
            require(cmd.actor == "USER", "USER_APPROVAL_REQUIRED")
        if p.kind == "LEAD" and row[state] == "DEFERRED":
            require(p.condition_change_ref is not None, "MATERIAL_CHANGE_REQUIRED")
        if p.kind == "PROPOSAL" and p.target in {"AUTO_AUTHORIZED", "USER_APPROVED", "SCHEDULED"}:
            require(
                row["plan_version_id"] == run["active_plan_version_id"],
                "CONFLICT_STALE_PLAN_VERSION",
            )
            grant = self._grant(ctx, cmd, run)
            # Snapshot is read in handle() before BEGIN (see _references_for_workflow).
            require(row["snapshot_artifact_id"] in content, "PROPOSAL_SNAPSHOT_NOT_PREPARED")
            scope = parse_contract(dto.ProposalScope, content[row["snapshot_artifact_id"]])
            require(
                set(scope.providers) <= set(grant.providers)
                and set(scope.categories) <= set(grant.categories)
                and set(scope.task_types) <= set(grant.task_types),
                "PROPOSAL_AWAITING_APPROVAL",
            )
            cost, tokens, requests = accounting.total(ctx, cmd.run_id)
            require(
                cost + scope.estimated_cost_usd <= grant.max_cost_usd
                and tokens + scope.estimated_tokens <= grant.max_tokens
                and requests + scope.estimated_requests <= grant.max_requests,
                "BUDGET_EXCEEDED",
            )
        if p.condition_change_ref:
            self._attach(ctx, p.condition_change_ref, "RUN", cmd.run_id, "WORKFLOW_REVIEW")
        ctx.execute(
            f"UPDATE {table} SET {state}=?,revision=revision+1 WHERE {key}=?",
            (p.target, p.entity_id),
        )
        self._emit(
            ctx,
            cmd,
            run,
            p.entity_id,
            str(row[state]),
            p.target,
            int(row["revision"]),
            reason=p.reason,
        )
        return dto.CommandResult(
            accepted=True, code="COMMITTED", state=p.target, revision=int(row["revision"]) + 1
        )
