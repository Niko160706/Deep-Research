"""Synthetic, local-only control-plane fixtures; no provider integration."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

from gidr.application.commands import StateAuthority
from gidr.contracts import commands as dto
from gidr.contracts.common import new_id
from gidr.contracts.control import Question, ResearchPlanProposal
from gidr.infrastructure.artifacts.repository import ArtifactRepository
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.db.authority import PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.db.migrate import migrate


class Clock:
    def __init__(self):
        self.value = datetime(2020, 1, 1, tzinfo=UTC)

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += timedelta(seconds=seconds)


class Harness:
    def __init__(self, root):
        self.root = root
        self.clock = Clock()
        self.db = Database(root / "state/db.sqlite3")
        migrate(self.db)
        self.storage = PersistenceAuthority(self.db)
        self.store = ArtifactStore(root / "artifacts")
        self.repo = ArtifactRepository(self.store, self.storage)
        self.authority = StateAuthority(self.storage, self.repo, clock=self.clock)

    def row(self, table, key, value):
        with self.db.reader() as conn:
            return conn.execute(f"SELECT * FROM {table} WHERE {key}=?", (value,)).fetchone()

    def version(self, run):
        with self.db.reader() as conn:
            row = conn.execute(
                "SELECT p.version FROM runs r LEFT JOIN plan_versions p ON p.plan_version_id=r.active_plan_version_id WHERE r.run_id=?",
                (run,),
            ).fetchone()
        return (row[0] if row else None) or 1

    def command(
        self,
        payload,
        *,
        run="RUN-one",
        actor="USER",
        aggregate=None,
        grant="GRANT-one",
        revision=None,
        command_id=None,
    ):
        if aggregate is None:
            if isinstance(payload, (dto.ChangeTask, dto.ReserveTask)):
                aggregate = payload.task_id
            elif isinstance(
                payload,
                (dto.StartAttempt, dto.RenewAttempt, dto.FinishAttempt, dto.ReleaseReservation),
            ):
                aggregate = payload.token.task_id
            elif isinstance(payload, (dto.CreateWorkflow, dto.ChangeWorkflow)):
                aggregate = payload.entity_id
            elif isinstance(payload, dto.ReconcileUsage):
                aggregate = payload.attempt_id
            elif isinstance(payload, dto.RevokeGrant):
                aggregate = grant
            else:
                aggregate = run
        if revision is None:
            if aggregate.startswith("TASK-"):
                row = self.row("tasks", "task_id", aggregate)
            elif isinstance(payload, dto.ReconcileUsage):
                row = self.row("task_attempts", "attempt_id", aggregate)
            elif isinstance(payload, (dto.ChangeWorkflow, dto.CreateWorkflow)):
                table, key = {
                    "LEAD": ("leads", "lead_id"),
                    "PROPOSAL": ("critic_proposals", "proposal_id"),
                    "PLAN_CHANGE": ("plan_change_proposals", "proposal_id"),
                    "ASSESSMENT": ("assessments", "assessment_id"),
                }[payload.kind]
                row = self.row(table, key, aggregate)
            elif isinstance(payload, dto.RevokeGrant):
                row = self.row("grants", "grant_id", grant)
            else:
                row = self.row("runs", "run_id", run)
            revision = row["revision"] if row else 0
        return dto.StateCommand(
            command_id=command_id or new_id("CMD"),
            trace_id=new_id("TRACE"),
            run_id=run,
            actor=actor,
            command_type=payload.action,
            aggregate_id=aggregate,
            expected_revision=revision,
            plan_version=self.version(run),
            grant_id=grant,
            idempotency_key=command_id or new_id("CMD"),
            payload=payload,
            created_at_utc=self.clock(),
        )

    def send(self, payload, **kwargs):
        command = self.command(payload, **kwargs)
        holder = (
            payload.token.holder
            if isinstance(payload, (dto.StartAttempt, dto.RenewAttempt, dto.FinishAttempt))
            else None
        )
        return self.authority.port(command.actor, holder).handle(command)

    def policy(self, grant="GRANT-one", **changes):
        return dto.GrantPolicy(
            grant_id=grant,
            providers=("local",),
            categories=("local",),
            task_types=("compute", "other"),
            max_cost_usd=Decimal(2),
            max_tokens=100,
            max_requests=10,
            **changes,
        )

    def plan(self, run="RUN-one", name="one", expected=0):
        budget = self.store.publish(b"Synthetic budget policy, no real billing")
        return ResearchPlanProposal(
            run_id=run,
            created_at_utc=self.clock(),
            proposal_id="PROP-plan-" + name,
            project_id="PRJ-" + name,
            session_id="SES-" + name,
            expected_plan_version=expected,
            objective="Synthetic offline control-plane test",
            scope_in=("Synthetic local computation",),
            questions=(
                Question(
                    question_id="RQ-" + name,
                    text="Synthetic question?",
                    evidence_requirements=("Original synthetic fixture",),
                ),
            ),
            source_categories=("local",),
            budget_policy_ref=budget,
        )

    def setup(self, name="one", *, start=True, policy=None):
        run = "RUN-" + name
        result = self.send(
            dto.CreateRun(
                project_id="PRJ-" + name,
                session_id="SES-" + name,
                title="Synthetic fixture",
                owner="fixture",
            ),
            run=run,
        )
        assert result.accepted, result
        assert self.send(dto.ChangeRun(target="PLANNING", reason="USER_REQUEST"), run=run).accepted
        ref = self.store.publish(self.plan(run, name).model_dump_json().encode())
        result = self.send(
            dto.PublishPlan(snapshot_ref=ref, plan_id="PLAN-" + name, plan_version_id="PV-" + name),
            run=run,
        )
        assert result.accepted, result
        assert self.send(
            dto.ChangeRun(target="PLAN_READY", reason="PREVIEW_VALIDATED"), run=run
        ).accepted
        if start:
            policy = policy or self.policy("GRANT-" + name)
            result = self.send(dto.StartRun(grant=policy), run=run, grant=policy.grant_id)
            assert result.accepted, result
        return run

    def task(self, name="one", **changes):
        ref = self.store.publish(("Synthetic task input " + name).encode())
        return dto.ScheduledTask(
            task_id="TASK-" + name,
            question_id="RQ-one",
            task_type="compute",
            reason_code="SYNTHETIC_TEST",
            input_refs=(ref,),
            **changes,
        )

    def add(self, *specs):
        result = self.send(dto.AddTasks(tasks=specs), actor="SYSTEM")
        assert result.accepted, result
        return result

    def ready(self, task="TASK-one", **kwargs):
        return self.send(
            dto.ChangeTask(task_id=task, target="READY", reason="POLICY_VALIDATED"),
            actor="SCHEDULER",
            **kwargs,
        )

    def reserve(self, task="TASK-one", attempt="ATT-one", **kwargs):
        return self.send(
            dto.ReserveTask(task_id=task, attempt_id=attempt, worker_id="worker"),
            actor="SCHEDULER",
            **kwargs,
        )

    def running(self, task="TASK-one", attempt="ATT-one"):
        result = self.reserve(task, attempt)
        assert result.accepted, result
        token = result.token
        assert token
        result = self.send(dto.StartAttempt(token=token), actor="WORKER")
        assert result.accepted, result
        return token

    def finish(self, token, **changes):
        ref = self.store.publish(b"Synthetic validated output")
        payload = dto.FinishAttempt(
            token=token,
            target="SUCCEEDED",
            usage=dto.Usage(state="KNOWN", cost_usd=Decimal(0), tokens=0, requests=0),
            reason="SYNTHETIC_VALIDATED",
            result_ref=ref,
        )
        if changes:
            payload = dto.FinishAttempt.model_validate({**payload.model_dump(), **changes})
        return self.send(payload, actor="WORKER")

    def pause(self):
        assert self.send(dto.ChangeRun(target="PAUSING", reason="USER_PAUSE")).accepted
        ref = self.authority.checkpoint("RUN-one")
        return self.send(
            dto.ChangeRun(target="PAUSED", reason="CHECKPOINT_DURABLE", checkpoint_ref=ref)
        )
