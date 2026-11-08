"""B03 transaction, authorization, concurrency and recovery invariants."""

import asyncio
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest

from gidr.application.commands import Rejected, StateAuthority
from gidr.application.writer import AsyncStateWriter
from gidr.contracts import commands as dto
from gidr.infrastructure.db.migrate import bundled_migrations, migrate

spec = importlib.util.spec_from_file_location(
    "gidr_state_support", Path(__file__).with_name("state_support.py")
)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


@pytest.fixture
def h(tmp_path):
    harness = support.Harness(tmp_path)
    with harness.authority:
        yield harness


def test_preview_has_no_grant_or_dispatch_and_role_cannot_be_forged(h):
    h.setup(start=False)
    h.add(h.task())
    assert not h.ready().accepted
    assert not h.reserve().accepted
    with h.db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM grants").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM task_attempts").fetchone()[0] == 0
    command = h.command(dto.StartRun(grant=h.policy()))
    with pytest.raises(Rejected, match="ACTOR_NOT_AUTHORIZED"):
        h.authority.port("WORKER", "worker").handle(command)
    assert h.row("runs", "run_id", "RUN-one")["state"] == "PLAN_READY"


def test_two_schedulers_one_reservation_and_replay(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    revision = h.row("tasks", "task_id", "TASK-one")["revision"]
    commands = [
        h.command(
            dto.ReserveTask(task_id="TASK-one", attempt_id="ATT-" + str(i), worker_id="worker"),
            actor="SCHEDULER",
            revision=revision,
        )
        for i in range(2)
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(h.authority.port("SCHEDULER").handle, commands))
    assert sum(r.accepted for r in results) == 1
    winner = next(c for c, r in zip(commands, results, strict=True) if r.accepted)
    assert h.authority.port("SCHEDULER").handle(winner) == next(r for r in results if r.accepted)
    with h.db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM task_attempts").fetchone()[0] == 1
        assert (
            conn.execute("SELECT count(*) FROM budget_ledger WHERE kind='RESERVE'").fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE event_type='command.rejected'"
            ).fetchone()[0]
            == 1
        )


@pytest.mark.parametrize("resource", ["cost", "tokens", "requests"])
def test_concurrent_tasks_cannot_spend_same_remaining_budget(h, resource):
    h.setup()
    kwargs = (
        {"max_cost_usd": Decimal("1.1")}
        if resource == "cost"
        else {"max_tokens": 60}
        if resource == "tokens"
        else {"max_requests": 6}
    )
    h.add(h.task("one", **kwargs), h.task("two", **kwargs))
    assert h.ready("TASK-one").accepted and h.ready("TASK-two").accepted
    commands = [
        h.command(
            dto.ReserveTask(task_id="TASK-" + name, attempt_id="ATT-" + name, worker_id="worker"),
            actor="SCHEDULER",
        )
        for name in ("one", "two")
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(h.authority.port("SCHEDULER").handle, commands))
    assert sum(x.accepted for x in results) == 1
    assert next(x for x in results if not x.accepted).code == "BUDGET_EXCEEDED"


def test_dependency_cycle_rolls_back_whole_batch_and_dependencies_gate_ready(h):
    h.setup()
    first = h.task("one", dependencies=("TASK-two",))
    second = h.task("two", dependencies=("TASK-one",))
    bad = h.send(dto.AddTasks(tasks=(first, second)), actor="SYSTEM")
    assert bad.code == "DEPENDENCY_CYCLE"
    assert h.row("tasks", "task_id", "TASK-one") is None
    h.add(h.task("one"), h.task("two", dependencies=("TASK-one",)))
    assert h.ready("TASK-two").code == "DEPENDENCIES_NOT_SATISFIED"
    assert h.ready().accepted
    token = h.running()
    assert h.finish(token).accepted
    assert h.ready("TASK-two").accepted


def test_worker_result_reference_state_usage_and_event_commit_atomically(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    ref = h.store.publish(b"Synthetic committed output")
    payload = dto.FinishAttempt(
        token=token,
        target="SUCCEEDED",
        usage=dto.Usage(state="KNOWN", cost_usd=Decimal(".1"), tokens=5, requests=1),
        reason="VALIDATED",
        result_ref=ref,
    )
    command = h.command(payload, actor="WORKER")
    result = h.authority.port("WORKER", "worker").handle(command)
    assert result.accepted
    assert h.authority.port("WORKER", "worker").handle(command) == result
    assert h.row("tasks", "task_id", "TASK-one")["state"] == "SUCCEEDED"
    assert h.row("task_attempts", "attempt_id", "ATT-one")["output_artifact_id"] == ref.artifact_id
    assert h.row("attempt_accounts", "attempt_id", "ATT-one")["cost_usd"] == "0.1"
    with h.db.reader() as conn:
        assert (
            conn.execute("SELECT count(*) FROM budget_ledger WHERE kind='SETTLE'").fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM artifact_refs WHERE artifact_id=?", (ref.artifact_id,)
            ).fetchone()[0]
            >= 1
        )
        h.db.check(conn)


def test_unknown_usage_holds_budget_requires_explicit_retry_and_known_reconciliation(h):
    h.setup()
    h.add(h.task(max_cost_usd=Decimal("1.5")))
    assert h.ready().accepted
    token = h.running()
    result = h.finish(
        token, target="RETRY_WAIT", result_ref=None, usage=dto.Usage(state="UNKNOWN_USAGE")
    )
    assert result.accepted
    account = h.row("attempt_accounts", "attempt_id", "ATT-one")
    assert account["status"] == "UNCERTAIN" and account["cost_usd"] is None
    assert h.ready().code == "UNCERTAIN_RETRY_REQUIRES_USER_APPROVAL"
    assert h.send(
        dto.ChangeTask(task_id="TASK-one", target="READY", reason="APPROVED_RETRY")
    ).accepted
    assert h.reserve(attempt="ATT-two").code == "BUDGET_EXCEEDED"
    observation = h.store.publish(b"Synthetic billing observation; no real API called")
    known = dto.Usage(state="KNOWN", cost_usd=Decimal(".1"), tokens=1, requests=1)
    assert h.send(
        dto.ReconcileUsage(attempt_id="ATT-one", usage=known, observation_ref=observation),
        actor="SYSTEM",
    ).accepted
    assert h.reserve(attempt="ATT-three").accepted


def test_usage_overrun_is_accounted_and_stops_dispatch(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    result = h.finish(
        token, usage=dto.Usage(state="KNOWN", cost_usd=Decimal(3), tokens=0, requests=0)
    )
    assert result.accepted
    assert h.row("attempt_accounts", "attempt_id", "ATT-one")["cost_usd"] == "3"
    assert h.row("runs", "run_id", "RUN-one")["state"] == "PAUSING"


def test_session_switch_stops_dispatch_and_waits_for_checkpoint(h):
    h.setup()
    h.setup("two", start=False)
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    result = h.send(dto.SwitchSession(session_id="SES-two"))
    assert result.accepted
    with h.db.reader() as conn:
        assert (
            conn.execute("SELECT pending_session_id FROM navigation_state").fetchone()[0]
            == "SES-two"
        )
    assert (
        h.send(dto.OpenSession(session_id="SES-two"), run="RUN-two").code
        == "SESSION_SWITCH_REQUIRES_CHECKPOINT"
    )
    assert (
        h.send(dto.ChangeRun(target="PAUSED", reason="PAUSE_WITHOUT_CHECKPOINT")).code
        == "INFLIGHT_RECONCILIATION_REQUIRED"
    )
    assert h.finish(token).accepted  # safe drain accepts the already issued, fenced result
    assert (
        h.send(dto.ChangeRun(target="PAUSED", reason="MISSING_CHECKPOINT")).code
        == "DURABLE_CHECKPOINT_REQUIRED"
    )
    ref = h.authority.checkpoint("RUN-one")
    assert h.send(dto.ChangeRun(target="PAUSED", reason="DURABLE", checkpoint_ref=ref)).accepted
    with h.db.reader() as conn:
        nav = conn.execute("SELECT * FROM navigation_state").fetchone()
        assert nav["session_id"] == "SES-two" and nav["pending_session_id"] is None
        assert conn.execute("SELECT run_id FROM active_run_gate").fetchone()[0] is None
    assert h.send(dto.OpenSession(session_id="SES-one")).code == "SESSION_OPENED_NO_RESUME"
    assert h.row("runs", "run_id", "RUN-one")["state"] == "PAUSED"
    assert h.send(dto.ResumeRun()).accepted


def test_failed_or_stale_checkpoint_blocks_second_run(h):
    h.setup()
    h.setup("two", start=False)
    assert h.send(dto.ChangeRun(target="PAUSING", reason="PAUSE")).accepted
    ref = h.authority.checkpoint("RUN-one")
    # A rejected command is itself an event; stale checkpoint must be regenerated.
    assert not h.send(dto.ChangeRun(target="COMPLETED", reason="INVALID")).accepted
    assert (
        h.send(dto.ChangeRun(target="PAUSED", reason="STALE", checkpoint_ref=ref)).code
        == "CHECKPOINT_STALE_OR_MISMATCH"
    )
    assert (
        h.send(dto.StartRun(grant=h.policy("GRANT-two")), run="RUN-two", grant="GRANT-two").code
        == "GLOBAL_RUN_BUSY"
    )
    assert h.row("grants", "grant_id", "GRANT-two") is None  # grant insertion rolled back too


def test_expiry_never_hands_gate_to_other_run_and_fencing_blocks_late_worker(h):
    h.setup()
    h.setup("two", start=False)
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    h.clock.advance(121)
    assert h.finish(token).code == "RUN_LEASE_EXPIRED"
    assert (
        h.send(dto.StartRun(grant=h.policy("GRANT-two")), run="RUN-two", grant="GRANT-two").code
        == "GLOBAL_RUN_BUSY"
    )
    assert h.send(dto.RecoverRun(), actor="SYSTEM").accepted
    assert h.row("attempt_accounts", "attempt_id", "ATT-one")["status"] == "UNCERTAIN"
    ref = h.authority.checkpoint("RUN-one")
    assert h.send(
        dto.ChangeRun(target="PAUSED", reason="RECONCILED", checkpoint_ref=ref), actor="SYSTEM"
    ).accepted
    assert h.send(dto.ResumeRun()).accepted
    assert not h.finish(token).accepted
    assert h.row("tasks", "task_id", "TASK-one")["state"] == "RETRY_WAIT"


def test_exclusive_instance_required_even_when_ttl_expired(h):
    h.setup()
    h.clock.advance(5000)
    other = StateAuthority(h.storage, h.repo, clock=h.clock)
    with pytest.raises(TimeoutError, match="LOCAL_LOCK_TIMEOUT"):
        with other:
            pass
    assert h.row("runs", "run_id", "RUN-one")["state"] == "RUNNING"


def test_provider_tasks_remain_disabled_until_security_gate(h):
    h.setup()
    h.add(h.task(work_kind="PROVIDER"))
    assert h.ready().code == "EXTERNAL_SECURITY_GATE_UNVERIFIED_B04"


def test_grant_expiry_revocation_and_explicit_renewal(h):
    from datetime import timedelta

    policy = h.policy(expires_at_utc=h.clock() + timedelta(seconds=5))
    h.setup(policy=policy)
    h.add(h.task())
    h.clock.advance(6)
    assert h.ready().code == "GRANT_EXPIRED"
    assert h.send(dto.ApproveGrant(grant=h.policy("GRANT-new")), grant="GRANT-new").accepted
    assert h.ready(grant="GRANT-new").accepted
    assert h.send(dto.RevokeGrant(reason="USER_REVOKED"), grant="GRANT-new").accepted
    assert h.reserve(grant="GRANT-new").code == "GRANT_REVOKED"


def test_unissued_reservation_rollback_refunds_without_claiming_a_remote_refund(h):
    h.setup()
    h.add(h.task(max_cost_usd=Decimal("1.5")))
    assert h.ready().accepted
    result = h.reserve()
    assert result.accepted
    assert h.send(
        dto.ReleaseReservation(token=result.token, reason="NOT_ISSUED"), actor="SCHEDULER"
    ).accepted
    assert h.row("attempt_accounts", "attempt_id", "ATT-one")["status"] == "RELEASED"
    assert h.reserve(attempt="ATT-two").accepted


def test_max_attempts_cooldown_and_terminal_success(h):
    h.setup()
    h.add(h.task(max_attempts=1))
    assert h.ready().accepted
    token = h.running()
    assert h.finish(token, target="RETRY_WAIT", result_ref=None).accepted
    assert h.ready().accepted
    assert h.reserve(attempt="ATT-two").code == "ATTEMPT_LIMIT"


def test_bounded_async_writer_serializes_commands_and_closes(h):
    h.setup()
    commands = [h.command(dto.OpenSession(session_id="SES-one")) for _ in range(8)]

    async def scenario():
        async with AsyncStateWriter(h.authority.port("USER"), capacity=2) as writer:
            results = await asyncio.gather(*(writer.submit(c) for c in commands))
            assert all(r.accepted for r in results)
        with pytest.raises(RuntimeError, match="WRITER_CLOSED"):
            await writer.submit(commands[0])

    asyncio.run(scenario())


def test_upgrade_b02_database_keeps_checksums_and_data(tmp_path):
    from gidr.infrastructure.db.database import Database

    db = Database(tmp_path / "state/db.sqlite3")
    migrate(db, target_version=2)
    old = bundled_migrations()[:2]
    with db.reader() as conn:
        hashes = [
            r[0] for r in conn.execute("SELECT checksum FROM schema_migrations ORDER BY version")
        ]
    assert migrate(db) == 3
    with db.reader() as conn:
        assert hashes == [
            r[0]
            for r in conn.execute(
                "SELECT checksum FROM schema_migrations WHERE version<=2 ORDER BY version"
            )
        ]
        assert len(old) == 2
        db.check(conn)


def test_plan_publication_is_atomic_and_stale_worker_cannot_merge(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    plan = h.plan(expected=1)
    ref = h.store.publish(plan.model_dump_json().encode())
    create = dto.CreateWorkflow(
        kind="PLAN_CHANGE", entity_id="PROP-change", snapshot_ref=ref, reason="USER_PROPOSAL"
    )
    assert h.send(create).accepted
    with h.db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM plan_versions").fetchone()[0] == 1
    assert h.send(
        dto.ChangeWorkflow(
            kind="PLAN_CHANGE", entity_id="PROP-change", target="REVIEWED", reason="USER_REVIEW"
        )
    ).accepted
    bad = dto.PublishPlan(
        snapshot_ref=ref,
        plan_id="PLAN-one",
        plan_version_id="PV-two",
        change_proposal_id="PROP-change",
        grant=h.policy("GRANT-two"),
    )
    # A mismatched command grant cannot partially approve the proposal or publish a version.
    assert not h.send(bad, grant="GRANT-wrong").accepted
    assert h.row("plan_change_proposals", "proposal_id", "PROP-change")["status"] == "REVIEWED"
    assert h.row("plan_versions", "plan_version_id", "PV-two") is None
    assert h.send(bad, grant="GRANT-two").accepted
    assert h.version("RUN-one") == 2
    assert h.finish(token).code == "CONFLICT_STALE_PLAN_VERSION"
    assert h.row("tasks", "task_id", "TASK-one")["state"] == "RUNNING"
    assert h.row("task_attempts", "attempt_id", "ATT-one")["output_artifact_id"] is None
    assert h.repo.inventory()["orphans"]
    assert h.send(dto.RecoverRun(), actor="SYSTEM").accepted
    with h.db.reader() as conn:
        event = conn.execute(
            "SELECT event_payload_json FROM audit_events WHERE event_type='recover_run.committed' AND aggregate_id='TASK-one' ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        import json

        data = json.loads(event[0])
        assert data["plan_version"] == 1 and data["grant_id"] == "GRANT-one"


def test_lead_retained_and_proposals_cannot_self_authorize_outside_grant(h):
    h.setup()
    assert h.send(
        dto.CreateWorkflow(
            kind="LEAD", entity_id="LEAD-one", question_id="RQ-one", reason="UNVERIFIED_LEAD"
        )
    ).accepted
    for target in ("TRIAGED", "DEFERRED"):
        assert h.send(
            dto.ChangeWorkflow(kind="LEAD", entity_id="LEAD-one", target=target, reason="SCREENING")
        ).accepted
    assert (
        h.send(
            dto.ChangeWorkflow(
                kind="LEAD", entity_id="LEAD-one", target="QUEUED", reason="NO_CHANGE"
            )
        ).code
        == "MATERIAL_CHANGE_REQUIRED"
    )
    assert h.row("leads", "lead_id", "LEAD-one")["validation_label"] == "UNVERIFIED_LEAD"
    scope = dto.ProposalScope(
        providers=("outside",),
        categories=("local",),
        task_types=("compute",),
        estimated_cost_usd=Decimal(0),
        estimated_tokens=0,
        estimated_requests=0,
    )
    ref = h.store.publish(scope.model_dump_json().encode())
    create = dto.CreateWorkflow(
        kind="PROPOSAL",
        entity_id="PROP-one",
        snapshot_ref=ref,
        fingerprint="gap-one",
        material_version="material-one",
        reason="CRITIC_PROPOSAL",
    )
    assert h.send(create, actor="SYSTEM").accepted
    duplicate = dto.CreateWorkflow.model_validate({**create.model_dump(), "entity_id": "PROP-two"})
    assert h.send(duplicate, actor="SYSTEM").code == "DUPLICATE_PROPOSAL_SUPPRESSED"
    assert h.send(
        dto.ChangeWorkflow(
            kind="PROPOSAL", entity_id="PROP-one", target="VALIDATED", reason="SCHEMA_VALIDATED"
        ),
        actor="SYSTEM",
    ).accepted
    assert (
        h.send(
            dto.ChangeWorkflow(
                kind="PROPOSAL", entity_id="PROP-one", target="AUTO_AUTHORIZED", reason="TRY_AUTO"
            ),
            actor="SYSTEM",
        ).code
        == "PROPOSAL_AWAITING_APPROVAL"
    )
    assert h.send(
        dto.ChangeWorkflow(
            kind="PROPOSAL",
            entity_id="PROP-one",
            target="AWAITING_APPROVAL",
            reason="OUTSIDE_SCOPE",
        ),
        actor="SYSTEM",
    ).accepted
    assert h.row("critic_proposals", "proposal_id", "PROP-one")["status"] == "AWAITING_APPROVAL"
    with h.db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM task_attempts").fetchone()[0] == 0


def test_in_scope_proposal_scheduling_is_control_state_not_scientific_application(h):
    h.setup()
    scope = dto.ProposalScope(
        providers=("local",),
        categories=("local",),
        task_types=("compute",),
        estimated_cost_usd=Decimal(".1"),
        estimated_tokens=1,
        estimated_requests=1,
    )
    ref = h.store.publish(scope.model_dump_json().encode())
    assert h.send(
        dto.CreateWorkflow(
            kind="PROPOSAL", entity_id="PROP-one", snapshot_ref=ref, reason="SYNTHETIC"
        ),
        actor="SYSTEM",
    ).accepted
    for target in ("VALIDATED", "AUTO_AUTHORIZED", "SCHEDULED"):
        assert h.send(
            dto.ChangeWorkflow(
                kind="PROPOSAL", entity_id="PROP-one", target=target, reason="POLICY"
            ),
            actor="SYSTEM",
        ).accepted
    assert (
        h.send(
            dto.ChangeWorkflow(
                kind="PROPOSAL", entity_id="PROP-one", target="APPLIED", reason="NO_APPLICATION"
            ),
            actor="SYSTEM",
        ).code
        == "PROPOSAL_APPLICATION_REQUIRES_B11"
    )


def test_logical_key_replay_and_conflicting_reuse(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    command = h.command(
        dto.ReserveTask(task_id="TASK-one", attempt_id="ATT-one", worker_id="worker"),
        actor="SCHEDULER",
    )
    first = h.authority.port("SCHEDULER").handle(command)
    assert first.accepted
    from gidr.contracts.common import new_id

    alias = dto.StateCommand.model_validate(
        {**command.model_dump(), "command_id": new_id("CMD"), "trace_id": new_id("TRACE")}
    )
    assert h.authority.port("SCHEDULER").handle(alias) == first
    different = dto.StateCommand.model_validate(
        {
            **alias.model_dump(),
            "command_id": new_id("CMD"),
            "payload": dto.ReserveTask(
                task_id="TASK-one", attempt_id="ATT-other", worker_id="worker"
            ),
        }
    )
    assert h.authority.port("SCHEDULER").handle(different).code == "IDEMPOTENCY_KEY_CONFLICT"
    with h.db.reader() as conn:
        assert conn.execute("SELECT count(*) FROM task_attempts").fetchone()[0] == 1


def test_cooldown_and_heartbeat_require_fresh_current_tokens(h):
    from datetime import timedelta

    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    h.clock.advance(10)
    assert h.send(dto.RenewAttempt(token=token), actor="WORKER").accepted
    gate = h.row("active_run_gate", "singleton", 1)
    assert h.send(dto.RenewRun(generation=gate["generation"]), actor="SYSTEM").accepted
    until = h.clock() + timedelta(seconds=10)
    assert h.finish(token, target="RETRY_WAIT", result_ref=None, not_before_utc=until).accepted
    assert h.ready().code == "TASK_COOLDOWN_ACTIVE"
    h.clock.advance(11)
    assert h.ready().accepted


def test_missing_or_corrupt_result_never_commits_success(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    token = h.running()
    ref = h.store.publish(b"Synthetic corrupt result")
    h.store.path(ref.sha256).write_bytes(b"Corrupted")
    result = h.finish(token, result_ref=ref)
    assert not result.accepted and h.row("tasks", "task_id", "TASK-one")["state"] == "RUNNING"
    assert list(h.store.quarantine.iterdir())


def test_assessment_publication_is_gated_and_conflict_review_is_bounded(h):
    from gidr.contracts.common import Contract
    from gidr.infrastructure.artifacts.repository import ReferenceRequest
    from gidr.infrastructure.audit.events import event_for

    h.setup()
    ref = h.store.publish(b"Synthetic UNASSESSED snapshot, no ESC validation claim")

    def seed(ctx):
        h.repo.register(ctx, ref)
        ctx.execute(
            "INSERT INTO claims VALUES(?,?,?,?,?)",
            ("CLM-fixture", "RUN-one", ref.artifact_id, "DESCRIPTIVE", "{}"),
        )
        ctx.execute(
            "INSERT INTO assessments(assessment_id,claim_id,version,contract_id,contract_version,lifecycle,verdict,snapshot_artifact_id,created_at_utc) VALUES('ASM-fixture','CLM-fixture',1,'synthetic','1','DRAFT','UNASSESSED',?,?)",
            (ref.artifact_id, h.clock().isoformat()),
        )
        h.repo.attach(
            ctx,
            ReferenceRequest(
                reference=ref, owner_type="ASSESSMENT", owner_id="ASM-fixture", role="SNAPSHOT"
            ),
        )
        ctx.emit(
            event_for(
                ctx,
                component="evidence",
                event_type="test.synthetic_unassessed_fixture",
                aggregate_id="ASM-fixture",
                run_id="RUN-one",
            )
        )
        return Contract()

    with h.repo.prepared((ref,)):
        h.storage.execute(
            "CMD-assessment-fixture", "TRACE-assessment-fixture", Contract(), Contract, seed
        )

    def change(target, **kwargs):
        return h.send(
            dto.ChangeWorkflow(
                kind="ASSESSMENT",
                entity_id="ASM-fixture",
                target=target,
                reason="SYNTHETIC_REVIEW",
                **kwargs,
            ),
            actor="SYSTEM",
        )

    assert change("VALIDATING").accepted
    assert change("COMMITTED").code == "ASSESSMENT_ELIGIBILITY_REQUIRES_B10"
    for _ in range(2):
        assert change("CONFLICT_REVIEW").accepted
        assert change("VALIDATING").code == "EXPLICIT_REVIEW_ROUTE_REQUIRED"
        assert change("VALIDATING", condition_change_ref=ref).accepted
    assert change("CONFLICT_REVIEW").accepted
    assert change("VALIDATING", condition_change_ref=ref).code == "VALIDATION_ATTEMPT_LIMIT"
    row = h.row("assessments", "assessment_id", "ASM-fixture")
    assert row["verdict"] == "UNASSESSED" and row["validation_attempts"] == 3


def test_estimated_overrun_retains_larger_reservation_and_halts_dispatch(h):
    h.setup()
    h.add(h.task(max_cost_usd=Decimal(1)))
    assert h.ready().accepted
    token = h.running()
    assert h.finish(token, usage=dto.Usage(state="ESTIMATED", cost_usd=Decimal(3))).accepted
    row = h.row("attempt_accounts", "attempt_id", "ATT-one")
    assert row["status"] == "UNCERTAIN" and row["cost_usd"] == "3"
    assert h.row("runs", "run_id", "RUN-one")["state"] == "PAUSING"


def test_inflight_limit_prevents_unbounded_reservations(h):
    h.setup()
    h.add(*(h.task(str(i)) for i in range(4)))
    for i in range(4):
        assert h.ready("TASK-" + str(i)).accepted
    for i in range(3):
        assert h.reserve("TASK-" + str(i), "ATT-" + str(i)).accepted
    assert h.reserve("TASK-3", "ATT-3").code == "CONCURRENCY_LIMIT"


def test_cas_stale_workflow_and_unauthorized_commands_are_audited(h):
    h.setup()
    assert h.send(
        dto.CreateWorkflow(kind="LEAD", entity_id="LEAD-fixture", reason="SYNTHETIC")
    ).accepted
    payload = dto.ChangeWorkflow(
        kind="LEAD", entity_id="LEAD-fixture", target="TRIAGED", reason="VALIDATE"
    )
    first = h.command(payload, actor="SYSTEM")
    assert h.authority.port("SYSTEM").handle(first).accepted
    assert h.send(payload, actor="SYSTEM", revision=0).code == "CONFLICT_STALE_REVISION"
    wrong = h.command(dto.ChangeRun(target="PAUSING", reason="FAKE_USER"))
    with pytest.raises(Rejected, match="ACTOR_NOT_AUTHORIZED"):
        h.authority.port("WORKER", "worker").handle(wrong)
    with h.db.reader() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE component='security' AND event_type='command.principal_denied'"
            ).fetchone()[0]
            == 1
        )


def test_navigation_cannot_record_a_session_under_a_different_run(h):
    h.setup()
    h.setup("two", start=False)
    assert h.pause().accepted
    assert h.send(dto.OpenSession(session_id="SES-two")).code == "SESSION_RUN_MISMATCH"
    assert h.send(dto.OpenSession(session_id="SES-two"), run="RUN-two").accepted


def test_corrupt_known_proposal_snapshot_updates_catalog_and_audit(h):
    h.setup()
    scope = dto.ProposalScope(
        providers=("local",),
        categories=("local",),
        task_types=("compute",),
        estimated_cost_usd=Decimal(0),
        estimated_tokens=0,
        estimated_requests=0,
    )
    ref = h.store.publish(scope.model_dump_json().encode())
    assert h.send(
        dto.CreateWorkflow(
            kind="PROPOSAL", entity_id="PROP-one", snapshot_ref=ref, reason="SYNTHETIC"
        ),
        actor="SYSTEM",
    ).accepted
    h.store.path(ref.sha256).write_bytes(b"Synthetic corrupted snapshot")
    assert not h.send(
        dto.ChangeWorkflow(
            kind="PROPOSAL", entity_id="PROP-one", target="VALIDATED", reason="VALIDATE"
        ),
        actor="SYSTEM",
    ).accepted
    assert h.row("artifact_catalog", "artifact_id", ref.artifact_id)["status"] == "QUARANTINED"
    assert h.row("critic_proposals", "proposal_id", "PROP-one")["status"] == "PROPOSED"
    with h.db.reader() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE event_type='artifact.integrity_failed'"
            ).fetchone()[0]
            == 1
        )


def test_completion_and_direct_success_skip_are_rejected(h):
    h.setup()
    h.add(h.task())
    assert h.ready().accepted
    assert (
        h.send(
            dto.ChangeTask(task_id="TASK-one", target="SUCCEEDED", reason="SKIP"), actor="SYSTEM"
        ).code
        == "SPECIALIZED_COMMAND_REQUIRED"
    )
    assert (
        h.send(dto.ChangeRun(target="COMPLETING", reason="FAKE_DONE"), actor="SYSTEM").code
        == "UNRESOLVED_TASKS_REQUIRE_PARTIAL_PAUSE"
    )
