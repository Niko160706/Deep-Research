"""Durable model-call authority and atomic task-result publication."""

from __future__ import annotations

import hashlib
import secrets
import sqlite3
from collections.abc import Callable
from datetime import timedelta
from decimal import Decimal
from sqlite3 import Row
from typing import cast
from urllib.parse import urlsplit

from gidr.application import accounting
from gidr.application.commands import StateAuthority, stamp, time_value
from gidr.contracts import commands as state_dto
from gidr.contracts.common import (
    ArtifactRef,
    AttemptId,
    Contract,
    GrantId,
    Language,
    ModelExecutionId,
    Name,
    RunId,
    TaskId,
    new_id,
)
from gidr.contracts.control import AuditEvent
from gidr.contracts.models import (
    FinancialCharge,
    ModelErrorCode,
    ModelExecutionResult,
    ModelTask,
    TokenUsage,
    TypedDecision,
)
from gidr.domain.transitions import allowed
from gidr.infrastructure.artifacts.repository import ArtifactRepository, ReferenceRequest
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.db.authority import PersistenceAuthority, WriteContext
from gidr.infrastructure.models.ports import (
    ModelAttemptDenied,
    ModelCall,
    ModelDispatchPermit,
    ModelHTTPDispatch,
    ProviderProfile,
    ReservedModelAttempt,
)


class _ModelCommand(Contract):
    action: Name
    run_id: RunId | None = None
    task_id: TaskId | None = None
    worker_attempt_id: AttemptId | None = None
    execution_id: ModelExecutionId | None = None
    provider_id: Name | None = None
    model_attempt_id: AttemptId | None = None
    reason_code: Name
    grant_id: GrantId | None = None
    input_refs: tuple[ArtifactRef, ...] = ()
    output_ref: ArtifactRef | None = None
    old_state: Name | None = None
    new_state: Name | None = None
    revision_before: int | None = None


class _ModelReceipt(Contract):
    accepted: bool
    code: Name
    model_attempt_id: AttemptId | None = None
    estimated_cost_usd: Decimal | None = None


class _Rejected(Exception):
    def __init__(self, code: str):
        self.code = code


_MODEL_ENDPOINTS = {
    "jev": ("api.typesafe.ai", "/v1/systemone"),
    "gemini": ("generativelanguage.googleapis.com", "/v1beta/openai/chat/completions"),
    "groq": ("api.groq.com", "/openai/v1/chat/completions"),
    "siliconflow": ("api.siliconflow.cn", "/v1/chat/completions"),
}


class StateAuthorityModelBridge:
    """Implements model ports through the active StateAuthority and SQLite ledger."""

    def __init__(self, authority: StateAuthority, dispatch: ModelHTTPDispatch | None = None):
        self._state = authority
        self._storage: PersistenceAuthority = authority.storage
        self._artifacts: ArtifactRepository = authority.artifacts
        self._dispatch = dispatch

    def bind_dispatch(self, dispatch: ModelHTTPDispatch) -> None:
        if self._dispatch is not None:
            raise ValueError("MODEL_DISPATCH_ALREADY_BOUND")
        self._dispatch = dispatch

    async def reserve(
        self,
        task: ModelTask,
        call: ModelCall,
        execution_id: str,
        profile: ProviderProfile,
        attempt_number: int,
    ) -> ReservedModelAttempt:
        permit_key = secrets.token_urlsafe(32)
        model_attempt_id = new_id("ATT")
        call_bytes = call.model_dump_json().encode("utf-8")
        request_hash = hashlib.sha256(call_bytes).hexdigest()
        receipt = self._reserve_with_inputs(
            task,
            call,
            execution_id,
            profile,
            attempt_number,
            model_attempt_id,
            permit_key,
            request_hash,
        )
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)
        if self._dispatch is None or receipt.model_attempt_id is None:
            raise ModelAttemptDenied("MODEL_DISPATCH_NOT_CONFIGURED")
        permit = ModelDispatchPermit(
            attempt_id=receipt.model_attempt_id,
            run_id=task.run_id,
            task_id=task.task_id,
            execution_id=execution_id,
            provider_id=profile.provider_id,
            reservation_id=task.reservation_id,
            dispatch=self._dispatch,
            opaque_token=permit_key,
        )
        return ReservedModelAttempt(permit=permit, estimated_cost_usd=receipt.estimated_cost_usd)

    def _reserve_with_inputs(
        self,
        task: ModelTask,
        call: ModelCall,
        execution_id: str,
        profile: ProviderProfile,
        attempt_number: int,
        model_attempt_id: str,
        permit_key: str,
        request_hash: str,
    ) -> _ModelReceipt:
        with self._artifacts.prepared(task.input_refs):
            return self._reserve(
                task,
                call,
                execution_id,
                profile,
                attempt_number,
                model_attempt_id,
                permit_key,
                request_hash,
            )

    def _reserve(
        self,
        task: ModelTask,
        call: ModelCall,
        execution_id: str,
        profile: ProviderProfile,
        attempt_number: int,
        model_attempt_id: str,
        permit_key: str,
        request_hash: str,
    ) -> _ModelReceipt:
        return self._transact(
            task,
            "MODEL_ATTEMPT_RESERVED",
            "provider.model_attempt_reserved",
            lambda ctx: self._reserve_in_context(
                ctx,
                task,
                call,
                execution_id,
                profile,
                attempt_number,
                model_attempt_id,
                permit_key,
                request_hash,
            ),
        )

    def _reserve_in_context(
        self,
        ctx: WriteContext,
        task: ModelTask,
        call: ModelCall,
        execution_id: str,
        profile: ProviderProfile,
        attempt_number: int,
        model_attempt_id: str,
        permit_key: str,
        request_hash: str,
    ) -> _ModelReceipt:
        try:
            rows = self._authorized_rows(ctx, task)
            _run, db_task, _worker_attempt, account, grant = rows
            reject(profile.provider_id in grant.providers, "PROVIDER_OUT_OF_SCOPE")
            spec = state_dto.ScheduledTask.model_validate_json(db_task["policy_json"])
            reject(spec.task_type == task.task_type, "MODEL_TASK_TYPE_MISMATCH")
            reject(profile.provider_id == spec.provider_id, "TASK_PROVIDER_MISMATCH")
            reject(spec.category in grant.categories, "CATEGORY_OUT_OF_SCOPE")
            reject(spec.task_type in grant.task_types, "TASK_TYPE_OUT_OF_SCOPE")
            reject(
                spec.category == "model" and spec.work_kind == "PROVIDER",
                "MODEL_TASK_SCOPE_INVALID",
            )
            reject(profile.supports(task, call), "MODEL_PROFILE_CAPABILITY_MISMATCH")
            reject(call.input_language == task.input_language, "MODEL_LANGUAGE_CONTEXT_MISMATCH")
            authorized_inputs = ctx.execute(
                "SELECT ac.artifact_id,ac.sha256,ac.size_bytes FROM artifact_refs ar JOIN artifact_catalog ac USING(artifact_id) WHERE ar.owner_type='RUN' AND ar.owner_id=? AND ar.role=? AND ac.status='AVAILABLE'",
                (task.run_id, "TASK_INPUT_" + task.task_id),
            ).fetchall()
            expected_inputs = {
                (row["artifact_id"], row["sha256"], row["size_bytes"]) for row in authorized_inputs
            }
            supplied_inputs = {
                (ref.artifact_id, ref.sha256, ref.size_bytes) for ref in task.input_refs
            }
            reject(
                bool(expected_inputs) and supplied_inputs == expected_inputs,
                "MODEL_INPUT_SCOPE_MISMATCH",
            )
            allowed_attempts = min(
                task.attempt_limit, spec.max_attempts, grant.max_attempts, account["max_requests"]
            )
            prior = ctx.execute(
                "SELECT count(*) FROM model_attempts WHERE model_execution_id=?", (execution_id,)
            ).fetchone()[0]
            reject(attempt_number == prior + 1 and prior < allowed_attempts, "MODEL_ATTEMPT_LIMIT")
            existing = ctx.execute(
                "SELECT parent_attempt_id,task_id FROM model_executions WHERE model_execution_id=?",
                (execution_id,),
            ).fetchone()
            if existing is None:
                prior_execution = ctx.execute(
                    "SELECT model_execution_id FROM model_executions WHERE parent_attempt_id=?",
                    (task.reservation_id,),
                ).fetchone()
                reject(prior_execution is None, "MODEL_EXECUTION_ALREADY_RESERVED")
                ctx.execute(
                    "INSERT INTO model_executions(model_execution_id,task_id,provider_id,model_id,model_version,prompt_version,input_artifact_id,output_artifact_id,usage_state,cost_state,cost_usd,error_code,parent_attempt_id,result_json) VALUES(?,?,?,?,?,?,?,NULL,'UNKNOWN_USAGE','UNKNOWN',NULL,NULL,?,NULL)",
                    (
                        execution_id,
                        task.task_id,
                        profile.provider_id,
                        profile.model_id,
                        None,
                        task.prompt_version,
                        task.input_refs[0].artifact_id,
                        task.reservation_id,
                    ),
                )
                for reference in task.input_refs:
                    self._artifacts.attach(
                        ctx,
                        ReferenceRequest(
                            reference=reference,
                            owner_type="RUN",
                            owner_id=task.run_id,
                            role="MODEL_INPUT_" + execution_id,
                        ),
                    )
            else:
                reject(
                    existing["parent_attempt_id"] == task.reservation_id
                    and existing["task_id"] == task.task_id,
                    "MODEL_EXECUTION_LINEAGE_MISMATCH",
                )

            reserved_cost = Decimal(account["max_cost_usd"]) / Decimal(allowed_attempts)
            serialized_size = len(call.model_dump_json().encode("utf-8"))
            token_upper_bound = serialized_size + call.max_output_tokens
            prior_budget = ctx.execute(
                "SELECT reserved_cost_usd,reserved_tokens,reserved_requests FROM model_attempt_controls WHERE parent_attempt_id=?",
                (task.reservation_id,),
            ).fetchall()
            used_tokens = sum(int(row["reserved_tokens"]) for row in prior_budget)
            used_requests = sum(int(row["reserved_requests"]) for row in prior_budget)
            used_cost = sum((Decimal(row["reserved_cost_usd"]) for row in prior_budget), Decimal(0))
            reject(
                used_tokens + token_upper_bound <= int(account["max_tokens"]),
                "MODEL_TOKEN_RESERVATION_EXCEEDED",
            )
            reject(used_requests + 1 <= int(account["max_requests"]), "MODEL_REQUEST_LIMIT")
            reject(
                used_cost + reserved_cost <= Decimal(account["max_cost_usd"]),
                "MODEL_COST_RESERVATION_EXCEEDED",
            )
            if (
                profile.estimated_input_usd_per_million is not None
                and profile.estimated_output_usd_per_million is not None
            ):
                estimate = (
                    Decimal(serialized_size) * profile.estimated_input_usd_per_million
                    + Decimal(call.max_output_tokens) * profile.estimated_output_usd_per_million
                ) / Decimal(1_000_000)
                reject(estimate <= reserved_cost, "MODEL_ESTIMATE_EXCEEDS_RESERVED_COST")

            if (
                ctx.execute(
                    "SELECT 1 FROM provider_health WHERE adapter_id=?", (profile.provider_id,)
                ).fetchone()
                is None
            ):
                ctx.execute(
                    "INSERT INTO provider_health(adapter_id,adapter_version,status,reason_code) VALUES(?,?,'MOCK_TESTED','LIVE_SMOKE_NOT_VERIFIED')",
                    (profile.provider_id, "gidr-model-v1"),
                )
            now = stamp(self._state.clock())
            ctx.execute(
                "INSERT INTO model_attempts(model_attempt_id,model_execution_id,attempt_no,idempotency_key,output_artifact_id,error_code) VALUES(?,?,?,?,NULL,NULL)",
                (
                    model_attempt_id,
                    execution_id,
                    attempt_number,
                    execution_id + ":" + str(attempt_number),
                ),
            )
            ctx.execute(
                "INSERT INTO model_attempt_controls(model_attempt_id,parent_attempt_id,provider_id,state,permit_sha256,input_sha256,reserved_cost_usd,reserved_tokens,reserved_requests,usage_state,created_at_utc,updated_at_utc) VALUES(?,?,?,'RESERVED',?,?,?,?,1,'UNKNOWN_USAGE',?,?)",
                (
                    model_attempt_id,
                    task.reservation_id,
                    profile.provider_id,
                    hashlib.sha256(permit_key.encode()).hexdigest(),
                    request_hash,
                    str(reserved_cost),
                    token_upper_bound,
                    now,
                    now,
                ),
            )
            return _ModelReceipt(
                accepted=True,
                code="RESERVED",
                model_attempt_id=model_attempt_id,
                estimated_cost_usd=reserved_cost,
            )
        except _Rejected as exc:
            return _ModelReceipt(accepted=False, code=exc.code)
        except Exception:
            return _ModelReceipt(accepted=False, code="MODEL_RESERVATION_REJECTED")

    def _authorized_rows(
        self, ctx: WriteContext, task: ModelTask
    ) -> tuple[sqlite3.Row, sqlite3.Row, sqlite3.Row, sqlite3.Row, state_dto.GrantPolicy]:
        run = ctx.execute(
            "SELECT r.*,p.version AS active_version FROM runs r JOIN plan_versions p ON p.plan_version_id=r.active_plan_version_id WHERE r.run_id=?",
            (task.run_id,),
        ).fetchone()
        reject(run is not None and run["state"] == "RUNNING", "RUN_NOT_RUNNING")
        gate = ctx.execute("SELECT * FROM active_run_gate WHERE singleton=1").fetchone()
        reject(
            gate is not None
            and gate["run_id"] == task.run_id
            and gate["holder"] == self._state.nonce
            and time_value(gate["expires_at_utc"]) > self._state.clock(),
            "RUN_GATE_NOT_OWNED_OR_EXPIRED",
        )
        reject(task.plan_version == run["active_version"], "STALE_PLAN_VERSION")
        db_task = ctx.execute(
            "SELECT * FROM tasks WHERE task_id=? AND run_id=?", (task.task_id, task.run_id)
        ).fetchone()
        reject(
            db_task is not None
            and db_task["state"] == "RUNNING"
            and db_task["plan_version_id"] == run["active_plan_version_id"],
            "MODEL_TASK_NOT_RUNNING",
        )
        worker_attempt = ctx.execute(
            "SELECT * FROM task_attempts WHERE attempt_id=? AND task_id=?",
            (task.reservation_id, task.task_id),
        ).fetchone()
        reject(
            worker_attempt is not None
            and worker_attempt["state"] == "RUNNING"
            and worker_attempt["gate_generation"] == gate["generation"]
            and worker_attempt["plan_version_id"] == run["active_plan_version_id"]
            and worker_attempt["grant_id"] == task.authorization_ref
            and time_value(worker_attempt["expires_at_utc"]) > self._state.clock(),
            "WORKER_ATTEMPT_INVALID",
        )
        account = ctx.execute(
            "SELECT * FROM attempt_accounts WHERE attempt_id=? AND status='HELD'",
            (task.reservation_id,),
        ).fetchone()
        reject(account is not None, "MODEL_BUDGET_NOT_HELD")
        grant_row = ctx.execute(
            "SELECT * FROM grants WHERE grant_id=? AND run_id=?",
            (task.authorization_ref, task.run_id),
        ).fetchone()
        reject(
            grant_row is not None
            and not grant_row["revoked"]
            and grant_row["plan_version_id"] == run["active_plan_version_id"],
            "MODEL_GRANT_INVALID",
        )
        grant = state_dto.GrantPolicy.model_validate_json(grant_row["scope_json"])
        reject(
            (grant.expires_at_utc is None or grant.expires_at_utc > self._state.clock())
            and (
                grant_row["expires_at_utc"] is None
                or time_value(grant_row["expires_at_utc"]) > self._state.clock()
            ),
            "MODEL_GRANT_EXPIRED",
        )
        return run, db_task, worker_attempt, account, grant

    async def settle_failure(
        self,
        task: ModelTask,
        permit: ModelDispatchPermit,
        tokens: TokenUsage,
        charge: FinancialCharge,
        error_code: ModelErrorCode,
    ) -> None:
        receipt = self._settle_failure(task, permit, tokens, charge, error_code)
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)

    def _settle_failure(
        self,
        task: ModelTask,
        permit: ModelDispatchPermit,
        tokens: TokenUsage,
        charge: FinancialCharge,
        error_code: ModelErrorCode,
    ) -> _ModelReceipt:
        def operation(ctx: WriteContext) -> _ModelReceipt:
            try:
                self._authorized_rows(ctx, task)
                row = self._permit_row(ctx, task, permit)
                reject(row["state"] in {"RESERVED", "DISPATCHED"}, "MODEL_ATTEMPT_ALREADY_SETTLED")
                token_value = tokens.input_tokens if tokens.state != "UNKNOWN_USAGE" else None
                output_value = tokens.output_tokens if tokens.state != "UNKNOWN_USAGE" else None
                cost_value = charge.usd if charge.state != "UNKNOWN" else None
                ctx.execute(
                    "UPDATE model_attempt_controls SET state='FAILED',input_tokens=?,output_tokens=?,actual_cost_usd=?,usage_state=?,error_code=?,updated_at_utc=? WHERE model_attempt_id=?",
                    (
                        token_value,
                        output_value,
                        None if cost_value is None else str(cost_value),
                        "UNKNOWN_USAGE" if tokens.state == "UNKNOWN_USAGE" else tokens.state,
                        error_code,
                        stamp(self._state.clock()),
                        permit.attempt_id,
                    ),
                )
                ctx.execute(
                    "UPDATE model_attempts SET error_code=? WHERE model_attempt_id=?",
                    (error_code, permit.attempt_id),
                )
                return _ModelReceipt(
                    accepted=True, code="SETTLED", model_attempt_id=permit.attempt_id
                )
            except _Rejected as exc:
                return _ModelReceipt(accepted=False, code=exc.code)
            except Exception:
                return _ModelReceipt(accepted=False, code="MODEL_SETTLEMENT_REJECTED")

        return self._transact(
            task, "MODEL_ATTEMPT_SETTLED", "provider.model_attempt_settled", operation
        )

    async def record_failure(self, task: ModelTask, result: ModelExecutionResult) -> None:
        receipt = self._record_failure(task, result)
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)

    def _record_failure(self, task: ModelTask, result: ModelExecutionResult) -> _ModelReceipt:
        def operation(ctx: WriteContext) -> _ModelReceipt:
            try:
                self._authorized_rows(ctx, task)
                row = ctx.execute(
                    "SELECT parent_attempt_id FROM model_executions WHERE model_execution_id=? AND task_id=?",
                    (result.model_execution_id, task.task_id),
                ).fetchone()
                reject(
                    row is not None and row["parent_attempt_id"] == task.reservation_id,
                    "MODEL_EXECUTION_NOT_FOUND",
                )
                ctx.execute(
                    "UPDATE model_executions SET provider_id=?,model_id=?,model_version=?,usage_state=?,cost_state=?,cost_usd=?,error_code=?,result_json=? WHERE model_execution_id=?",
                    (
                        result.provider_id,
                        result.model_id,
                        result.model_version,
                        result.tokens.state,
                        result.charge.state,
                        None if result.charge.usd is None else str(result.charge.usd),
                        result.error_code,
                        result.model_dump_json(),
                        result.model_execution_id,
                    ),
                )
                return _ModelReceipt(accepted=True, code="FAILURE_RECORDED")
            except _Rejected as exc:
                return _ModelReceipt(accepted=False, code=exc.code)
            except Exception:
                return _ModelReceipt(accepted=False, code="MODEL_FAILURE_AUDIT_REJECTED")

        request = _ModelCommand(
            action="MODEL_EXECUTION_FAILED",
            run_id=task.run_id,
            task_id=task.task_id,
            worker_attempt_id=task.reservation_id,
            grant_id=task.authorization_ref,
            execution_id=result.model_execution_id,
            reason_code=result.error_code or "MODEL_FAILED",
        )
        return self._commit(
            request, operation, task.run_id, "provider.model_execution_failed", task.task_id
        )

    async def cooldown_remaining(self, provider_id: str) -> float:
        # This bounded indexed read is deliberately synchronous.  Keeping the
        # StateAuthority lifetime guard and SQLite reader in the caller avoids
        # executor/lifecycle races during shutdown and is sub-millisecond locally.
        return self._cooldown_remaining(provider_id)

    def _cooldown_remaining(self, provider_id: str) -> float:
        with self._state.model_operation(), self._storage.database.reader() as conn:
            row = conn.execute(
                "SELECT until_utc FROM provider_cooldowns WHERE adapter_id=?", (provider_id,)
            ).fetchone()
            if row is None:
                value = 0.0
            else:
                value = max(0.0, (time_value(row[0]) - self._state.clock()).total_seconds())
        return value

    async def impose_cooldown(self, provider_id: str, seconds: float) -> None:
        seconds = max(0.0, min(120.0, float(seconds)))
        self._impose_cooldown(provider_id, seconds)

    def _impose_cooldown(self, provider_id: str, seconds: float) -> None:
        request = _ModelCommand(
            action="PROVIDER_COOLDOWN", provider_id=provider_id, reason_code="RATE_LIMIT"
        )

        def operation(ctx: WriteContext) -> _ModelReceipt:
            if (
                ctx.execute(
                    "SELECT 1 FROM provider_health WHERE adapter_id=?", (provider_id,)
                ).fetchone()
                is None
            ):
                ctx.execute(
                    "INSERT INTO provider_health(adapter_id,adapter_version,status,reason_code) VALUES(?,?,'MOCK_TESTED','LIVE_SMOKE_NOT_VERIFIED')",
                    (provider_id, "gidr-model-v1"),
                )
            until = stamp(self._state.clock() + timedelta(seconds=seconds))
            ctx.execute(
                "INSERT INTO provider_cooldowns(adapter_id,until_utc,reason_code) VALUES(?,?,?) ON CONFLICT(adapter_id) DO UPDATE SET until_utc=excluded.until_utc,reason_code=excluded.reason_code",
                (provider_id, until, "RATE_LIMIT"),
            )
            return _ModelReceipt(accepted=True, code="COOLDOWN_RECORDED")

        self._commit(request, operation, None, "provider.cooldown_recorded", provider_id)

    async def begin_dispatch(
        self, permit: ModelDispatchPermit, url: str, method: str, body: bytes
    ) -> None:
        receipt = self._begin_dispatch(permit, url, method, body)
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)

    def _begin_dispatch(
        self, permit: ModelDispatchPermit, url: str, method: str, body: bytes
    ) -> _ModelReceipt:
        task = self._load_task(permit)
        request = self._request(task, "MODEL_DISPATCH_STARTED", permit, "MODEL_DISPATCH_STARTED")

        def operation(ctx: WriteContext) -> _ModelReceipt:
            try:
                self._authorized_rows(ctx, task)
                row = self._permit_row(ctx, task, permit)
                endpoint = _MODEL_ENDPOINTS.get(permit.provider_id)
                parsed = urlsplit(url)
                reject(
                    row["state"] == "RESERVED"
                    and method == "POST"
                    and endpoint is not None
                    and parsed.scheme == "https"
                    and parsed.hostname == endpoint[0]
                    and parsed.port in {None, 443}
                    and parsed.path == endpoint[1]
                    and not parsed.query
                    and not parsed.fragment
                    and not parsed.username
                    and not parsed.password,
                    "MODEL_PERMIT_NOT_DISPATCHABLE",
                )
                ctx.execute(
                    "UPDATE model_attempt_controls SET state='DISPATCHED',dispatch_url_sha256=?,dispatch_body_sha256=?,updated_at_utc=? WHERE model_attempt_id=?",
                    (
                        hashlib.sha256(url.encode()).hexdigest(),
                        hashlib.sha256(body).hexdigest(),
                        stamp(self._state.clock()),
                        permit.attempt_id,
                    ),
                )
                return _ModelReceipt(
                    accepted=True, code="DISPATCH_STARTED", model_attempt_id=permit.attempt_id
                )
            except _Rejected as exc:
                return _ModelReceipt(accepted=False, code=exc.code)
            except Exception:
                return _ModelReceipt(accepted=False, code="MODEL_DISPATCH_REJECTED")

        receipt = self._transact(
            task,
            "MODEL_DISPATCH_STARTED",
            "provider.model_dispatch_started",
            operation,
            request=request,
        )
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)
        return receipt

    async def finish_dispatch(self, permit: ModelDispatchPermit, status: int, body: bytes) -> None:
        receipt = self._finish_dispatch(permit, status, body)
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)

    def _finish_dispatch(
        self, permit: ModelDispatchPermit, status: int, body: bytes
    ) -> _ModelReceipt:
        task = self._load_task(permit)
        request = self._request(task, "MODEL_DISPATCH_FINISHED", permit, "MODEL_DISPATCH_FINISHED")

        def operation(ctx: WriteContext) -> _ModelReceipt:
            try:
                self._authorized_rows(ctx, task)
                row = self._permit_row(ctx, task, permit)
                reject(row["state"] == "DISPATCHED", "MODEL_DISPATCH_NOT_STARTED")
                ctx.execute(
                    "UPDATE model_attempt_controls SET response_status=?,response_sha256=?,updated_at_utc=? WHERE model_attempt_id=?",
                    (
                        status,
                        hashlib.sha256(body).hexdigest(),
                        stamp(self._state.clock()),
                        permit.attempt_id,
                    ),
                )
                return _ModelReceipt(
                    accepted=True, code="DISPATCH_FINISHED", model_attempt_id=permit.attempt_id
                )
            except _Rejected as exc:
                return _ModelReceipt(accepted=False, code=exc.code)
            except Exception:
                return _ModelReceipt(accepted=False, code="MODEL_DISPATCH_FINISH_REJECTED")

        receipt = self._transact(
            task,
            "MODEL_DISPATCH_FINISHED",
            "provider.model_dispatch_finished",
            operation,
            request=request,
        )
        if not receipt.accepted:
            raise ModelAttemptDenied(receipt.code)
        return receipt

    def _load_task(self, permit: ModelDispatchPermit) -> ModelTask:
        with self._state.model_operation(), self._storage.database.reader() as conn:
            execution = conn.execute(
                "SELECT parent_attempt_id FROM model_executions WHERE model_execution_id=? AND task_id=?",
                (permit.execution_id, permit.task_id),
            ).fetchone()
            reject(execution is not None, "MODEL_EXECUTION_NOT_FOUND")
            parent_attempt = str(execution["parent_attempt_id"])
            task_row = conn.execute(
                "SELECT * FROM tasks WHERE task_id=?", (permit.task_id,)
            ).fetchone()
            run_row = conn.execute("SELECT * FROM runs WHERE run_id=?", (permit.run_id,)).fetchone()
            grant_row = conn.execute(
                "SELECT grant_id FROM task_attempts WHERE attempt_id=?", (parent_attempt,)
            ).fetchone()
            reject(
                task_row is not None and run_row is not None and grant_row is not None,
                "MODEL_CONTEXT_NOT_FOUND",
            )
            model_row = conn.execute(
                "SELECT * FROM model_executions WHERE model_execution_id=?", (permit.execution_id,)
            ).fetchone()
            policy = state_dto.ScheduledTask.model_validate_json(task_row["policy_json"])
            refs = conn.execute(
                "SELECT ac.* FROM artifact_refs ar JOIN artifact_catalog ac USING(artifact_id) WHERE ar.owner_type='RUN' AND ar.owner_id=? AND ar.role=? AND ac.status='AVAILABLE' ORDER BY ac.artifact_id",
                (permit.run_id, "MODEL_INPUT_" + permit.execution_id),
            ).fetchall()
            require_refs = tuple(
                ArtifactRef(
                    artifact_id=row["artifact_id"],
                    sha256=row["sha256"],
                    size_bytes=row["size_bytes"],
                )
                for row in refs
            )
            reject(bool(require_refs) and model_row is not None, "MODEL_INPUT_PROVENANCE_MISSING")
            plan_version = conn.execute(
                "SELECT version FROM plan_versions WHERE plan_version_id=?",
                (task_row["plan_version_id"],),
            ).fetchone()[0]
        return ModelTask(
            run_id=permit.run_id,
            created_at_utc=self._state.clock(),
            task_id=permit.task_id,
            plan_version=plan_version,
            authorization_ref=grant_row[0],
            task_type=task_row["type"],
            required_capabilities=("model",),
            input_refs=require_refs,
            input_language=Language(code="unknown", status="UNKNOWN"),
            risk_class="LOW",
            reservation_id=parent_attempt,
            attempt_limit=policy.max_attempts,
            preferred_provider=model_row["provider_id"],
            prompt_version=model_row["prompt_version"],
            privacy_scope="PUBLIC_SOURCE",
        )

    def _permit_row(
        self, ctx: WriteContext, task: ModelTask, permit: ModelDispatchPermit
    ) -> sqlite3.Row:
        reject(
            permit.task_id == task.task_id
            and permit.run_id == task.run_id
            and permit.reservation_id == task.reservation_id,
            "MODEL_PERMIT_CONTEXT_MISMATCH",
        )
        row = ctx.execute(
            "SELECT c.* FROM model_attempt_controls c JOIN model_attempts a USING(model_attempt_id) WHERE c.model_attempt_id=? AND a.model_execution_id=?",
            (permit.attempt_id, permit.execution_id),
        ).fetchone()
        reject(
            row is not None
            and row["provider_id"] == permit.provider_id
            and secrets.compare_digest(
                row["permit_sha256"], hashlib.sha256(str(permit.opaque_token).encode()).hexdigest()
            ),
            "MODEL_PERMIT_INVALID",
        )
        assert row is not None
        return cast(Row, row)

    def _request(
        self, task: ModelTask, action: str, permit: ModelDispatchPermit, reason: str
    ) -> _ModelCommand:
        return _ModelCommand(
            action=action,
            run_id=task.run_id,
            task_id=task.task_id,
            worker_attempt_id=task.reservation_id,
            grant_id=task.authorization_ref,
            execution_id=permit.execution_id,
            provider_id=permit.provider_id,
            model_attempt_id=permit.attempt_id,
            reason_code=reason,
        )

    def _transact(
        self,
        task: ModelTask,
        action: str,
        event_type: str,
        operation: Callable[[WriteContext], _ModelReceipt],
        *,
        request: _ModelCommand | None = None,
    ) -> _ModelReceipt:
        request = request or _ModelCommand(
            action=action,
            run_id=task.run_id,
            task_id=task.task_id,
            worker_attempt_id=task.reservation_id,
            reason_code=action,
        )
        return self._commit(request, operation, task.run_id, event_type, task.task_id)

    def _commit(
        self,
        request: _ModelCommand,
        operation: Callable[[WriteContext], _ModelReceipt],
        run_id: str | None,
        event_type: str,
        aggregate_id: str,
    ) -> _ModelReceipt:
        def commit(ctx: WriteContext) -> _ModelReceipt:
            receipt = operation(ctx)
            event = event_for(
                ctx,
                component="model_gateway",
                event_type=event_type,
                aggregate_id=aggregate_id,
                run_id=run_id,
            )
            revision_after = (
                None if request.revision_before is None else request.revision_before + 1
            )
            ctx.emit(
                AuditEvent.model_validate(
                    {
                        **event.model_dump(),
                        "actor": "WORKER" if run_id is not None else "SYSTEM",
                        "aggregate_type": "TASK" if request.task_id is not None else "PROVIDER",
                        "task_id": request.task_id,
                        "attempt_id": request.worker_attempt_id,
                        "grant_id": request.grant_id,
                        "input_refs": request.input_refs,
                        "output_refs": (
                            (request.output_ref,)
                            if receipt.accepted and request.output_ref is not None
                            else ()
                        ),
                        "old_state": request.old_state,
                        "new_state": request.new_state,
                        "revision_before": request.revision_before,
                        "revision_after": revision_after,
                        "cause": request.action,
                        "reason_code": receipt.code,
                        "security": None,
                    }
                )
            )
            return receipt

        with self._state.model_operation():
            return self._storage.execute(
                new_id("CMD"), new_id("TRACE"), request, _ModelReceipt, commit
            )

    def _task_revision(self, task_id: str) -> int:
        with self._state.model_operation(), self._storage.database.reader() as conn:
            row = conn.execute("SELECT revision FROM tasks WHERE task_id=?", (task_id,)).fetchone()
            reject(row is not None, "MODEL_TASK_NOT_FOUND")
            return int(row[0])

    async def publish(
        self,
        task: ModelTask,
        execution_id: str,
        output: bytes,
        media_type: str,
        *,
        permit: ModelDispatchPermit,
        attempt_tokens: TokenUsage,
        attempt_charge: FinancialCharge,
        provider_id: str,
        model_id: str,
        model_version: str | None,
        prompt_version: str,
        decision: TypedDecision | None,
        tokens: TokenUsage,
        charge: FinancialCharge,
        attempt_ids: tuple[str, ...],
    ) -> ArtifactRef:
        return self._publish(
            task,
            execution_id,
            output,
            media_type,
            permit,
            attempt_tokens,
            attempt_charge,
            provider_id,
            model_id,
            model_version,
            prompt_version,
            decision,
            tokens,
            charge,
            attempt_ids,
        )

    def _publish(
        self,
        task: ModelTask,
        execution_id: str,
        output: bytes,
        media_type: str,
        permit: ModelDispatchPermit,
        attempt_tokens: TokenUsage,
        attempt_charge: FinancialCharge,
        provider_id: str,
        model_id: str,
        model_version: str | None,
        prompt_version: str,
        decision: TypedDecision | None,
        tokens: TokenUsage,
        charge: FinancialCharge,
        attempt_ids: tuple[str, ...],
    ) -> ArtifactRef:
        if media_type != "application/json":
            raise ValueError("MODEL_OUTPUT_MEDIA_TYPE_DENIED")
        output_ref = self._artifacts.store.publish(output)
        with self._artifacts.prepared((output_ref, *task.input_refs)):
            result = ModelExecutionResult(
                model_execution_id=execution_id,
                run_id=task.run_id,
                created_at_utc=self._state.clock(),
                task_id=task.task_id,
                attempt_ids=attempt_ids,
                provider_id=provider_id,
                model_id=model_id,
                model_version=model_version,
                prompt_version=prompt_version,
                status="SUCCESS",
                output_ref=output_ref,
                decision=decision,
                tokens=tokens,
                charge=charge,
                fallback_used=provider_id != task.preferred_provider,
            )
            request = self._request(
                task, "MODEL_OUTPUT_PUBLISHED", permit, "MODEL_OUTPUT_PUBLISHED"
            ).model_copy(
                update={
                    "input_refs": task.input_refs,
                    "output_ref": output_ref,
                    "old_state": "RUNNING",
                    "new_state": "SUCCEEDED",
                    "revision_before": self._task_revision(task.task_id),
                }
            )

            def operation(ctx: WriteContext) -> _ModelReceipt:
                try:
                    self._authorized_rows(ctx, task)
                    selected = self._permit_row(ctx, task, permit)
                    reject(
                        selected["state"] == "DISPATCHED"
                        and selected["provider_id"] == provider_id,
                        "MODEL_OUTPUT_ATTEMPT_INVALID",
                    )
                    rows = ctx.execute(
                        "SELECT model_attempt_id,state FROM model_attempt_controls WHERE model_attempt_id IN (SELECT model_attempt_id FROM model_attempts WHERE model_execution_id=?)",
                        (execution_id,),
                    ).fetchall()
                    reject(
                        len(rows) == len(attempt_ids)
                        and {r["model_attempt_id"] for r in rows} == set(attempt_ids),
                        "MODEL_ATTEMPT_LINEAGE_INVALID",
                    )
                    reject(
                        all(r["state"] in {"FAILED", "DISPATCHED"} for r in rows),
                        "MODEL_ATTEMPT_STATE_INVALID",
                    )
                    execution = ctx.execute(
                        "SELECT parent_attempt_id FROM model_executions WHERE model_execution_id=? AND task_id=?",
                        (execution_id, task.task_id),
                    ).fetchone()
                    reject(
                        execution is not None
                        and execution["parent_attempt_id"] == task.reservation_id,
                        "MODEL_EXECUTION_LINEAGE_INVALID",
                    )
                    # Register before updating the immediate foreign key; attach()
                    # below remains in this same transaction.
                    self._artifacts.register(ctx, output_ref)
                    ctx.execute(
                        "UPDATE model_executions SET provider_id=?,model_id=?,model_version=?,output_artifact_id=?,usage_state=?,cost_state=?,cost_usd=?,error_code=NULL,result_json=? WHERE model_execution_id=?",
                        (
                            provider_id,
                            model_id,
                            model_version,
                            output_ref.artifact_id,
                            tokens.state,
                            charge.state,
                            None if charge.usd is None else str(charge.usd),
                            result.model_dump_json(),
                            execution_id,
                        ),
                    )
                    for reference in task.input_refs:
                        self._artifacts.attach(
                            ctx,
                            ReferenceRequest(
                                reference=reference,
                                owner_type="MODEL_EXECUTION",
                                owner_id=execution_id,
                                role="INPUT",
                            ),
                        )
                    self._artifacts.attach(
                        ctx,
                        ReferenceRequest(
                            reference=output_ref,
                            owner_type="MODEL_EXECUTION",
                            owner_id=execution_id,
                            role="OUTPUT",
                        ),
                    )
                    ctx.execute(
                        "UPDATE model_attempt_controls SET state=CASE WHEN model_attempt_id=? THEN 'SUCCEEDED' ELSE state END,input_tokens=?,output_tokens=?,actual_cost_usd=?,usage_state=?,updated_at_utc=? WHERE model_attempt_id IN (SELECT model_attempt_id FROM model_attempts WHERE model_execution_id=?)",
                        (
                            permit.attempt_id,
                            attempt_tokens.input_tokens,
                            attempt_tokens.output_tokens,
                            None if attempt_charge.usd is None else str(attempt_charge.usd),
                            attempt_tokens.state,
                            stamp(self._state.clock()),
                            execution_id,
                        ),
                    )
                    ctx.execute(
                        "UPDATE model_attempts SET output_artifact_id=? WHERE model_attempt_id=?",
                        (output_ref.artifact_id, permit.attempt_id),
                    )
                    task_row = ctx.execute(
                        "SELECT state,revision FROM tasks WHERE task_id=?", (task.task_id,)
                    ).fetchone()
                    reject(
                        task_row is not None and allowed("TASK", task_row["state"], "SUCCEEDED"),
                        "TASK_SUCCESS_TRANSITION_DENIED",
                    )
                    ctx.execute(
                        "UPDATE task_attempts SET state='SUCCEEDED',output_artifact_id=?,usage_state=?,revision=revision+1 WHERE attempt_id=?",
                        (
                            output_ref.artifact_id,
                            "UNKNOWN_USAGE"
                            if charge.state == "UNKNOWN" or tokens.state == "UNKNOWN_USAGE"
                            else tokens.state,
                            task.reservation_id,
                        ),
                    )
                    ctx.execute(
                        "UPDATE tasks SET state='SUCCEEDED',revision=revision+1,updated_at_utc=? WHERE task_id=?",
                        (stamp(self._state.clock()), task.task_id),
                    )
                    requests = len(attempt_ids)
                    if charge.state == "UNKNOWN" or tokens.state == "UNKNOWN_USAGE":
                        usage = state_dto.Usage(state="UNKNOWN_USAGE")
                    else:
                        token_total = (tokens.input_tokens or 0) + (tokens.output_tokens or 0)
                        usage = state_dto.Usage(
                            state="KNOWN"
                            if charge.state == "KNOWN" and tokens.state == "KNOWN"
                            else "ESTIMATED",
                            cost_usd=charge.usd,
                            tokens=token_total,
                            requests=requests,
                        )
                    accounting.settle(
                        ctx,
                        task.run_id,
                        task.task_id,
                        task.reservation_id,
                        usage,
                        stamp(self._state.clock()),
                    )
                    return _ModelReceipt(accepted=True, code="MODEL_OUTPUT_COMMITTED")
                except _Rejected as exc:
                    return _ModelReceipt(accepted=False, code=exc.code)
                except Exception:
                    return _ModelReceipt(accepted=False, code="MODEL_OUTPUT_TRANSACTION_REJECTED")

            receipt = self._transact(
                task,
                "MODEL_OUTPUT_PUBLISHED",
                "provider.model_output_published",
                operation,
                request=request,
            )
            if not receipt.accepted:
                raise RuntimeError(receipt.code)
        return output_ref


def reject(condition: bool, reason: str) -> None:
    if not condition:
        raise _Rejected(reason)
