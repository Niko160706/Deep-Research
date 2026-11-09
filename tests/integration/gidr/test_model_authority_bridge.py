from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from gidr.contracts import commands as state_dto
from gidr.contracts.common import Language, new_id
from gidr.contracts.models import (
    ChoiceDecision,
    FinancialCharge,
    ModelTask,
    TokenUsage,
)
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.gateway import ModelGatewayService
from gidr.infrastructure.models.ports import (
    DecisionQuestion,
    ExecutableModelCall,
    ModelCall,
    ModelHTTPResponse,
    ProviderOutput,
    ProviderProfile,
)
from gidr.infrastructure.models.state_bridge import StateAuthorityModelBridge

spec = importlib.util.spec_from_file_location(
    "gidr_model_state_support", Path(__file__).with_name("state_support.py")
)
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


@pytest.fixture
def h(tmp_path):
    harness = support.Harness(tmp_path)
    with harness.authority:
        yield harness


class AuditedSyntheticDispatch:
    """Synthetic transport that exercises durable permit transitions, without I/O."""

    def __init__(self, bridge: StateAuthorityModelBridge):
        self.bridge = bridge
        self.calls = 0

    async def send(
        self,
        *,
        permit,
        url,
        method,
        headers,
        body,
        timeout_seconds,
        max_response_bytes,
    ):
        self.calls += 1
        await self.bridge.begin_dispatch(permit, url, method, body)
        response_body = b'{"synthetic":"model response"}'
        await self.bridge.finish_dispatch(permit, 200, response_body)
        return ModelHTTPResponse(200, {"content-type": "application/json"}, response_body)


class SyntheticJev:
    def __init__(self, profile: ProviderProfile):
        self.profile = profile

    async def invoke(self, call: ModelCall, permit):
        await permit.dispatch.send(
            permit=permit,
            url="https://api.typesafe.ai/v1/systemone",
            method="POST",
            headers={"authorization": "Bearer synthetic-only", "content-type": "application/json"},
            body=call.model_dump_json().encode(),
            timeout_seconds=1,
            max_response_bytes=1024,
        )
        return ProviderOutput(
            model_id=self.profile.model_id,
            model_version="synthetic-contract-test",
            tokens=TokenUsage(state="KNOWN", input_tokens=24, output_tokens=3),
            charge=FinancialCharge(state="UNKNOWN", usd=None),
            decision=ChoiceDecision(choices=("YES", "NO"), selected="YES"),
        )


def _task(h, run: str, task_id: str, grant_id: str):
    ref = h.store.publish(b"Synthetic source input for model bridge integration.")
    spec = state_dto.ScheduledTask(
        task_id=task_id,
        question_id="RQ-model",
        task_type="classification",
        reason_code="SYNTHETIC_MODEL_TASK",
        input_refs=(ref,),
        provider_id="jev",
        category="model",
        work_kind="PROVIDER",
        max_cost_usd=Decimal(2),
        max_tokens=10000,
        max_requests=2,
        max_attempts=2,
    )
    assert h.send(state_dto.AddTasks(tasks=(spec,)), actor="SYSTEM", run=run).accepted
    assert h.send(
        state_dto.ChangeTask(task_id=task_id, target="READY", reason="MODEL_TASK_APPROVED"),
        actor="SCHEDULER",
        run=run,
        grant=grant_id,
    ).accepted
    reserved = h.send(
        state_dto.ReserveTask(task_id=task_id, attempt_id=new_id("ATT"), worker_id="worker-model"),
        actor="SCHEDULER",
        run=run,
        grant=grant_id,
    )
    assert reserved.accepted and reserved.token is not None
    assert h.send(
        state_dto.StartAttempt(token=reserved.token),
        actor="WORKER",
        run=run,
        grant=grant_id,
    ).accepted
    model_task = ModelTask(
        run_id=run,
        created_at_utc=datetime.now(UTC),
        task_id=task_id,
        plan_version=1,
        authorization_ref=grant_id,
        task_type="classification",
        required_capabilities=("classification",),
        input_refs=(ref,),
        input_language=Language(code="en", status="DECLARED"),
        risk_class="LOW",
        reservation_id=reserved.token.attempt_id,
        attempt_limit=1,
        preferred_provider="jev",
        prompt_version="test-prompt-v1",
        privacy_scope="PUBLIC_SOURCE",
    )
    return model_task


@pytest.mark.asyncio
async def test_model_gateway_uses_state_authority_and_atomically_publishes_success(h):
    grant = h.policy(
        "GRANT-model",
        providers=("jev",),
        categories=("model",),
        task_types=("classification",),
        max_cost_usd=Decimal(5),
        max_tokens=20000,
        max_requests=10,
    )
    run = h.setup(name="model", policy=grant, categories=("model",))
    task = _task(h, run, "TASK-model", grant.grant_id)
    bridge = StateAuthorityModelBridge(h.authority)
    dispatch = AuditedSyntheticDispatch(bridge)
    bridge.bind_dispatch(dispatch)
    profile = ProviderProfile(
        provider_id="jev",
        model_id="synthetic-jev-model",
        capabilities=frozenset({"classification"}),
        operations=frozenset({"DECISION"}),
        languages=frozenset({"en"}),
    )
    gateway = ModelGatewayService(
        (SyntheticJev(profile),),
        authority=bridge,
        publisher=bridge,
        secret_policy=SecretPolicy(),
    )
    call = ModelCall(
        operation="DECISION",
        content="Synthetic source text.",
        input_language=Language(code="en", status="DECLARED"),
        question=DecisionQuestion(
            key="relevance",
            kind="CHOICE",
            instructions="Choose one label.",
            choices=("YES", "NO"),
        ),
        max_output_tokens=256,
        prompt_version="test-prompt-v1",
    )

    result = await gateway.execute(task, ExecutableModelCall(contract=call))

    assert result.status == "SUCCESS", result.model_dump_json()
    assert result.output_ref is not None
    assert result.charge.state == "UNKNOWN"
    assert dispatch.calls == 1
    assert h.row("tasks", "task_id", task.task_id)["state"] == "SUCCEEDED"
    assert h.row("task_attempts", "attempt_id", task.reservation_id)["state"] == "SUCCEEDED"
    execution = h.row("model_executions", "model_execution_id", result.model_execution_id)
    assert execution["result_json"] is not None
    assert execution["output_artifact_id"] == result.output_ref.artifact_id
    control = h.row("model_attempt_controls", "model_attempt_id", result.attempt_ids[0])
    assert control["state"] == "SUCCEEDED"
    assert control["response_status"] == 200
    assert control["dispatch_body_sha256"] is not None
    with h.db.reader() as conn:
        assert (
            conn.execute(
                "SELECT count(*) FROM audit_events WHERE event_type='provider.model_output_published'"
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT count(*) FROM artifact_refs WHERE owner_type='MODEL_EXECUTION' AND owner_id=?",
                (result.model_execution_id,),
            ).fetchone()[0]
            == 2
        )
