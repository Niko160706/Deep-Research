from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel

from gidr.contracts.common import ArtifactRef, Language, new_id
from gidr.contracts.models import (
    FinancialCharge,
    ModelTask,
    NoulDecision,
    ScoreDecision,
    TokenUsage,
)
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.gateway import (
    ModelGatewayService,
    ModelOutputPersistenceError,
    ModelRouteUnavailable,
)
from gidr.infrastructure.models.generative import OpenAICompatibleProvider
from gidr.infrastructure.models.ports import (
    DecisionQuestion,
    ExecutableModelCall,
    ModelAttemptDenied,
    ModelCall,
    ModelDispatchPermit,
    ModelHTTPResponse,
    ModelOutputPublisher,
    ProviderFailure,
    ProviderOutput,
    ProviderProfile,
    ReservedModelAttempt,
)
from gidr.infrastructure.models.transport import AuthorizedModelTransport, ModelTransportDenied
from gidr.infrastructure.models.typesafe import TypeSafeJevProvider
from gidr.infrastructure.security.events import SecurityObservation
from gidr.infrastructure.security.secrets import SecretManager


class FakeDispatch:
    def __init__(self, response: ModelHTTPResponse | None = None) -> None:
        self.response = response or ModelHTTPResponse(
            200, {"content-type": "application/json"}, b"{}"
        )
        self.calls: list[dict[str, object]] = []

    async def send(
        self,
        *,
        permit: ModelDispatchPermit,
        url: str,
        method: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> ModelHTTPResponse:
        self.calls.append(
            {
                "permit": permit,
                "url": url,
                "method": method,
                "headers": dict(headers),
                "body": body,
                "timeout": timeout_seconds,
                "max_response_bytes": max_response_bytes,
            }
        )
        return self.response


class AuditSinkStub:
    def __init__(self) -> None:
        self.observations: list[SecurityObservation] = []

    def record_security(self, observation: SecurityObservation) -> None:
        self.observations.append(observation)


class FakeAuthority:
    def __init__(self, dispatch: FakeDispatch | None = None) -> None:
        self.dispatch = dispatch or FakeDispatch()
        self.reservations: list[tuple[str, int]] = []
        self.settlements: list[tuple[str, TokenUsage, FinancialCharge]] = []
        self.failures = []
        self.cooldowns: dict[str, float] = {}
        self.deny = False

    async def reserve(
        self,
        task: ModelTask,
        call,
        execution_id: str,
        profile: ProviderProfile,
        attempt_number: int,
    ):
        if self.deny:
            raise ModelAttemptDenied("TEST_BUDGET_DENIED")
        attempt_id = new_id("ATT")
        self.reservations.append((profile.provider_id, attempt_number))
        permit = ModelDispatchPermit(
            attempt_id=attempt_id,
            run_id=task.run_id,
            task_id=task.task_id,
            execution_id=execution_id,
            provider_id=profile.provider_id,
            reservation_id=task.reservation_id,
            dispatch=self.dispatch,
            opaque_token=object(),
        )
        return ReservedModelAttempt(permit=permit, estimated_cost_usd=None)

    async def settle_failure(
        self,
        task: ModelTask,
        permit: ModelDispatchPermit,
        tokens: TokenUsage,
        charge: FinancialCharge,
        error_code: str,
    ) -> None:
        self.settlements.append((permit.attempt_id, tokens, charge))

    async def cooldown_remaining(self, provider_id: str) -> float:
        return self.cooldowns.get(provider_id, 0.0)

    async def impose_cooldown(self, provider_id: str, seconds: float) -> None:
        self.cooldowns[provider_id] = seconds

    async def record_failure(self, task: ModelTask, result) -> None:
        self.failures.append(result)


class FakePublisher(ModelOutputPublisher):
    def __init__(self) -> None:
        self.outputs: list[bytes] = []
        self.metadata: list[dict[str, object]] = []

    async def publish(self, task, execution_id, output, media_type, **metadata):
        self.outputs.append(output)
        self.metadata.append(metadata)
        return ArtifactRef(artifact_id=new_id("ART"), sha256="a" * 64, size_bytes=len(output))


class FailPublisher(FakePublisher):
    async def publish(self, task, execution_id, output, media_type, **metadata):
        raise RuntimeError("synthetic-publisher-error-must-not-escape")


class FakeProvider:
    def __init__(self, profile: ProviderProfile, outcomes: list[ProviderOutput | str]) -> None:
        self.profile = profile
        self.outcomes = outcomes
        self.calls = 0

    async def invoke(self, call: ModelCall, permit: ModelDispatchPermit) -> ProviderOutput:
        self.calls += 1
        value = self.outcomes.pop(0)
        if value == "TIMEOUT":
            raise ProviderFailure("TIMEOUT")
        if value == "RATE_LIMIT":
            raise ProviderFailure("RATE_LIMIT", retry_after_seconds=999)
        if value == "AUTH":
            raise ProviderFailure("AUTH")
        if value == "SCHEMA_INVALID":
            raise ProviderFailure("SCHEMA_INVALID")
        assert isinstance(value, ProviderOutput)
        return value


class SyntheticOutput(BaseModel):
    answer: str


def profile(
    provider_id: str,
    *,
    operations: tuple[str, ...] = ("DECISION", "GENERATION", "TRANSLATION"),
    accepts_unknown: bool = True,
    json_mode: bool = True,
) -> ProviderProfile:
    return ProviderProfile(
        provider_id=provider_id,
        model_id=f"{provider_id}-configured-model",
        capabilities=frozenset({"classification", "reasoning", "translation"}),
        operations=frozenset(operations),
        languages=frozenset({"en", "ja", "ko"}),
        accepts_unknown_language=accepts_unknown,
        privacy_scopes=frozenset({"PUBLIC_SOURCE"}),
        supports_json_object=json_mode,
    )


def task(
    *, preferred: str = "jev", fallbacks: tuple[str, ...] = (), language: Language | None = None
) -> ModelTask:
    ref = ArtifactRef(artifact_id=new_id("ART"), sha256="b" * 64, size_bytes=2)
    return ModelTask(
        run_id=new_id("RUN"),
        created_at_utc=datetime(2026, 11, 8, tzinfo=UTC),
        task_id=new_id("TASK"),
        plan_version=1,
        authorization_ref="GRANT-TEST",
        task_type="relevance",
        required_capabilities=("classification",),
        input_refs=(ref,),
        input_language=language or Language(code="en", status="DECLARED"),
        risk_class="LOW",
        reservation_id="ATT-reservation-test",
        attempt_limit=3,
        preferred_provider=preferred,
        allowed_fallbacks=fallbacks,
        prompt_version="prompt-test-v1",
        privacy_scope="PUBLIC_SOURCE",
    )


def decision_call(content: str = "A source says something.") -> ExecutableModelCall:
    return ExecutableModelCall(
        contract=ModelCall(
            operation="DECISION",
            content=content,
            input_language=Language(code="en", status="DECLARED"),
            question=DecisionQuestion(
                key="relevance",
                kind="CHOICE",
                instructions="Choose the relevance label.",
                choices=("RELEVANT", "IRRELEVANT", "UNKNOWN"),
            ),
            prompt_version="prompt-test-v1",
        )
    )


def output_decision(selected: str = "RELEVANT") -> ProviderOutput:
    from gidr.contracts.models import ChoiceDecision, DecisionConfidence

    return ProviderOutput(
        model_id="mock-model-2026-01",
        model_version=None,
        decision=ChoiceDecision(
            choices=("RELEVANT", "IRRELEVANT", "UNKNOWN"),
            selected=selected,
            confidence=DecisionConfidence(
                semantics="PROVIDER_SCORE", value=0.8, rubric_version="test-rubric-v1"
            ),
            probabilities={"RELEVANT": 0.8, "IRRELEVANT": 0.1, "UNKNOWN": 0.1},
        ),
        tokens=TokenUsage(state="KNOWN", input_tokens=12, output_tokens=3),
        charge=FinancialCharge(state="UNKNOWN", usd=None),
    )


def result_payload(*, choices: dict[str, object] | None = None) -> bytes:
    return json.dumps(
        {
            "model": "jev-actual-response-model",
            "answers": choices
            or {
                "relevance": {
                    "type": "choice",
                    "choice": "RELEVANT",
                    "confidence": 0.82,
                    "probabilities": {"RELEVANT": 0.82, "IRRELEVANT": 0.08, "UNKNOWN": 0.1},
                }
            },
            "usage": {"input_tokens": 14, "output_tokens": 4},
        }
    ).encode()


def secrets(
    policy: SecretPolicy, sink: AuditSinkStub, key: str = "synthetic-key-canary"
) -> SecretManager:
    return SecretManager(
        secret_policy=policy,
        origins={"provider/typesafe": frozenset({"https://api.typesafe.ai:443"})},
        allow_environment=True,
        environ={"TYPESAFE_API_KEY": key},
        event_sink=sink,
    )


@pytest.mark.asyncio
async def test_official_jev_sdk_maps_actual_choice_probability_usage_and_unknown_charge() -> None:
    policy = SecretPolicy()
    sink = AuditSinkStub()
    dispatch = FakeDispatch(
        ModelHTTPResponse(200, {"content-type": "application/json"}, result_payload())
    )
    authority = FakeAuthority(dispatch)
    jev = TypeSafeJevProvider(
        profile("jev", operations=("DECISION",)),
        secret_manager=secrets(policy, sink),
        secret_policy=policy,
    )
    publisher = FakePublisher()
    gateway = ModelGatewayService(
        (jev,), authority=authority, publisher=publisher, secret_policy=policy
    )

    result = await gateway.execute(task(), decision_call())

    assert result.status == "SUCCESS"
    assert result.provider_id == "jev"
    assert result.model_id == "jev-actual-response-model"
    assert result.tokens == TokenUsage(state="KNOWN", input_tokens=14, output_tokens=4)
    assert result.charge == FinancialCharge(state="UNKNOWN", usd=None)
    assert result.decision is not None
    assert result.decision.selected == "RELEVANT"
    assert result.decision.probabilities == {
        "RELEVANT": 0.82,
        "IRRELEVANT": 0.08,
        "UNKNOWN": 0.1,
    }
    assert len(dispatch.calls) == 1
    outgoing = dispatch.calls[0]
    assert outgoing["url"] == "https://api.typesafe.ai/v1/systemone"
    assert "synthetic-key-canary" not in str(outgoing["body"])
    assert outgoing["headers"]["authorization"] == "Bearer synthetic-key-canary"
    assert sink.observations[0].credential_alias == "provider/typesafe"
    assert b"synthetic-key-canary" not in publisher.outputs[0]
    assert publisher.metadata[0]["attempt_tokens"] == TokenUsage(
        state="KNOWN", input_tokens=14, output_tokens=4
    )


@pytest.mark.asyncio
async def test_official_jev_adapter_uses_zero_sdk_retries_for_429() -> None:
    policy = SecretPolicy()
    sink = AuditSinkStub()
    dispatch = FakeDispatch(
        ModelHTTPResponse(
            429,
            {"content-type": "application/json", "retry-after": "999"},
            b'{"message":"rate limit"}',
        )
    )
    authority = FakeAuthority(dispatch)
    jev = TypeSafeJevProvider(
        profile("jev", operations=("DECISION",)),
        secret_manager=secrets(policy, sink),
        secret_policy=policy,
    )
    gateway = ModelGatewayService(
        (jev,), authority=authority, publisher=FakePublisher(), secret_policy=policy
    )

    result = await gateway.execute(task(), decision_call())

    assert result.status == "FAILED"
    assert result.error_code == "RATE_LIMIT"
    assert result.tokens.state == "UNKNOWN_USAGE"
    assert len(dispatch.calls) == 1
    assert authority.cooldowns["jev"] == 5.0


@pytest.mark.asyncio
async def test_jev_401_is_sanitized_and_does_not_fallback() -> None:
    policy = SecretPolicy()
    sink = AuditSinkStub()
    dispatch = FakeDispatch(
        ModelHTTPResponse(401, {"content-type": "application/json"}, b'{"message":"bad key"}')
    )
    authority = FakeAuthority(dispatch)
    jev = TypeSafeJevProvider(
        profile("jev", operations=("DECISION",)),
        secret_manager=secrets(policy, sink, key="synthetic-auth-canary"),
        secret_policy=policy,
    )
    fallback = FakeProvider(profile("gemini"), [output_decision()])
    gateway = ModelGatewayService(
        (jev, fallback),
        authority=authority,
        publisher=FakePublisher(),
        secret_policy=policy,
    )

    result = await gateway.execute(task(fallbacks=("gemini",)), decision_call())

    assert result.status == "FAILED"
    assert result.error_code == "AUTH"
    assert "synthetic-auth-canary" not in str(result)
    assert fallback.calls == 0
    assert len(dispatch.calls) == 1


@pytest.mark.asyncio
async def test_output_commit_failure_never_claims_success_or_retries_provider() -> None:
    provider = FakeProvider(profile("jev"), [output_decision()])
    authority = FakeAuthority()
    gateway = ModelGatewayService(
        (provider,),
        authority=authority,
        publisher=FailPublisher(),
        secret_policy=SecretPolicy(),
    )

    with pytest.raises(ModelOutputPersistenceError) as error:
        await gateway.execute(task(), decision_call())

    assert str(error.value) == "MODEL_OUTPUT_COMMIT_FAILED"
    assert provider.calls == 1
    assert authority.settlements == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "answer", "expected_type"),
    [
        (
            "NOUL",
            {"type": "noul", "noul": 0.73},
            NoulDecision,
        ),
        (
            "SCORE",
            {
                "type": "score",
                "score": 1.5,
                "confidence": 0.68,
                "legend": {"0": "low", "1": "medium", "2": "high"},
                "probabilities": {"0": 0.1, "1": 0.7, "2": 0.2},
            },
            ScoreDecision,
        ),
    ],
)
async def test_jev_noul_and_score_preserve_official_return_fields(
    kind: str, answer: dict[str, object], expected_type: type[object]
) -> None:
    policy = SecretPolicy()
    sink = AuditSinkStub()
    dispatch = FakeDispatch(
        ModelHTTPResponse(
            200,
            {"content-type": "application/json"},
            result_payload(choices={"typed_answer": answer}),
        )
    )
    jev = TypeSafeJevProvider(
        profile("jev", operations=("DECISION",)),
        secret_manager=secrets(policy, sink),
        secret_policy=policy,
    )
    gateway = ModelGatewayService(
        (jev,),
        authority=FakeAuthority(dispatch),
        publisher=FakePublisher(),
        secret_policy=policy,
    )
    question = (
        DecisionQuestion(
            key="typed_answer",
            kind="NOUL",
            instructions="Does the source support the claim?",
        )
        if kind == "NOUL"
        else DecisionQuestion(
            key="typed_answer",
            kind="SCORE",
            instructions="Rate the evidence quality.",
            levels=("low", "medium", "high"),
            rubric_version="evidence-quality-v2",
        )
    )
    result = await gateway.execute(
        task(),
        ExecutableModelCall(
            contract=ModelCall(
                operation="DECISION",
                content="Source data.",
                input_language=Language(code="en", status="DECLARED"),
                question=question,
                prompt_version="typed-v1",
            )
        ),
    )

    assert result.status == "SUCCESS"
    assert isinstance(result.decision, expected_type)
    if isinstance(result.decision, NoulDecision):
        assert result.decision.probability == 0.73
    else:
        assert isinstance(result.decision, ScoreDecision)
        assert result.decision.raw_provider_score == 1.5
        assert result.decision.score == 0.75
        assert result.decision.probabilities == {0: 0.1, 1: 0.7, 2: 0.2}
        assert result.decision.legend == {0: "low", 1: "medium", 2: "high"}


@pytest.mark.asyncio
async def test_unknown_language_is_routed_only_when_profile_explicitly_accepts_it() -> None:
    unknown = Language(code="unknown", status="UNKNOWN")
    preferred = FakeProvider(profile("jev", accepts_unknown=False), [output_decision()])
    fallback = FakeProvider(profile("gemini"), [output_decision()])
    gateway = ModelGatewayService(
        (preferred, fallback),
        authority=FakeAuthority(),
        publisher=FakePublisher(),
        secret_policy=SecretPolicy(),
    )

    result = await gateway.execute(
        task(preferred="jev", fallbacks=("gemini",), language=unknown), decision_call()
    )

    assert result.status == "SUCCESS"
    assert result.provider_id == "gemini"
    assert result.fallback_used is True
    assert preferred.calls == 0
    assert fallback.calls == 1


@pytest.mark.asyncio
async def test_timeout_retries_are_total_bounded_and_unknown_usage_is_never_zero() -> None:
    provider = FakeProvider(profile("jev"), ["TIMEOUT", output_decision()])
    authority = FakeAuthority()
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    publisher = FakePublisher()
    gateway = ModelGatewayService(
        (provider,),
        authority=authority,
        publisher=publisher,
        secret_policy=SecretPolicy(),
        max_total_attempts=2,
        sleep=fake_sleep,
    )

    result = await gateway.execute(task(), decision_call())

    assert result.status == "SUCCESS"
    assert len(result.attempt_ids) == 2
    assert result.tokens.state == "UNKNOWN_USAGE"
    assert result.charge.state == "UNKNOWN"
    assert len(authority.settlements) == 1
    assert authority.settlements[0][1].state == "UNKNOWN_USAGE"
    assert len(publisher.metadata) == 1
    assert publisher.metadata[0]["attempt_tokens"] == TokenUsage(
        state="KNOWN", input_tokens=12, output_tokens=3
    )
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_rate_limit_sets_cooldown_and_uses_only_explicit_fallback() -> None:
    jev = FakeProvider(profile("jev"), ["RATE_LIMIT"])
    gemini = FakeProvider(profile("gemini"), [output_decision()])
    authority = FakeAuthority()
    publisher = FakePublisher()
    gateway = ModelGatewayService(
        (jev, gemini),
        authority=authority,
        publisher=publisher,
        secret_policy=SecretPolicy(),
    )

    result = await gateway.execute(task(preferred="jev", fallbacks=("gemini",)), decision_call())

    assert result.status == "SUCCESS"
    assert result.provider_id == "gemini"
    assert result.fallback_used is True
    assert authority.cooldowns["jev"] == 5.0
    assert [provider for provider, _ in authority.reservations] == ["jev", "gemini"]
    assert len(authority.settlements) == 1
    assert len(publisher.metadata) == 1
    assert result.tokens.state == "UNKNOWN_USAGE"


@pytest.mark.asyncio
async def test_auth_failure_does_not_fallback_and_budget_refusal_prevents_call() -> None:
    jev = FakeProvider(profile("jev"), ["AUTH"])
    gemini = FakeProvider(profile("gemini"), [output_decision()])
    authority = FakeAuthority()
    gateway = ModelGatewayService(
        (jev, gemini),
        authority=authority,
        publisher=FakePublisher(),
        secret_policy=SecretPolicy(),
    )
    result = await gateway.execute(task(preferred="jev", fallbacks=("gemini",)), decision_call())
    assert result.error_code == "AUTH"
    assert jev.calls == 1 and gemini.calls == 0

    denied_authority = FakeAuthority()
    denied_authority.deny = True
    denied_gateway = ModelGatewayService(
        (gemini,),
        authority=denied_authority,
        publisher=FakePublisher(),
        secret_policy=SecretPolicy(),
    )
    before = gemini.calls
    with pytest.raises(ModelAttemptDenied):
        await denied_gateway.execute(task(preferred="jev", fallbacks=("gemini",)), decision_call())
    assert gemini.calls == before


@pytest.mark.asyncio
async def test_secret_in_model_content_and_incompatible_routes_are_rejected_before_authority() -> (
    None
):
    policy = SecretPolicy()
    policy.register_secret("synthetic-prompt-secret")
    provider = FakeProvider(profile("jev"), [output_decision()])
    authority = FakeAuthority()
    gateway = ModelGatewayService(
        (provider,), authority=authority, publisher=FakePublisher(), secret_policy=policy
    )
    with pytest.raises(ModelAttemptDenied):
        await gateway.execute(task(), decision_call("source synthetic-prompt-secret text"))
    base_call = decision_call()
    assert base_call.contract.question is not None
    secret_question = ExecutableModelCall(
        contract=base_call.contract.model_copy(
            update={
                "question": base_call.contract.question.model_copy(
                    update={"instructions": "include synthetic-prompt-secret"}
                )
            }
        )
    )
    with pytest.raises(ModelAttemptDenied):
        await gateway.execute(task(), secret_question)
    assert authority.reservations == []
    assert provider.calls == 0

    with pytest.raises(ModelAttemptDenied, match="TYPED_DECISION_REQUIRES_JEV_PRIMARY"):
        await gateway.execute(task(preferred="gemini"), decision_call())

    unavailable = ModelGatewayService(
        (), authority=FakeAuthority(), publisher=FakePublisher(), secret_policy=policy
    )
    with pytest.raises(ModelRouteUnavailable):
        await unavailable.execute(task(), decision_call())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_id", "origin", "path"),
    [
        ("gemini", "https://generativelanguage.googleapis.com", "/v1beta/openai/chat/completions"),
        ("groq", "https://api.groq.com", "/openai/v1/chat/completions"),
        ("siliconflow", "https://api.siliconflow.cn", "/v1/chat/completions"),
    ],
)
async def test_supported_generative_providers_use_fixed_endpoint_and_actual_tokens(
    provider_id: str, origin: str, path: str
) -> None:
    policy = SecretPolicy()
    sink = AuditSinkStub()
    key = f"synthetic-{provider_id}-key"
    alias = f"provider/{provider_id}"
    secret_manager = SecretManager(
        secret_policy=policy,
        origins={alias: frozenset({origin + ":443"})},
        allow_environment=True,
        environ={
            "GEMINI_API_KEY": key if provider_id == "gemini" else "",
            "GROQ_API_KEY": key if provider_id == "groq" else "",
            "SILICONFLOW_API_KEY": key if provider_id == "siliconflow" else "",
        },
        event_sink=sink,
    )
    payload = json.dumps(
        {
            "model": f"{provider_id}-response-model",
            "choices": [{"message": {"content": '{"answer":"validated"}'}}],
            "usage": {"prompt_tokens": 21, "completion_tokens": 5},
        }
    ).encode()
    authority = FakeAuthority(
        FakeDispatch(ModelHTTPResponse(200, {"content-type": "application/json"}, payload))
    )
    provider_profile = profile(provider_id, operations=("GENERATION",), json_mode=True)
    adapter = OpenAICompatibleProvider(
        provider_profile, secret_manager=secret_manager, secret_policy=policy
    )

    def validate(raw: str) -> SyntheticOutput:
        return SyntheticOutput.model_validate_json(raw)

    call = ExecutableModelCall(
        contract=ModelCall(
            operation="GENERATION",
            content="Untrusted evidence data.",
            input_language=Language(code="en", status="DECLARED"),
            schema_name="synthetic_answer",
            output_schema_version="1.0",
            output_json_schema=SyntheticOutput.model_json_schema(),
            prompt_version="gen-v1",
        ),
        validator=validate,
    )
    publisher = FakePublisher()
    gateway = ModelGatewayService(
        (adapter,), authority=authority, publisher=publisher, secret_policy=policy
    )

    result = await gateway.execute(task(preferred=provider_id), call)

    sent = authority.dispatch.calls[0]
    assert sent["url"] == origin + path
    assert sent["headers"]["authorization"] == f"Bearer {key}"
    assert key.encode() not in sent["body"]
    assert result.status == "SUCCESS"
    assert result.model_id == f"{provider_id}-response-model"
    assert result.tokens == TokenUsage(state="KNOWN", input_tokens=21, output_tokens=5)
    assert result.charge.state == "UNKNOWN"
    assert publisher.outputs == [b'{"answer":"validated"}']


@pytest.mark.asyncio
async def test_generative_invalid_json_and_missing_usage_do_not_claim_schema_or_zero_cost() -> None:
    provider = FakeProvider(
        profile("gemini", operations=("GENERATION",)),
        [
            ProviderOutput(
                model_id="gemini-configured-model",
                model_version=None,
                structured_text='{"answer":"ok","answer":"duplicate"}',
                tokens=TokenUsage(state="UNKNOWN_USAGE"),
                charge=FinancialCharge(state="UNKNOWN", usd=None),
            )
        ],
    )
    authority = FakeAuthority()
    publisher = FakePublisher()

    def validate(raw: str) -> SyntheticOutput:
        return SyntheticOutput.model_validate_json(raw)

    gateway = ModelGatewayService(
        (provider,), authority=authority, publisher=publisher, secret_policy=SecretPolicy()
    )
    call = ExecutableModelCall(
        contract=ModelCall(
            operation="GENERATION",
            content="Data.",
            input_language=Language(code="unknown", status="UNKNOWN"),
            schema_name="answer",
            output_schema_version="1.0",
            prompt_version="gen-v1",
        ),
        validator=validate,
    )
    result = await gateway.execute(
        task(preferred="gemini", language=Language(code="unknown", status="UNKNOWN")), call
    )
    assert result.status == "FAILED"
    assert result.error_code == "SCHEMA_INVALID"
    assert result.tokens.state == "UNKNOWN_USAGE"
    assert result.charge.usd is None
    assert publisher.outputs == []


def test_route_requires_explicit_json_mode_and_does_not_invent_unknown_language_support() -> None:
    language = Language(code="unknown", status="UNKNOWN")
    task_value = task(preferred="gemini", language=language)
    call = ModelCall(
        operation="GENERATION",
        content="An ambiguous language source.",
        input_language=language,
        schema_name="answer",
        output_schema_version="1.0",
        prompt_version="gen-v1",
    )
    assert not profile("gemini", operations=("GENERATION",), accepts_unknown=False).supports(
        task_value, call
    )
    assert not profile("gemini", operations=("GENERATION",), json_mode=False).supports(
        task_value, call
    )


@pytest.mark.asyncio
async def test_transport_blocks_redirects_oversized_and_secret_bearing_responses() -> None:
    from gidr.infrastructure.models.ports import ModelHTTPResponse

    permit = ModelDispatchPermit(
        attempt_id=new_id("ATT"),
        run_id=new_id("RUN"),
        task_id=new_id("TASK"),
        execution_id=new_id("MOD"),
        provider_id="jev",
        reservation_id=new_id("ATT"),
        dispatch=FakeDispatch(ModelHTTPResponse(302, {"location": "http://127.0.0.1/"}, b"")),
        opaque_token=object(),
    )
    policy = SecretPolicy()
    transport = AuthorizedModelTransport(
        permit,
        allowed_origin="https://api.typesafe.ai",
        allowed_path="/v1/systemone",
        secret_policy=policy,
    )
    import httpx2

    with pytest.raises(ModelTransportDenied, match="MODEL_REDIRECT_BLOCKED"):
        await transport.handle_async_request(
            httpx2.Request(
                "POST",
                "https://api.typesafe.ai/v1/systemone",
                json={"state": "x"},
            )
        )

    policy.register_secret("synthetic-response-canary")
    secret_permit = ModelDispatchPermit(
        attempt_id=new_id("ATT"),
        run_id=new_id("RUN"),
        task_id=new_id("TASK"),
        execution_id=new_id("MOD"),
        provider_id="jev",
        reservation_id=new_id("ATT"),
        dispatch=FakeDispatch(
            ModelHTTPResponse(200, {}, b"synthetic-response-canary must never return")
        ),
        opaque_token=object(),
    )
    secret_transport = AuthorizedModelTransport(
        secret_permit,
        allowed_origin="https://api.typesafe.ai",
        allowed_path="/v1/systemone",
        secret_policy=policy,
    )
    with pytest.raises(ModelTransportDenied, match="MODEL_RESPONSE_SECRET_OR_ENCODING_BLOCKED"):
        await secret_transport.handle_async_request(
            httpx2.Request("POST", "https://api.typesafe.ai/v1/systemone", json={"state": "x"})
        )


def test_unknown_and_mixed_languages_are_not_mistaken_for_unsupported_source() -> None:
    from gidr.infrastructure.models.language import ConservativeLanguageDetector

    detector = ConservativeLanguageDetector()
    assert detector.detect("English words that may be Indonesian") == Language(
        code="unknown", status="UNKNOWN"
    )
    assert detector.detect("日本語の文章") == Language(code="ja", status="DETECTED")
    assert detector.detect("한국어 문장") == Language(code="ko", status="DETECTED")
    assert detector.detect("日本語 plus English") == Language(code="mixed", status="MIXED")
    assert detector.detect(" ") == Language(code="unknown", status="UNKNOWN")
