"""Vendor-neutral model calls and authority seams."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal, Protocol

from pydantic import Field

from gidr.contracts.common import (
    ArtifactRef,
    Contract,
    Language,
    Name,
    Positive,
    Text,
)
from gidr.contracts.models import (
    FinancialCharge,
    ModelErrorCode,
    ModelExecutionResult,
    ModelTask,
    TokenUsage,
    TypedDecision,
)


class DecisionQuestion(Contract):
    key: Name
    kind: Literal["NOUL", "CHOICE", "SCORE"]
    instructions: Text
    choices: tuple[Text, ...] = ()
    levels: tuple[Text, ...] = ()
    positive_criteria: Text | None = None
    negative_criteria: Text | None = None
    rubric_version: Text | None = None

    def validate_shape(self) -> None:
        if self.kind == "CHOICE" and len(self.choices) < 2:
            raise ValueError("CHOICE_REQUIRES_TWO_OPTIONS")
        if self.kind == "SCORE" and len(self.levels) < 2:
            raise ValueError("SCORE_REQUIRES_TWO_LEVELS")
        if self.kind == "SCORE" and not self.rubric_version:
            raise ValueError("SCORE_REQUIRES_RUBRIC_VERSION")
        if self.kind != "CHOICE" and self.choices:
            raise ValueError("UNEXPECTED_CHOICES")
        if self.kind != "SCORE" and self.levels:
            raise ValueError("UNEXPECTED_SCORE_LEVELS")


class ModelCall(Contract):
    """Request body. `content` must contain data, never credentials or commands."""

    operation: Literal["DECISION", "GENERATION", "TRANSLATION"]
    content: Text
    input_language: Language
    response_language: Language | None = None
    question: DecisionQuestion | None = None
    schema_name: Name | None = None
    output_schema_version: Text | None = None
    output_json_schema: Mapping[str, object] | None = None
    max_output_tokens: Positive = 2048
    prompt_version: Text

    def validate_shape(self) -> None:
        if self.operation == "DECISION" and self.question is None:
            raise ValueError("DECISION_QUESTION_REQUIRED")
        if self.question is not None:
            self.question.validate_shape()
        if self.operation in {"GENERATION", "TRANSLATION"} and (
            self.schema_name is None or self.output_schema_version is None
        ):
            raise ValueError("STRUCTURED_OUTPUT_SCHEMA_REQUIRED")
        if self.operation == "TRANSLATION" and self.response_language is None:
            raise ValueError("TRANSLATION_TARGET_LANGUAGE_REQUIRED")


class ProviderProfile(Contract):
    provider_id: Name
    model_id: Text
    capabilities: frozenset[Name]
    operations: frozenset[Literal["DECISION", "GENERATION", "TRANSLATION"]]
    languages: frozenset[str] = frozenset()
    accepts_unknown_language: bool = False
    privacy_scopes: frozenset[Literal["PUBLIC_SOURCE", "APPROVED_PRIVATE"]] = frozenset(
        {"PUBLIC_SOURCE"}
    )
    max_input_chars: Positive = 100_000
    max_output_tokens: Positive = 4096
    supports_json_object: bool = False
    estimated_input_usd_per_million: Decimal | None = Field(default=None, ge=0)
    estimated_output_usd_per_million: Decimal | None = Field(default=None, ge=0)

    def supports(self, task: ModelTask, call: ModelCall) -> bool:
        language = task.input_language
        language_ok = (
            self.accepts_unknown_language
            if language.status in {"UNKNOWN", "FAILED", "MIXED"}
            else language.code in self.languages
        )
        return (
            set(task.required_capabilities) <= self.capabilities
            and call.operation in self.operations
            and (call.operation == "DECISION" or self.supports_json_object)
            and task.privacy_scope in self.privacy_scopes
            and len(call.model_dump_json()) <= self.max_input_chars
            and call.max_output_tokens <= self.max_output_tokens
            and language_ok
        )


@dataclass(frozen=True, slots=True)
class ProviderOutput:
    model_id: str
    model_version: str | None
    tokens: TokenUsage
    charge: FinancialCharge
    decision: TypedDecision | None = None
    structured_text: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class ModelDispatchPermit:
    """Ephemeral permit created by a durable attempt authority before each call."""

    attempt_id: str
    run_id: str
    task_id: str
    execution_id: str
    provider_id: str
    reservation_id: str
    dispatch: ModelHTTPDispatch
    opaque_token: object

    def __repr__(self) -> str:
        return (
            f"ModelDispatchPermit(attempt_id={self.attempt_id!r}, "
            f"provider_id={self.provider_id!r}, token=<redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class ModelHTTPResponse:
    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def __repr__(self) -> str:
        return f"ModelHTTPResponse(status_code={self.status_code}, body=<redacted>)"


class ModelHTTPDispatch(Protocol):
    """Production implementation must pin DNS, reject redirects and sanitize logs."""

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
    ) -> ModelHTTPResponse: ...


@dataclass(frozen=True, slots=True)
class ReservedModelAttempt:
    permit: ModelDispatchPermit
    estimated_cost_usd: Decimal | None


class ModelAttemptAuthority(Protocol):
    """Atomic StateAuthority adapter: reserves and authorizes one provider attempt."""

    async def reserve(
        self,
        task: ModelTask,
        call: ModelCall,
        execution_id: str,
        profile: ProviderProfile,
        attempt_number: int,
    ) -> ReservedModelAttempt: ...

    async def settle_failure(
        self,
        task: ModelTask,
        permit: ModelDispatchPermit,
        tokens: TokenUsage,
        charge: FinancialCharge,
        error_code: ModelErrorCode,
    ) -> None: ...

    async def cooldown_remaining(self, provider_id: str) -> float: ...

    async def impose_cooldown(self, provider_id: str, seconds: float) -> None: ...

    async def record_failure(self, task: ModelTask, result: ModelExecutionResult) -> None: ...


class ModelOutputPublisher(Protocol):
    """Atomically stores output, canonical event, task success and final attempt usage."""

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
    ) -> ArtifactRef: ...


class ProviderAdapter(Protocol):
    profile: ProviderProfile

    async def invoke(self, call: ModelCall, permit: ModelDispatchPermit) -> ProviderOutput: ...


OutputValidator = Callable[[str], Awaitable[object] | object]


@dataclass(frozen=True, slots=True)
class ExecutableModelCall:
    """In-process call plus trusted validator; validators are never model-supplied."""

    contract: ModelCall
    validator: OutputValidator | None = None


class ModelAttemptDenied(RuntimeError):
    """No durable authorization/budget permit was granted; no provider was called."""


class ProviderFailure(RuntimeError):
    def __init__(self, error_code: ModelErrorCode, *, retry_after_seconds: float | None = None):
        allowed = {
            "TIMEOUT",
            "RATE_LIMIT",
            "AUTH",
            "SCHEMA_INVALID",
            "CONTEXT_TOO_LONG",
            "LANGUAGE_UNSUPPORTED",
            "POLICY_BLOCK",
            "BUDGET_EXCEEDED",
            "PROVIDER_OUTAGE",
        }
        if error_code not in allowed:
            error_code = "PROVIDER_OUTAGE"
        super().__init__(error_code)
        self.error_code: ModelErrorCode = error_code
        self.retry_after_seconds = retry_after_seconds


class ModelGateway(Protocol):
    async def execute(self, task: ModelTask, call: ExecutableModelCall) -> ModelExecutionResult: ...
