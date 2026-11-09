"""Single-owner model routing, retry, budget and output provenance gateway."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import cast

from pydantic import ValidationError

from gidr.contracts.common import new_id, unique_object
from gidr.contracts.models import (
    FinancialCharge,
    ModelErrorCode,
    ModelExecutionResult,
    ModelTask,
    TokenUsage,
)
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.ports import (
    ExecutableModelCall,
    ModelAttemptAuthority,
    ModelAttemptDenied,
    ModelCall,
    ModelDispatchPermit,
    ModelOutputPublisher,
    ProviderAdapter,
    ProviderFailure,
    ProviderOutput,
)

_FALLBACK_ERRORS = {
    "TIMEOUT",
    "RATE_LIMIT",
    "SCHEMA_INVALID",
    "CONTEXT_TOO_LONG",
    "LANGUAGE_UNSUPPORTED",
    "PROVIDER_OUTAGE",
}
_RETRY_ERRORS = {"TIMEOUT", "PROVIDER_OUTAGE"}


class ModelRouteUnavailable(RuntimeError):
    """No configured provider meets task capability/language/privacy constraints."""


class ModelOutputPersistenceError(RuntimeError):
    """The provider response could not be safely committed as a model artifact."""


class ModelGatewayService:
    """No direct provider calls: every attempt must be durably reserved first."""

    def __init__(
        self,
        providers: tuple[ProviderAdapter, ...],
        *,
        authority: ModelAttemptAuthority,
        publisher: ModelOutputPublisher,
        secret_policy: SecretPolicy,
        max_total_attempts: int = 3,
        base_retry_seconds: float = 0.25,
        max_retry_seconds: float = 5.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 1 <= max_total_attempts <= 10 or not 0 <= base_retry_seconds <= 30:
            raise ValueError("MODEL_GATEWAY_LIMIT_INVALID")
        if not base_retry_seconds <= max_retry_seconds <= 120:
            raise ValueError("MODEL_GATEWAY_BACKOFF_INVALID")
        ids = [provider.profile.provider_id for provider in providers]
        if len(set(ids)) != len(ids):
            raise ValueError("MODEL_PROVIDER_ID_DUPLICATE")
        self._providers = {provider.profile.provider_id: provider for provider in providers}
        self._authority = authority
        self._publisher = publisher
        self._secrets = secret_policy
        self._max_attempts = max_total_attempts
        self._base_retry = base_retry_seconds
        self._max_retry = max_retry_seconds
        self._sleep = sleep
        self._monotonic = monotonic

    async def execute(self, task: ModelTask, request: ExecutableModelCall) -> ModelExecutionResult:
        call = request.contract
        if call.operation == "DECISION" and task.preferred_provider != "jev":
            raise ModelAttemptDenied("TYPED_DECISION_REQUIRES_JEV_PRIMARY") from None
        try:
            call.validate_shape()
            serialized_call = call.model_dump_json()
            if len(serialized_call.encode("utf-8")) > 2_000_000:
                raise ValueError("MODEL_REQUEST_SIZE_LIMIT")
            self._secrets.validate_text(serialized_call)
        except (ValueError, ValidationError):
            raise ModelAttemptDenied("MODEL_INPUT_POLICY_BLOCKED") from None
        candidates = self._route(task, call)
        if not candidates:
            raise ModelRouteUnavailable("NO_CAPABILITY_COMPATIBLE_PROVIDER")
        attempt_limit = min(task.attempt_limit, self._max_attempts)
        execution_id = new_id("MOD")
        started = self._now()
        attempt_ids: list[str] = []
        usage_tokens: list[TokenUsage] = []
        usage_charges: list[FinancialCharge] = []
        last_error: ModelErrorCode = "PROVIDER_OUTAGE"
        selected_provider: ProviderAdapter | None = None
        selected_permit: ModelDispatchPermit | None = None
        last_provider_id = task.preferred_provider
        selected_output: ProviderOutput | None = None
        validated_output: bytes | None = None
        attempt_no = 0

        for adapter in candidates:
            if attempt_no >= attempt_limit:
                break
            try:
                remaining_cooldown = await self._authority.cooldown_remaining(
                    adapter.profile.provider_id
                )
            except Exception:
                raise ModelAttemptDenied("MODEL_COOLDOWN_STATE_UNAVAILABLE") from None
            if remaining_cooldown > 0:
                last_error = "RATE_LIMIT"
                continue
            same_provider_attempt = 0
            while attempt_no < attempt_limit:
                attempt_no += 1
                same_provider_attempt += 1
                try:
                    reserved = await self._authority.reserve(task, adapter.profile, attempt_no)
                except ModelAttemptDenied:
                    raise
                except Exception:
                    raise ModelAttemptDenied("MODEL_ATTEMPT_RESERVATION_FAILED") from None
                permit = reserved.permit
                if (
                    permit.task_id != task.task_id
                    or permit.run_id != task.run_id
                    or permit.provider_id != adapter.profile.provider_id
                    or permit.reservation_id != task.reservation_id
                ):
                    raise ModelAttemptDenied("MODEL_ATTEMPT_PERMIT_CONTEXT_MISMATCH")
                attempt_ids.append(permit.attempt_id)
                last_provider_id = adapter.profile.provider_id
                try:
                    output = await adapter.invoke(call, permit)
                except ProviderFailure as exc:
                    last_error = exc.error_code
                    unknown_tokens = TokenUsage(state="UNKNOWN_USAGE")
                    unknown_charge = FinancialCharge(state="UNKNOWN", usd=None)
                    await self._settle_failure(
                        task, permit, unknown_tokens, unknown_charge, exc.error_code
                    )
                    usage_tokens.append(unknown_tokens)
                    usage_charges.append(unknown_charge)
                    if exc.error_code == "RATE_LIMIT":
                        delay = self._bounded_delay(exc.retry_after_seconds, same_provider_attempt)
                        try:
                            await self._authority.impose_cooldown(
                                adapter.profile.provider_id, delay
                            )
                        except Exception:
                            raise ModelAttemptDenied("MODEL_COOLDOWN_COMMIT_FAILED") from None
                        break
                    if exc.error_code not in _RETRY_ERRORS:
                        break
                    if same_provider_attempt >= task.attempt_limit or attempt_no >= attempt_limit:
                        break
                    await self._pause(self._bounded_delay(None, same_provider_attempt))
                    continue
                except Exception:
                    # Unexpected adapter exceptions may follow a remote dispatch; retain uncertainty.
                    await self._settle_failure(
                        task,
                        permit,
                        TokenUsage(state="UNKNOWN_USAGE"),
                        FinancialCharge(state="UNKNOWN", usd=None),
                        "PROVIDER_OUTAGE",
                    )
                    usage_tokens.append(TokenUsage(state="UNKNOWN_USAGE"))
                    usage_charges.append(FinancialCharge(state="UNKNOWN", usd=None))
                    last_error = "PROVIDER_OUTAGE"
                    break
                try:
                    encoded = await self._validated_output(call, request, output)
                except ProviderFailure as exc:
                    last_error = exc.error_code
                    await self._settle_failure(
                        task, permit, output.tokens, output.charge, exc.error_code
                    )
                    usage_tokens.append(output.tokens)
                    usage_charges.append(output.charge)
                    break
                usage_tokens.append(output.tokens)
                usage_charges.append(output.charge)
                selected_provider = adapter
                selected_permit = permit
                selected_output = output
                validated_output = encoded
                break
            if selected_output is not None:
                break
            if last_error not in _FALLBACK_ERRORS:
                break

        if (
            selected_provider is None
            or selected_permit is None
            or selected_output is None
            or validated_output is None
        ):
            return self._failed_result(
                task,
                execution_id,
                tuple(attempt_ids),
                last_provider_id,
                call.prompt_version,
                last_error,
                usage_tokens,
                usage_charges,
                self._elapsed(started),
            )

        try:
            self._secrets.validate_text(validated_output.decode("utf-8"))
            output_ref = await self._publisher.publish(
                task,
                execution_id,
                validated_output,
                "application/json",
                permit=selected_permit,
                attempt_tokens=selected_output.tokens,
                attempt_charge=selected_output.charge,
                provider_id=selected_provider.profile.provider_id,
                model_id=selected_output.model_id,
                prompt_version=call.prompt_version,
                tokens=self._aggregate_tokens(usage_tokens),
                charge=self._aggregate_charges(usage_charges),
                attempt_ids=tuple(attempt_ids),
            )
        except Exception:
            raise ModelOutputPersistenceError("MODEL_OUTPUT_COMMIT_FAILED") from None
        return ModelExecutionResult(
            model_execution_id=execution_id,
            run_id=task.run_id,
            created_at_utc=datetime.now(UTC),
            task_id=task.task_id,
            attempt_ids=tuple(attempt_ids),
            provider_id=selected_provider.profile.provider_id,
            model_id=selected_output.model_id,
            model_version=selected_output.model_version,
            prompt_version=call.prompt_version,
            status="SUCCESS",
            output_ref=output_ref,
            decision=selected_output.decision,
            tokens=self._aggregate_tokens(usage_tokens),
            charge=self._aggregate_charges(usage_charges),
            latency_ms=self._elapsed(started),
            fallback_used=selected_provider.profile.provider_id != task.preferred_provider,
        )

    def _route(self, task: ModelTask, call: ModelCall) -> list[ProviderAdapter]:
        candidates: list[ProviderAdapter] = []
        for provider_id in (task.preferred_provider, *task.allowed_fallbacks):
            adapter = self._providers.get(provider_id)
            if adapter is None or adapter in candidates:
                continue
            if adapter.profile.supports(task, call):
                candidates.append(adapter)
        return candidates

    async def _validated_output(
        self, call: ModelCall, executable: ExecutableModelCall, output: ProviderOutput
    ) -> bytes:
        contract = executable.contract
        if contract.operation == "DECISION":
            if output.decision is None:
                raise ProviderFailure("SCHEMA_INVALID")
            encoded = output.decision.model_dump_json().encode("utf-8")
        else:
            if output.structured_text is None or executable.validator is None:
                raise ProviderFailure("SCHEMA_INVALID")
            try:
                parsed = json.loads(output.structured_text, object_pairs_hook=unique_object)
                validated = executable.validator(output.structured_text)
            except Exception:
                raise ProviderFailure("SCHEMA_INVALID") from None
            try:
                if inspect.isawaitable(validated):
                    validated = await validated
            except Exception:
                raise ProviderFailure("SCHEMA_INVALID") from None
            if hasattr(validated, "model_dump_json"):
                encoded = cast(bytes, validated.model_dump_json().encode("utf-8"))
            else:
                encoded = json.dumps(
                    validated if validated is not None else parsed,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
        return encoded

    def _failed_result(
        self,
        task: ModelTask,
        execution_id: str,
        attempt_ids: tuple[str, ...],
        provider_id: str,
        prompt_version: str,
        error_code: ModelErrorCode,
        tokens: list[TokenUsage],
        charges: list[FinancialCharge],
        latency_ms: int,
    ) -> ModelExecutionResult:
        if not attempt_ids:
            raise ModelRouteUnavailable("NO_MODEL_ATTEMPT_WAS_RESERVED")
        return ModelExecutionResult(
            model_execution_id=execution_id,
            run_id=task.run_id,
            created_at_utc=datetime.now(UTC),
            task_id=task.task_id,
            attempt_ids=attempt_ids,
            provider_id=provider_id,
            model_id=None,
            model_version=None,
            prompt_version=prompt_version,
            status="FAILED",
            error_code=error_code,
            tokens=self._aggregate_tokens(tokens),
            charge=self._aggregate_charges(charges),
            latency_ms=latency_ms,
            fallback_used=False,
        )

    def _bounded_delay(self, requested: float | None, retry_number: int) -> float:
        default = min(self._max_retry, self._base_retry * 2 ** max(0, retry_number - 1))
        value = default if requested is None else requested
        return max(0.0, min(self._max_retry, value))

    async def _pause(self, seconds: float) -> None:
        await self._sleep(seconds)

    def _now(self) -> float:
        return self._monotonic()

    def _elapsed(self, started: float) -> int:
        return max(0, int((self._now() - started) * 1000))

    async def _settle_failure(
        self,
        task: ModelTask,
        permit: ModelDispatchPermit,
        tokens: TokenUsage,
        charge: FinancialCharge,
        error_code: ModelErrorCode,
    ) -> None:
        try:
            await self._authority.settle_failure(task, permit, tokens, charge, error_code)
        except Exception:
            raise ModelAttemptDenied("MODEL_USAGE_SETTLEMENT_FAILED") from None

    @staticmethod
    def _aggregate_tokens(values: list[TokenUsage]) -> TokenUsage:
        if not values or any(value.state == "UNKNOWN_USAGE" for value in values):
            return TokenUsage(state="UNKNOWN_USAGE")
        return TokenUsage(
            state="KNOWN",
            input_tokens=sum(value.input_tokens or 0 for value in values),
            output_tokens=sum(value.output_tokens or 0 for value in values),
        )

    @staticmethod
    def _aggregate_charges(values: list[FinancialCharge]) -> FinancialCharge:
        if not values or any(value.state == "UNKNOWN" for value in values):
            return FinancialCharge(state="UNKNOWN", usd=None)
        if any(value.state == "ESTIMATED" for value in values):
            return FinancialCharge(
                state="ESTIMATED",
                usd=sum((value.usd or Decimal(0) for value in values), Decimal(0)),
            )
        return FinancialCharge(
            state="KNOWN", usd=sum((value.usd or Decimal(0) for value in values), Decimal(0))
        )
