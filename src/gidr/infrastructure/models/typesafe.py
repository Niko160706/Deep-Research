"""Official TypeSafe SDK adapter. Retries and authorization belong to the gateway."""

from __future__ import annotations

import math
from typing import Literal

from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    RetryPolicy,
    Score,
    ScoreAnswer,
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from gidr.contracts.models import (
    ChoiceDecision,
    DecisionConfidence,
    FinancialCharge,
    NoulDecision,
    ScoreDecision,
    TokenUsage,
    TypedDecision,
)
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.ports import (
    DecisionQuestion,
    ModelCall,
    ModelDispatchPermit,
    ProviderFailure,
    ProviderOutput,
    ProviderProfile,
)
from gidr.infrastructure.models.transport import AuthorizedModelTransport, ModelTransportDenied
from gidr.infrastructure.security.secrets import SecretManager, SecretUnavailable

_NO_RETRIES = RetryPolicy(max_retries=0)


class TypeSafeJevProvider:
    """Adapter for official `typesafe-sdk`; requests are dispatched through a permit."""

    def __init__(
        self,
        profile: ProviderProfile,
        *,
        secret_manager: SecretManager,
        secret_policy: SecretPolicy,
        origin: str = "https://api.typesafe.ai",
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 1_000_000,
    ) -> None:
        if profile.provider_id != "jev":
            raise ValueError("TYPESAFE_PROFILE_PROVIDER_MISMATCH")
        if not 0 < timeout_seconds <= 120 or max_response_bytes < 1:
            raise ValueError("TYPESAFE_LIMIT_INVALID")
        self.profile = profile
        self._secrets = secret_manager
        self._secret_policy = secret_policy
        self._origin = origin
        self._timeout = timeout_seconds
        self._max_response_bytes = max_response_bytes

    async def invoke(self, call: ModelCall, permit: ModelDispatchPermit) -> ProviderOutput:
        if call.operation != "DECISION" or call.question is None:
            raise ProviderFailure("POLICY_BLOCK")
        if permit.provider_id != "jev" or permit.reservation_id == "":
            raise ProviderFailure("POLICY_BLOCK")
        try:
            handle = self._secrets.resolve("provider/typesafe")
            api_key = handle.authorization_value(self._origin)
        except SecretUnavailable:
            raise ProviderFailure("AUTH") from None
        question = call.question
        if question.kind == "NOUL":
            sdk_question: Question = Noul(
                instructions=question.instructions,
                criteria={
                    "true": question.positive_criteria,
                    "false": question.negative_criteria,
                },
            )
        elif question.kind == "CHOICE":
            sdk_question = Choice(
                instructions=question.instructions,
                criteria={choice: None for choice in question.choices},
            )
        else:
            sdk_question = Score(instructions=question.instructions, criteria=list(question.levels))
        try:
            transport = AuthorizedModelTransport(
                permit,
                allowed_origin=self._origin,
                allowed_path="/v1/systemone",
                secret_policy=self._secret_policy,
                max_response_bytes=self._max_response_bytes,
                timeout_seconds=self._timeout,
            )
            async with AsyncTypeSafeClient(
                api_key=api_key,
                model=self.profile.model_id,
                retry=_NO_RETRIES,
                timeout=self._timeout,
                transport=transport,
                base_url=self._origin,
            ) as client:
                result = await client.system_one(
                    state={
                        "source_text": call.content,
                        "trust_label": "UNTRUSTED_SOURCE_DATA",
                    },
                    questions={question.key: sdk_question},
                    model=self.profile.model_id,
                    retry=_NO_RETRIES,
                )
        except TypeSafeRateLimitError as exc:
            retry_after_ms = exc.retry_after_ms
            retry_seconds = (
                retry_after_ms / 1000
                if isinstance(retry_after_ms, (int, float))
                and not isinstance(retry_after_ms, bool)
                and math.isfinite(retry_after_ms)
                else None
            )
            raise ProviderFailure("RATE_LIMIT", retry_after_seconds=retry_seconds) from None
        except TypeSafeAuthenticationError:
            raise ProviderFailure("AUTH") from None
        except TypeSafeAPITimeoutError:
            raise ProviderFailure("TIMEOUT") from None
        except TypeSafeAPIResponseValidationError:
            raise ProviderFailure("SCHEMA_INVALID") from None
        except TypeSafeAPIConnectionError:
            raise ProviderFailure("PROVIDER_OUTAGE") from None
        except ProviderFailure:
            raise
        except ModelTransportDenied:
            raise ProviderFailure("POLICY_BLOCK") from None
        except Exception:
            # SDK/http errors can include endpoint, response body or user content.
            raise ProviderFailure("PROVIDER_OUTAGE") from None
        answer_root = result.answers.get(question.key)
        if answer_root is None:
            raise ProviderFailure("SCHEMA_INVALID")
        try:
            answer = answer_root
            decision = _map_answer(question, answer)
        except (KeyError, TypeError, ValueError):
            raise ProviderFailure("SCHEMA_INVALID") from None
        usage_state: Literal["KNOWN", "UNKNOWN_USAGE"] = (
            "KNOWN"
            if (result.usage.input_tokens is not None and result.usage.output_tokens is not None)
            else "UNKNOWN_USAGE"
        )
        tokens = TokenUsage(
            state=usage_state,
            input_tokens=result.usage.input_tokens if usage_state == "KNOWN" else None,
            output_tokens=result.usage.output_tokens if usage_state == "KNOWN" else None,
        )
        return ProviderOutput(
            model_id=result.model,
            model_version=None,
            decision=decision,
            tokens=tokens,
            charge=FinancialCharge(state="UNKNOWN", usd=None),
        )


def _map_answer(
    question: DecisionQuestion, answer: NoulAnswer | ChoiceAnswer | ScoreAnswer
) -> TypedDecision:
    if question.kind == "NOUL" and isinstance(answer, NoulAnswer):
        probability = float(answer.noul)
        return NoulDecision(
            probability=probability,
            confidence=DecisionConfidence(semantics="PROVIDER_PROBABILITY", value=probability),
        )
    if question.kind == "CHOICE" and isinstance(answer, ChoiceAnswer):
        selected = answer.choice
        if selected not in question.choices:
            raise ValueError("CHOICE_OUTSIDE_ENUM")
        probabilities = None
        if answer.probabilities is not None:
            probabilities = {str(key): float(value) for key, value in answer.probabilities.items()}
        confidence = DecisionConfidence(
            semantics="PROVIDER_SCORE",
            value=float(answer.confidence),
            rubric_version="typesafe-choice-confidence-v1",
        )
        return ChoiceDecision(
            choices=question.choices,
            selected=selected,
            confidence=confidence,
            probabilities=probabilities,
        )
    if question.kind == "SCORE" and isinstance(answer, ScoreAnswer):
        raw = float(answer.score)
        count = len(question.levels)
        if count < 2 or not 0 <= raw <= count - 1:
            raise ValueError("SCORE_OUTSIDE_RUBRIC")
        score_probabilities = {
            int(key): float(value) for key, value in answer.probabilities.items()
        }
        score_legend = {int(key): str(value) for key, value in answer.legend.items()}
        if tuple(score_legend[index] for index in sorted(score_legend)) != question.levels:
            raise ValueError("SCORE_LEGEND_MISMATCH")
        confidence = DecisionConfidence(
            semantics="PROVIDER_SCORE",
            value=float(answer.confidence),
            rubric_version=question.rubric_version,
        )
        return ScoreDecision(
            score=raw / (count - 1),
            raw_provider_score=raw,
            rubric_version=question.rubric_version or "unspecified",
            confidence=confidence,
            probabilities=score_probabilities,
            legend=score_legend,
        )
    raise ValueError("PROVIDER_ANSWER_KIND_MISMATCH")
