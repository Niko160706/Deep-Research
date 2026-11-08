"""Vendor-neutral model outputs; provider SDK mapping is a later integration."""

from __future__ import annotations

from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from gidr.contracts.common import (
    ArtifactRef,
    AttemptId,
    ChunkId,
    Contract,
    DecisionId,
    Fraction,
    Language,
    ModelExecutionId,
    Name,
    Nonnegative,
    Positive,
    Record,
    TaskId,
    Text,
    TranslationId,
)

Money = Annotated[Decimal, Field(ge=0, allow_inf_nan=False)]


class TokenUsage(Contract):
    state: Literal["KNOWN", "ESTIMATED", "UNKNOWN_USAGE"] = "UNKNOWN_USAGE"
    input_tokens: Nonnegative | None = None
    output_tokens: Nonnegative | None = None

    @model_validator(mode="after")
    def truthful_usage(self) -> Self:
        if self.state == "UNKNOWN_USAGE":
            if self.input_tokens is not None or self.output_tokens is not None:
                raise ValueError("Unknown tokens cannot be recorded as measured counts")
        elif self.input_tokens is None or self.output_tokens is None:
            raise ValueError("Known or estimated usage requires both token counts")
        return self


class FinancialCharge(Contract):
    state: Literal["KNOWN", "ESTIMATED", "UNKNOWN"] = "UNKNOWN"
    usd: Money | None = None

    @model_validator(mode="after")
    def truthful_cost(self) -> Self:
        if (self.state == "UNKNOWN") != (self.usd is None):
            raise ValueError("Unknown charges must remain null")
        return self


class DecisionConfidence(Contract):
    semantics: Literal["UNAVAILABLE", "PROVIDER_PROBABILITY", "PROVIDER_SCORE"] = "UNAVAILABLE"
    value: Fraction | None = None
    rubric_version: Text | None = None

    @model_validator(mode="after")
    def truthful_confidence(self) -> Self:
        if (self.semantics == "UNAVAILABLE") != (self.value is None):
            raise ValueError("Unavailable confidence must remain null")
        if self.semantics == "PROVIDER_SCORE" and self.rubric_version is None:
            raise ValueError("Provider score requires its rubric semantics")
        return self


class ChoiceDecision(Contract):
    kind: Literal["CHOICE"] = "CHOICE"
    choices: Annotated[tuple[Text, ...], Field(min_length=2)]
    selected: Text
    confidence: DecisionConfidence = Field(default_factory=DecisionConfidence)

    @model_validator(mode="after")
    def valid_choice(self) -> Self:
        if self.selected not in self.choices or len(set(self.choices)) != len(self.choices):
            raise ValueError("Selected answer must be one of the distinct declared choices")
        return self


class ScoreDecision(Contract):
    kind: Literal["SCORE"] = "SCORE"
    score: Fraction
    rubric_version: Text
    confidence: DecisionConfidence = Field(default_factory=DecisionConfidence)


TypedDecision = Annotated[ChoiceDecision | ScoreDecision, Field(discriminator="kind")]


class DecisionRecord(Record):
    decision_id: DecisionId
    task_id: TaskId
    model_execution_id: ModelExecutionId
    input_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    decision: TypedDecision
    policy_outcome: Literal["PROPOSAL_ONLY"] = "PROPOSAL_ONLY"
    reason_code: Name


class ModelExecutionResult(Record):
    model_execution_id: ModelExecutionId
    task_id: TaskId
    attempt_ids: Annotated[tuple[AttemptId, ...], Field(min_length=1)]
    provider_id: Name
    model_id: Text | None
    model_version: Text | None
    prompt_version: Text
    output_schema_version: Literal["1.0"] = "1.0"
    status: Literal["SUCCESS", "FAILED", "DEFERRED"]
    output_ref: ArtifactRef | None = None
    decision: TypedDecision | None = None
    error_code: (
        Literal[
            "TIMEOUT",
            "RATE_LIMIT",
            "AUTH",
            "SCHEMA_INVALID",
            "CONTEXT_TOO_LONG",
            "LANGUAGE_UNSUPPORTED",
            "POLICY_BLOCK",
            "BUDGET_EXCEEDED",
            "PROVIDER_OUTAGE",
        ]
        | None
    ) = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    charge: FinancialCharge = Field(default_factory=FinancialCharge)
    latency_ms: Nonnegative | None = None
    fallback_used: bool = False

    @model_validator(mode="after")
    def valid_outcome(self) -> Self:
        if self.status == "SUCCESS":
            if self.output_ref is None or self.model_id is None or self.error_code is not None:
                raise ValueError(
                    "Successful model call requires model/output provenance and no error"
                )
        elif self.error_code is None or self.output_ref is not None or self.decision is not None:
            raise ValueError(
                "Failed/deferred call requires classified error, without successful output"
            )
        if len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("Attempt IDs cannot repeat")
        return self


class TranslationArtifact(Record):
    translation_id: TranslationId
    chunk_id: ChunkId
    original_text_ref: ArtifactRef
    source_language: Language
    target_language: Language
    translator_provider: Name
    translator_model: Text
    prompt_version: Text
    translated_artifact: ArtifactRef
    risk_flags: tuple[
        Literal["NEGATION", "MODALITY", "NUMERALS", "UNITS", "ENTITY", "UNCERTAINTY", "TENSE"], ...
    ] = ()
    independent_evidence: Literal[False] = False


class ModelTask(Record):
    task_id: TaskId
    plan_version: Positive | None
    authorization_ref: Text
    task_type: Name
    required_capabilities: Annotated[tuple[Name, ...], Field(min_length=1)]
    input_refs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    input_language: Language
    risk_class: Literal["LOW", "MEDIUM", "HIGH"]
    output_schema_version: Literal["1.0"] = "1.0"
    reservation_id: Text
    attempt_limit: Positive
    preferred_provider: Name
    allowed_fallbacks: tuple[Name, ...] = ()
    prompt_version: Text
    privacy_scope: Literal["PUBLIC_SOURCE", "APPROVED_PRIVATE"]
