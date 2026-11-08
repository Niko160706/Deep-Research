"""Strict internal command payloads. External/model content cannot mint authority."""

from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from gidr.contracts.common import (
    ArtifactRef,
    AttemptId,
    Contract,
    GrantId,
    Name,
    Nonnegative,
    Positive,
    ProjectId,
    ProposalId,
    QuestionId,
    SessionId,
    TaskId,
    Text,
    UTCDateTime,
)
from gidr.contracts.control import TaskCommand, TaskSpecification

Money = Annotated[Decimal, Field(ge=0, le=10**12, max_digits=24, decimal_places=9)]

Actor = Literal["USER", "SCHEDULER", "WORKER", "SYSTEM"]
TTL = Annotated[int, Field(ge=1, le=3600)]
BoundedNames = Annotated[tuple[Name, ...], Field(max_length=128)]


class GrantPolicy(Contract):
    grant_id: GrantId
    providers: BoundedNames
    categories: BoundedNames
    task_types: BoundedNames
    max_cost_usd: Money
    max_tokens: Annotated[int, Field(ge=0, le=10**12)]
    max_requests: Annotated[int, Field(ge=0, le=10**9)]
    max_attempts: Annotated[int, Field(ge=1, le=100)] = 3
    expires_at_utc: UTCDateTime | None = None


class CreateRun(Contract):
    action: Literal["CREATE_RUN"] = "CREATE_RUN"
    project_id: ProjectId
    session_id: SessionId
    title: Annotated[Text, Field(max_length=4096)]
    owner: Name


class ChangeRun(Contract):
    action: Literal["CHANGE_RUN"] = "CHANGE_RUN"
    target: Name
    reason: Name
    checkpoint_ref: ArtifactRef | None = None


class PublishPlan(Contract):
    action: Literal["PUBLISH_PLAN"] = "PUBLISH_PLAN"
    snapshot_ref: ArtifactRef
    plan_id: Name
    plan_version_id: Name
    change_proposal_id: ProposalId | None = None
    grant: GrantPolicy | None = None


class StartRun(Contract):
    action: Literal["START_RUN"] = "START_RUN"
    grant: GrantPolicy
    lease_ttl_seconds: TTL = 120


class ResumeRun(Contract):
    action: Literal["RESUME_RUN"] = "RESUME_RUN"
    lease_ttl_seconds: TTL = 120


class RenewRun(Contract):
    action: Literal["RENEW_RUN"] = "RENEW_RUN"
    generation: Nonnegative
    lease_ttl_seconds: TTL = 120


class SwitchSession(Contract):
    action: Literal["SWITCH_SESSION"] = "SWITCH_SESSION"
    session_id: SessionId


class OpenSession(Contract):
    action: Literal["OPEN_SESSION"] = "OPEN_SESSION"
    session_id: SessionId


class RevokeGrant(Contract):
    action: Literal["REVOKE_GRANT"] = "REVOKE_GRANT"
    reason: Name


class ApproveGrant(Contract):
    action: Literal["APPROVE_GRANT"] = "APPROVE_GRANT"
    grant: GrantPolicy


class ReleaseReservation(Contract):
    action: Literal["RELEASE_RESERVATION"] = "RELEASE_RESERVATION"
    token: "AttemptToken"
    reason: Name


class ScheduledTask(TaskSpecification):
    provider_id: Name = "local"
    category: Name = "local"
    work_kind: Literal["LOCAL", "PROVIDER"] = "LOCAL"
    max_cost_usd: Money = Decimal(0)
    max_tokens: Annotated[int, Field(ge=0, le=10**12)] = 0
    max_requests: Annotated[int, Field(ge=0, le=10**9)] = 0
    max_attempts: Annotated[int, Field(ge=1, le=100)] = 3


class AddTasks(Contract):
    action: Literal["ADD_TASKS"] = "ADD_TASKS"
    tasks: Annotated[tuple[ScheduledTask, ...], Field(min_length=1, max_length=256)]


class ChangeTask(Contract):
    action: Literal["CHANGE_TASK"] = "CHANGE_TASK"
    task_id: TaskId
    target: Name
    reason: Name
    condition_change_ref: ArtifactRef | None = None
    not_before_utc: UTCDateTime | None = None


class ReserveTask(Contract):
    action: Literal["RESERVE_TASK"] = "RESERVE_TASK"
    task_id: TaskId
    attempt_id: AttemptId
    worker_id: Name
    lease_ttl_seconds: TTL = 120


class AttemptToken(Contract):
    task_id: TaskId
    attempt_id: AttemptId
    holder: Name
    generation: Positive
    gate_generation: Positive


class StartAttempt(Contract):
    action: Literal["START_ATTEMPT"] = "START_ATTEMPT"
    token: AttemptToken


class RenewAttempt(Contract):
    action: Literal["RENEW_ATTEMPT"] = "RENEW_ATTEMPT"
    token: AttemptToken
    lease_ttl_seconds: TTL = 120


class Usage(Contract):
    state: Literal["KNOWN", "ESTIMATED", "UNKNOWN_USAGE"]
    cost_usd: Money | None = None
    tokens: Nonnegative | None = None
    requests: Nonnegative | None = None

    @model_validator(mode="after")
    def uncertainty_is_explicit(self) -> Self:
        values = (self.cost_usd, self.tokens, self.requests)
        if self.state == "KNOWN" and any(value is None for value in values):
            raise ValueError("Known usage requires every measured quantity")
        if self.state == "UNKNOWN_USAGE" and any(value is not None for value in values):
            raise ValueError("Unknown usage cannot invent measurements")
        if self.state == "ESTIMATED" and all(value is None for value in values):
            raise ValueError("Estimated usage requires an observation")
        return self


class FinishAttempt(Contract):
    action: Literal["FINISH_ATTEMPT"] = "FINISH_ATTEMPT"
    token: AttemptToken
    target: Literal["SUCCEEDED", "RETRY_WAIT", "DEFERRED", "BLOCKED", "FAILED", "CANCELLED"]
    usage: Usage
    reason: Name
    result_ref: ArtifactRef | None = None
    safe_cancellation: bool = False
    not_before_utc: UTCDateTime | None = None

    @model_validator(mode="after")
    def success_has_output(self) -> Self:
        if (self.target == "SUCCEEDED") != (self.result_ref is not None):
            raise ValueError("Only success carries a validated result reference")
        return self


class ReconcileUsage(Contract):
    action: Literal["RECONCILE_USAGE"] = "RECONCILE_USAGE"
    attempt_id: AttemptId
    usage: Usage
    observation_ref: ArtifactRef


class RecoverRun(Contract):
    action: Literal["RECOVER_RUN"] = "RECOVER_RUN"


class CreateWorkflow(Contract):
    action: Literal["CREATE_WORKFLOW"] = "CREATE_WORKFLOW"
    kind: Literal["LEAD", "PROPOSAL", "PLAN_CHANGE"]
    entity_id: Name
    snapshot_ref: ArtifactRef | None = None
    question_id: QuestionId | None = None
    fingerprint: Name = "initial"
    material_version: Name = "initial"
    reason: Name


class ChangeWorkflow(Contract):
    action: Literal["CHANGE_WORKFLOW"] = "CHANGE_WORKFLOW"
    kind: Literal["LEAD", "PROPOSAL", "PLAN_CHANGE", "ASSESSMENT"]
    entity_id: Name
    target: Name
    reason: Name
    condition_change_ref: ArtifactRef | None = None


class ProposalScope(Contract):
    """Control-plane eligibility only, never epistemic truth or an ESC result."""

    providers: BoundedNames
    categories: BoundedNames
    task_types: BoundedNames
    estimated_cost_usd: Money
    estimated_tokens: Nonnegative
    estimated_requests: Nonnegative


type Payload = (
    CreateRun
    | ChangeRun
    | PublishPlan
    | StartRun
    | ResumeRun
    | RenewRun
    | SwitchSession
    | OpenSession
    | RevokeGrant
    | ApproveGrant
    | ReleaseReservation
    | AddTasks
    | ChangeTask
    | ReserveTask
    | StartAttempt
    | RenewAttempt
    | FinishAttempt
    | ReconcileUsage
    | RecoverRun
    | CreateWorkflow
    | ChangeWorkflow
)


class StateCommand(TaskCommand[Annotated[Payload, Field(discriminator="action")]]):
    @model_validator(mode="after")
    def matching_action(self) -> Self:
        if self.command_type != self.payload.action:
            raise ValueError("Command type and payload action must agree")
        return self


class CommandResult(Contract):
    accepted: bool
    code: Name
    state: Name | None = None
    revision: Nonnegative | None = None
    token: AttemptToken | None = None


class StateCheckpoint(Contract):
    run_id: str
    revision: Nonnegative
    event_seq: Nonnegative
    gate_generation: Nonnegative
    plan_version: Nonnegative
    created_at_utc: UTCDateTime


def expiry(now: datetime, value: datetime | None) -> bool:
    return value is not None and value <= now
