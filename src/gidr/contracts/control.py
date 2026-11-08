"""Command, plan and event envelopes without persistence or dispatch logic."""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from gidr.contracts.common import (
    ArtifactRef,
    AttemptId,
    CommandId,
    Contract,
    CredentialAlias,
    Digest,
    EventId,
    GrantId,
    Name,
    Nonnegative,
    Positive,
    ProjectId,
    ProposalId,
    QuestionId,
    Record,
    RunId,
    SessionId,
    SourceId,
    TaskId,
    Text,
    TraceId,
    UTCDateTime,
)
from gidr.domain.states import Methodology


class Question(Contract):
    question_id: QuestionId
    text: Text
    evidence_requirements: Annotated[tuple[Text, ...], Field(min_length=1)]
    methodology: Methodology = Methodology.AUTO
    dependencies: tuple[QuestionId, ...] = ()


class ResearchPlanProposal(Record):
    proposal_id: ProposalId
    project_id: ProjectId
    session_id: SessionId
    expected_plan_version: Nonnegative
    objective: Text
    scope_in: Annotated[tuple[Text, ...], Field(min_length=1)]
    scope_out: tuple[Text, ...] = ()
    questions: Annotated[tuple[Question, ...], Field(min_length=1)]
    source_categories: Annotated[tuple[Name, ...], Field(min_length=1)]
    budget_policy_ref: ArtifactRef
    status: Literal["PROPOSED"] = "PROPOSED"

    @model_validator(mode="after")
    def valid_questions(self) -> Self:
        by_id = {question.question_id: question for question in self.questions}
        if len(by_id) != len(self.questions):
            raise ValueError("Duplicate question ID")
        visited: set[str] = set()
        pending = set(by_id)
        for question in self.questions:
            if any(dep not in by_id for dep in question.dependencies):
                raise ValueError("Question dependency is missing")
        while pending:
            ready = {key for key in pending if set(by_id[key].dependencies) <= visited}
            if not ready:
                raise ValueError("Question dependencies form a cycle")
            pending -= ready
            visited |= ready
        return self


class TaskSpecification(Contract):
    task_id: TaskId
    question_id: QuestionId
    task_type: Name
    input_refs: tuple[ArtifactRef, ...] = ()
    dependencies: tuple[TaskId, ...] = ()
    reason_code: Name

    @model_validator(mode="after")
    def no_self_dependency(self) -> Self:
        if self.task_id in self.dependencies:
            raise ValueError("Task cannot depend on itself")
        return self


class ReservationPayload(Contract):
    task_id: TaskId
    attempt_id: AttemptId
    lease_ttl_seconds: Positive
    budget_policy_ref: ArtifactRef


class ResultPayload(Contract):
    task_id: TaskId
    attempt_id: AttemptId
    result_ref: ArtifactRef
    fencing_generation: Nonnegative


class TaskCommand[P: Contract](Contract):
    command_id: CommandId
    trace_id: TraceId
    run_id: RunId
    actor: Literal["USER", "SCHEDULER", "WORKER", "SYSTEM"]
    command_type: Name
    aggregate_id: Text
    expected_revision: Nonnegative
    plan_version: Positive
    grant_id: GrantId | None
    idempotency_key: Text
    payload: P
    created_at_utc: UTCDateTime


class SecurityEventDetail(Contract):
    """Minimal security provenance; never carries URLs, headers, keys or response text."""

    category: Name
    request_id: Text
    source_id: SourceId | None = None
    url_sha256: Digest | None = None
    content_sha256: Digest | None = None
    status_code: Nonnegative | None = None
    byte_count: Nonnegative | None = None
    mime: Text | None = None
    credential_alias: CredentialAlias | None = None
    provider_id: Name | None = None


class AuditEvent(Contract):
    # Global storage/migration events have no run; never invent a run ID.
    run_id: RunId | None
    created_at_utc: UTCDateTime
    actor: Literal["USER", "SCHEDULER", "WORKER", "SYSTEM"] | None = None
    seq: Positive | None = None
    committed_at_utc: UTCDateTime | None = None
    event_id: EventId
    trace_id: TraceId
    command_id: CommandId | None
    session_id: SessionId | None
    task_id: TaskId | None
    attempt_id: AttemptId | None
    plan_version: Positive | None
    grant_id: GrantId | None
    aggregate_type: Name
    aggregate_id: Text
    revision_before: Nonnegative | None
    revision_after: Nonnegative | None
    old_state: Name | None
    new_state: Name | None
    component: Name
    event_type: Name
    cause: Name
    policy_version: Text
    input_refs: tuple[ArtifactRef, ...] = ()
    output_refs: tuple[ArtifactRef, ...] = ()
    security: SecurityEventDetail | None = None
    reason_code: Name
    occurred_at_utc: UTCDateTime

    @model_validator(mode="after")
    def valid_revisions(self) -> Self:
        if self.revision_before is not None and self.revision_after is not None:
            if self.revision_after < self.revision_before:
                raise ValueError("Audit revisions cannot go backwards")
        return self
