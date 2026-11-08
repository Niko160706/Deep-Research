"""Project/session/run identities remain distinct before persistence is added."""

from typing import Literal, Self

from pydantic import model_validator

from gidr.contracts.common import (
    Contract,
    Name,
    Nonnegative,
    Positive,
    ProjectId,
    RunId,
    SessionId,
    Text,
    UTCDateTime,
)
from gidr.domain.states import RunState


class Project(Contract):
    project_id: ProjectId
    title: Text
    owner: Name
    created_at_utc: UTCDateTime


class Session(Contract):
    session_id: SessionId
    project_id: ProjectId
    created_at_utc: UTCDateTime
    opened_at_utc: UTCDateTime | None = None


class ResearchRun(Contract):
    run_id: RunId
    project_id: ProjectId
    initiating_session_id: SessionId
    created_at_utc: UTCDateTime
    state: RunState = RunState.CREATED
    revision: Nonnegative = 0
    plan_version: Positive | None = None
    operation: Literal["INITIAL", "EXTEND", "INVESTIGATE", "VERIFY", "REFRESH"]
    parent_run_ids: tuple[RunId, ...] = ()
    stop_reason: Name | None = None

    @model_validator(mode="after")
    def no_self_parent(self) -> Self:
        if self.run_id in self.parent_run_ids:
            raise ValueError("A run cannot be its own parent")
        if len(set(self.parent_run_ids)) != len(self.parent_run_ids):
            raise ValueError("Parent run IDs cannot repeat")
        return self
