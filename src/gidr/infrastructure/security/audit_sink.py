"""Persist only sanitized security observations through the canonical audit writer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from gidr.contracts.common import (
    AttemptId,
    Contract,
    CredentialAlias,
    Digest,
    GrantId,
    Name,
    Nonnegative,
    Positive,
    RunId,
    SourceId,
    TaskId,
    new_id,
)
from gidr.contracts.control import AuditEvent, SecurityEventDetail
from gidr.infrastructure.db.authority import PersistenceAuthority, WriteContext
from gidr.infrastructure.security.events import SecurityCategory, SecurityObservation

MimeType = Annotated[str, StringConstraints(pattern=r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")]


class SecurityAuditCommand(Contract):
    category: SecurityCategory
    reason_code: Name
    request_id: Name
    source_id: SourceId | None = None
    url_sha256: Digest | None = None
    content_sha256: Digest | None = None
    status_code: Annotated[int, Field(ge=0, le=599)] | None = None
    byte_count: Nonnegative | None = None
    mime: MimeType | None = None
    credential_alias: CredentialAlias | None = None
    run_id: RunId | None = None
    plan_version: Positive | None = None
    grant_id: GrantId | None = None
    provider_id: Name | None = None
    task_id: TaskId | None = None
    attempt_id: AttemptId | None = None


class SecurityAuditReceipt(Contract):
    status: Literal["RECORDED"]


class SecurityAuditFailure(RuntimeError):
    def __init__(self) -> None:
        super().__init__("SECURITY_AUDIT_PERSISTENCE_FAILED")


class SQLiteSecurityEventSink:
    """An event must commit before a caller proceeds to external I/O or parsing."""

    def __init__(self, authority: PersistenceAuthority) -> None:
        self._authority = authority

    def record_security(self, observation: SecurityObservation) -> None:
        try:
            command = SecurityAuditCommand.model_validate(
                {
                    field: getattr(observation, field)
                    for field in SecurityAuditCommand.model_fields
                    if hasattr(observation, field)
                }
            )
            command_id = new_id("CMD")
            trace_id = new_id("TRACE")
            timestamp = datetime.now(UTC)
            event = AuditEvent(
                run_id=command.run_id,
                created_at_utc=timestamp,
                actor="SYSTEM",
                event_id=new_id("EVT"),
                trace_id=trace_id,
                command_id=command_id,
                session_id=None,
                task_id=command.task_id,
                attempt_id=command.attempt_id,
                plan_version=command.plan_version,
                grant_id=command.grant_id,
                aggregate_type="SECURITY",
                aggregate_id=command.request_id,
                revision_before=None,
                revision_after=None,
                old_state=None,
                new_state=None,
                component="security",
                event_type=command.category,
                cause="SECURITY_POLICY",
                policy_version="1.0",
                security=SecurityEventDetail(
                    category=command.category,
                    request_id=command.request_id,
                    source_id=command.source_id,
                    url_sha256=command.url_sha256,
                    content_sha256=command.content_sha256,
                    status_code=command.status_code,
                    byte_count=command.byte_count,
                    mime=command.mime,
                    credential_alias=command.credential_alias,
                    provider_id=command.provider_id,
                ),
                reason_code=command.reason_code,
                occurred_at_utc=timestamp,
            )

            def persist(context: WriteContext) -> SecurityAuditReceipt:
                context.emit(event)
                return SecurityAuditReceipt(status="RECORDED")

            self._authority.execute(
                command_id,
                trace_id,
                command,
                SecurityAuditReceipt,
                persist,
            )
        except Exception:
            raise SecurityAuditFailure() from None
