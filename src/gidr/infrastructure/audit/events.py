"""Factories create proposals; only PersistenceAuthority commits them."""

from datetime import UTC, datetime

from gidr.contracts.common import new_id
from gidr.contracts.control import AuditEvent
from gidr.infrastructure.db.authority import WriteContext


def event_for(
    context: WriteContext,
    *,
    component: str,
    event_type: str,
    aggregate_id: str,
    run_id: str | None = None,
) -> AuditEvent:
    timestamp = datetime.now(UTC)
    return AuditEvent(
        run_id=run_id,
        created_at_utc=timestamp,
        event_id=new_id("EVT"),
        trace_id=context.trace_id,
        command_id=context.command_id,
        session_id=None,
        task_id=None,
        attempt_id=None,
        plan_version=None,
        grant_id=None,
        aggregate_type=component.upper(),
        aggregate_id=aggregate_id,
        revision_before=None,
        revision_after=None,
        old_state=None,
        new_state=None,
        component=component,
        event_type=event_type,
        cause="PERSISTENCE_COMMAND",
        policy_version="1.0",
        reason_code="LOCAL_COMMIT",
        occurred_at_utc=timestamp,
    )
