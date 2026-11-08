"""Literal transition graphs from SRS §10; authorization is a separate gate."""

from collections.abc import Mapping
from types import MappingProxyType


def graph(rows: dict[str, str]) -> Mapping[str, frozenset[str]]:
    return MappingProxyType({key: frozenset(value.split()) for key, value in rows.items()})


RUN = graph(
    {
        "CREATED": "PLANNING CANCELLED",
        "PLANNING": "PLAN_READY FAILED CANCELLED",
        "PLAN_READY": "PLANNING RUNNING CANCELLED",
        "RUNNING": "PAUSING COMPLETING FAILED INTERRUPTED",
        "PAUSING": "PAUSED FAILED INTERRUPTED",
        "PAUSED": "RECOVERING CANCELLED",
        "INTERRUPTED": "RECOVERING FAILED",
        "RECOVERING": "RUNNING PAUSED FAILED",
        "COMPLETING": "COMPLETED PAUSED FAILED",
        "COMPLETED": "",
        "FAILED": "",
        "CANCELLED": "",
    }
)
TASK = graph(
    {
        "PENDING": "WAITING_DEPENDENCY READY AWAITING_APPROVAL BLOCKED CANCELLED",
        "WAITING_DEPENDENCY": "READY BLOCKED CANCELLED",
        "READY": "RESERVED DEFERRED AWAITING_APPROVAL BLOCKED CANCELLED",
        "RESERVED": "RUNNING READY RETRY_WAIT",
        "RUNNING": "SUCCEEDED RETRY_WAIT DEFERRED BLOCKED FAILED CANCELLED",
        "RETRY_WAIT": "READY DEFERRED FAILED CANCELLED",
        "DEFERRED": "READY AWAITING_APPROVAL BLOCKED CANCELLED",
        "AWAITING_APPROVAL": "READY CANCELLED BLOCKED",
        "BLOCKED": "READY CANCELLED",
        "SUCCEEDED": "",
        "FAILED": "READY",
        "CANCELLED": "",
    }
)
LEAD = graph(
    {
        "DISCOVERED": "TRIAGED",
        "TRIAGED": "QUEUED DEFERRED AWAITING_APPROVAL REJECTED_WITH_REASON",
        "QUEUED": "INVESTIGATING",
        "INVESTIGATING": "RESOLVED DEFERRED BLOCKED",
        "DEFERRED": "QUEUED",
        "AWAITING_APPROVAL": "",
        "REJECTED_WITH_REASON": "",
        "RESOLVED": "",
        "BLOCKED": "",
    }
)
PROPOSAL = graph(
    {
        "PROPOSED": "VALIDATED",
        "VALIDATED": "AUTO_AUTHORIZED AWAITING_APPROVAL REJECTED DUPLICATE_SUPPRESSED",
        "AUTO_AUTHORIZED": "SCHEDULED",
        "SCHEDULED": "APPLIED FAILED_APPLICATION",
        "AWAITING_APPROVAL": "USER_APPROVED USER_REJECTED EXPIRED",
        "USER_APPROVED": "SCHEDULED",
        "REJECTED": "",
        "DUPLICATE_SUPPRESSED": "",
        "APPLIED": "",
        "FAILED_APPLICATION": "",
        "USER_REJECTED": "",
        "EXPIRED": "",
    }
)
ASSESSMENT = graph(
    {
        "DRAFT": "VALIDATING",
        "VALIDATING": "COMMITTED REJECTED_NEEDS_EVIDENCE CONFLICT_REVIEW",
        "CONFLICT_REVIEW": "VALIDATING",
        "COMMITTED": "",
        "REJECTED_NEEDS_EVIDENCE": "",
    }
)
PLAN_CHANGE = graph(
    {"PROPOSED": "REVIEWED", "REVIEWED": "APPROVED REJECTED", "APPROVED": "", "REJECTED": ""}
)
GRAPHS = MappingProxyType(
    {
        "RUN": RUN,
        "TASK": TASK,
        "LEAD": LEAD,
        "PROPOSAL": PROPOSAL,
        "ASSESSMENT": ASSESSMENT,
        "PLAN_CHANGE": PLAN_CHANGE,
    }
)


def allowed(kind: str, before: str, after: str) -> bool:
    return after in GRAPHS.get(kind, {}).get(before, frozenset())
