"""Sanitized security observations suitable for audit sinks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

SecurityCategory = Literal[
    "OUTBOUND_REQUEST",
    "REDIRECT_ALLOWED",
    "UNSAFE_URL_BLOCKED",
    "REDIRECT_BLOCKED",
    "DNS_REBINDING_BLOCKED",
    "DISALLOWED_MIME",
    "QUARANTINE_SCAN_FLAG",
    "PARSER_SANDBOX_FAILURE",
    "PARSER_DISPATCH",
    "PARSER_COMPLETED",
    "PROMPT_INJECTION_SUSPECTED",
    "SECRET_ACCESS_DENIED",
    "CREDENTIAL_PROVIDER_CONFIG",
    "PROVIDER_AUTH_FAILURE",
    "UNAUTHORIZED_ACTION",
    "SUSPICIOUS_EGRESS",
]


@dataclass(frozen=True, slots=True)
class SecurityObservation:
    category: SecurityCategory
    reason_code: str
    request_id: str
    source_id: str | None = None
    url_sha256: str | None = None
    content_sha256: str | None = None
    status_code: int | None = None
    byte_count: int | None = None
    mime: str | None = None
    credential_alias: str | None = None
    run_id: str | None = None
    plan_version: int | None = None
    grant_id: str | None = None
    provider_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None


class SecurityEventSink(Protocol):
    def record_security(self, observation: SecurityObservation) -> None: ...


def url_digest(url: str) -> str:
    """Hash full URL because its query may carry personal or secret material."""
    return hashlib.sha256(url.encode("utf-8", errors="replace")).hexdigest()
