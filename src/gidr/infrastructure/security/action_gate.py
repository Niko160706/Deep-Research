"""Grant-bound, single-use capabilities for outbound network operations."""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Annotated, Literal
from urllib.parse import urlsplit

from pydantic import Field, ValidationError

from gidr.contracts.commands import AttemptToken, GrantPolicy
from gidr.contracts.common import (
    Contract,
    CredentialAlias,
    GrantId,
    Name,
    Nonnegative,
    Positive,
    RunId,
    Text,
)
from gidr.infrastructure.network.policy import UnsafeURL, URLPolicy
from gidr.infrastructure.security.events import SecurityEventSink, SecurityObservation, url_digest

Money = Annotated[Decimal, Field(ge=0, le=10**12, max_digits=24, decimal_places=9)]


class ProviderScope(Contract):
    provider_id: Name
    allowed_hosts: tuple[Name, ...]
    credential_aliases: tuple[CredentialAlias, ...] = ()


class GrantSnapshot(Contract):
    """Read-only projection supplied by the trusted state authority adapter."""

    run_id: RunId
    grant: GrantPolicy
    plan_version: Positive
    active: bool
    user_approved: bool
    expires_at_utc: datetime | None = None
    remaining_cost_usd: Money
    remaining_tokens: Nonnegative
    remaining_requests: Nonnegative
    provider_scopes: tuple[ProviderScope, ...]
    attempt_token: AttemptToken
    task_provider_id: Name
    task_category: Name
    task_type: Name
    task_state: Literal["RUNNING"]
    lease_valid: bool


class NetworkActionProposal(Contract):
    action_id: Name
    run_id: RunId
    plan_version: Positive
    grant_id: GrantId
    provider_id: Name
    category: Name
    task_type: Name
    attempt_token: AttemptToken
    url: Text
    estimated_cost_usd: Money = Decimal(0)
    estimated_tokens: Nonnegative = 0
    estimated_requests: Positive = 1
    credential_alias: CredentialAlias | None = None
    source_id: Name | None = None


class NetworkActionDenied(PermissionError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class NetworkAuthorization:
    """Opaque capability; only its issuing gate can validate it, and only once."""

    __slots__ = (
        "_credential_alias",
        "_expires",
        "_lock",
        "_seal",
        "_source_id",
        "_url_hash",
        "_used",
    )

    def __init__(
        self,
        seal: object,
        url_hash: str,
        source_id: str | None,
        credential_alias: str | None,
        expires: datetime,
    ):
        self._seal = seal
        self._url_hash = url_hash
        self._source_id = source_id
        self._credential_alias = credential_alias
        self._expires = expires
        self._used = False
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return "NetworkAuthorization(<opaque>)"


class NetworkActionGate:
    """Authorize only against a current USER-approved persisted grant snapshot."""

    def __init__(
        self,
        *,
        grant_lookup: Callable[[str], GrantSnapshot | None],
        reserve_action: Callable[[NetworkActionProposal], bool],
        url_policy: URLPolicy,
        event_sink: SecurityEventSink,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        capability_ttl_seconds: int = 30,
    ) -> None:
        if not 1 <= capability_ttl_seconds <= 120:
            raise ValueError("CAPABILITY_TTL_INVALID")
        self._grant_lookup = grant_lookup
        self._reserve_action = reserve_action
        self._url_policy = url_policy
        self._events = event_sink
        self._clock = clock
        self._ttl = capability_ttl_seconds
        self._seal = object()

    def authorize(self, proposal: NetworkActionProposal) -> NetworkAuthorization:
        try:
            proposal = NetworkActionProposal.model_validate(proposal)
        except ValidationError:
            raise NetworkActionDenied("ACTION_PROPOSAL_INVALID") from None
        now = self._clock().astimezone(UTC)
        try:
            grant = self._grant_lookup(proposal.grant_id)
            if grant is None or not grant.active or not grant.user_approved:
                raise NetworkActionDenied("GRANT_INACTIVE_OR_UNAPPROVED")
            if grant.grant.grant_id != proposal.grant_id or grant.run_id != proposal.run_id:
                raise NetworkActionDenied("GRANT_LINEAGE_MISMATCH")
            if grant.plan_version != proposal.plan_version:
                raise NetworkActionDenied("STALE_PLAN_VERSION")
            if grant.expires_at_utc is not None and grant.expires_at_utc <= now:
                raise NetworkActionDenied("GRANT_EXPIRED")
            if grant.grant.expires_at_utc is not None and grant.grant.expires_at_utc <= now:
                raise NetworkActionDenied("GRANT_EXPIRED")
            if proposal.provider_id not in grant.grant.providers:
                raise NetworkActionDenied("PROVIDER_OUT_OF_SCOPE")
            if proposal.category not in grant.grant.categories:
                raise NetworkActionDenied("CATEGORY_OUT_OF_SCOPE")
            if proposal.task_type not in grant.grant.task_types:
                raise NetworkActionDenied("TASK_TYPE_OUT_OF_SCOPE")
            if (
                proposal.attempt_token != grant.attempt_token
                or grant.task_provider_id != proposal.provider_id
                or grant.task_category != proposal.category
                or grant.task_type != proposal.task_type
                or grant.task_state != "RUNNING"
                or not grant.lease_valid
            ):
                raise NetworkActionDenied("ATTEMPT_OR_TASK_CONTEXT_STALE")
            if proposal.estimated_cost_usd > grant.remaining_cost_usd:
                raise NetworkActionDenied("COST_BUDGET_EXCEEDED")
            if proposal.estimated_tokens > grant.remaining_tokens:
                raise NetworkActionDenied("TOKEN_BUDGET_EXCEEDED")
            if proposal.estimated_requests > grant.remaining_requests:
                raise NetworkActionDenied("REQUEST_BUDGET_EXCEEDED")
            scopes = [
                scope
                for scope in grant.provider_scopes
                if scope.provider_id == proposal.provider_id
            ]
            if len(scopes) != 1:
                raise NetworkActionDenied("PROVIDER_SCOPE_NOT_CONFIGURED")
            parts = urlsplit(proposal.url)
            hostname = parts.hostname
            if hostname is None:
                raise NetworkActionDenied("URL_HOST_INVALID")
            hostname = hostname.encode("idna").decode("ascii").lower().rstrip(".")
            if hostname not in scopes[0].allowed_hosts:
                raise NetworkActionDenied("DOMAIN_OUT_OF_SCOPE")
            if (
                proposal.credential_alias is not None
                and proposal.credential_alias not in scopes[0].credential_aliases
            ):
                raise NetworkActionDenied("CREDENTIAL_ALIAS_OUT_OF_SCOPE")
            self._url_policy.validate(proposal.url)
            if not self._reserve_action(proposal):
                raise NetworkActionDenied("ACTION_BUDGET_RESERVATION_DENIED")
            expires = now + timedelta(seconds=self._ttl)
            authorization = NetworkAuthorization(
                self._seal,
                url_digest(proposal.url),
                proposal.source_id,
                proposal.credential_alias,
                expires,
            )
            self._record("OUTBOUND_REQUEST", "ACTION_AUTHORIZED", proposal)
            return authorization
        except UnsafeURL as exc:
            self._record("UNSAFE_URL_BLOCKED", exc.reason, proposal)
            raise NetworkActionDenied(exc.reason) from exc
        except NetworkActionDenied as exc:
            self._record("UNAUTHORIZED_ACTION", exc.reason, proposal)
            raise
        except Exception:
            self._record("UNAUTHORIZED_ACTION", "STATE_AUTHORITY_UNAVAILABLE", proposal)
            raise NetworkActionDenied("STATE_AUTHORITY_UNAVAILABLE") from None

    def consume(
        self,
        authorization: object,
        url: str,
        source_id: str | None,
        credential_alias: str | None,
    ) -> None:
        if not isinstance(authorization, NetworkAuthorization):
            raise NetworkActionDenied("NETWORK_CAPABILITY_REQUIRED")
        now = self._clock().astimezone(UTC)
        with authorization._lock:
            if authorization._seal is not self._seal:
                raise NetworkActionDenied("NETWORK_CAPABILITY_INVALID")
            if authorization._used:
                raise NetworkActionDenied("NETWORK_CAPABILITY_REPLAYED")
            if authorization._expires <= now:
                raise NetworkActionDenied("NETWORK_CAPABILITY_EXPIRED")
            if (
                authorization._url_hash != url_digest(url)
                or authorization._source_id != source_id
                or authorization._credential_alias != credential_alias
            ):
                raise NetworkActionDenied("NETWORK_CAPABILITY_CONTEXT_MISMATCH")
            authorization._used = True

    def _record(
        self,
        category: Literal["OUTBOUND_REQUEST", "UNAUTHORIZED_ACTION", "UNSAFE_URL_BLOCKED"],
        reason: str,
        proposal: NetworkActionProposal,
    ) -> None:
        try:
            self._events.record_security(
                SecurityObservation(
                    category=category,
                    reason_code=reason,
                    request_id="ACT-" + secrets.token_hex(12),
                    source_id=proposal.source_id,
                    url_sha256=url_digest(proposal.url),
                    run_id=proposal.run_id,
                    plan_version=proposal.plan_version,
                    grant_id=proposal.grant_id,
                    provider_id=proposal.provider_id,
                    task_id=proposal.attempt_token.task_id,
                    attempt_id=proposal.attempt_token.attempt_id,
                )
            )
        except Exception:
            raise NetworkActionDenied("SECURITY_AUDIT_FAILURE") from None
