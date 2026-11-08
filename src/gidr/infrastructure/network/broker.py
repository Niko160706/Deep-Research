"""Bounded HTTPS fetches pinned to validated DNS answers; no ambient proxies."""

from __future__ import annotations

import hashlib
import http.client
import secrets
import socket
import ssl
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from gidr.infrastructure.artifacts.quarantine import (
    QuarantineError,
    QuarantineRecord,
    QuarantineStore,
)
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.network.policy import UnsafeURL, URLPolicy, ValidatedURL
from gidr.infrastructure.security.events import (
    SecurityCategory,
    SecurityEventSink,
    SecurityObservation,
    url_digest,
)
from gidr.infrastructure.security.secrets import SecretHandle


@dataclass(frozen=True, slots=True)
class FetchResponse:
    final_url: str
    status: int
    headers: Mapping[str, str]
    body: bytes
    redirects: int
    quarantine: QuarantineRecord


class FetchError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, target: ValidatedURL, address: str, timeout: float):
        context = ssl.create_default_context()
        super().__init__(target.hostname, target.port, timeout=timeout, context=context)
        self._target = target
        self._address = address
        self._ssl_context = context

    def connect(self) -> None:
        raw = socket.create_connection((self._address, self._target.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw, server_hostname=self._target.hostname)


class FetchBroker:
    def __init__(
        self,
        policy: URLPolicy,
        *,
        max_bytes: int = 2_000_000,
        timeout_seconds: float = 15.0,
        max_redirects: int = 5,
        event_sink: SecurityEventSink | None = None,
        quarantine_store: QuarantineStore | None = None,
        authorization_validator: Callable[[object, str, str | None, str | None], None]
        | None = None,
        secret_policy: SecretPolicy | None = None,
        clock: Callable[[], float] = time.monotonic,
        connector: Callable[
            [ValidatedURL, str, float], http.client.HTTPSConnection
        ] = _PinnedHTTPSConnection,
    ) -> None:
        if max_bytes < 1 or not 0 < timeout_seconds <= 120 or not 0 <= max_redirects <= 10:
            raise ValueError("FETCH_LIMIT_INVALID")
        self._policy = policy
        self._event_sink = event_sink
        self._quarantine = quarantine_store
        self._authorization_validator = authorization_validator
        self._secret_policy = secret_policy
        self._max_bytes = max_bytes
        self._timeout = timeout_seconds
        self._max_redirects = max_redirects
        self._clock = clock
        self._connector = connector

    def fetch(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        credential: object | None = None,
        authorization: object | None = None,
        allowed_origin: str | None = None,
        source_id: str | None = None,
    ) -> FetchResponse:
        request_id = "REQ-" + secrets.token_hex(12)
        if (
            self._event_sink is None
            or self._quarantine is None
            or self._secret_policy is None
            or self._quarantine.secret_policy is not self._secret_policy
        ):
            raise FetchError("SECURITY_AUDIT_OR_QUARANTINE_REQUIRED")
        if self._authorization_validator is None or authorization is None:
            self._record(
                "UNAUTHORIZED_ACTION", "NETWORK_CAPABILITY_REQUIRED", request_id, source_id, url
            )
            raise FetchError("NETWORK_CAPABILITY_REQUIRED")
        credential_alias = credential.alias if isinstance(credential, SecretHandle) else None
        try:
            self._authorization_validator(authorization, url, source_id, credential_alias)
        except Exception:
            self._record(
                "UNAUTHORIZED_ACTION", "NETWORK_CAPABILITY_DENIED", request_id, source_id, url
            )
            raise FetchError("NETWORK_CAPABILITY_DENIED") from None
        try:
            self._secret_policy.validate_text(url)
        except ValueError:
            self._record(
                "SECRET_ACCESS_DENIED", "SECRET_IN_URL_BLOCKED", request_id, source_id, url
            )
            raise FetchError("SECRET_IN_URL_BLOCKED") from None
        if credential_alias is not None:
            sensitive_parameters = {
                "api_key",
                "access_token",
                "authorization",
                "password",
                "secret",
                "token",
            }
            if any(
                key.casefold() in sensitive_parameters for key, _ in parse_qsl(urlsplit(url).query)
            ):
                self._record(
                    "SECRET_ACCESS_DENIED",
                    "CREDENTIAL_QUERY_PARAMETER_BLOCKED",
                    request_id,
                    source_id,
                    url,
                )
                raise FetchError("CREDENTIAL_QUERY_PARAMETER_BLOCKED")
        current = url
        safe_headers = self._headers(headers or {})
        deadline = self._clock() + self._timeout
        try:
            for redirect_count in range(self._max_redirects + 1):
                target = self._policy.validate(current)
                if credential is not None:
                    if allowed_origin != f"https://{target.hostname}:{target.port}":
                        raise FetchError("CREDENTIAL_ORIGIN_DENIED")
                    if not isinstance(credential, SecretHandle):
                        raise FetchError("CREDENTIAL_HANDLE_INVALID")
                    try:
                        token = credential.authorization_value(current)
                    except Exception as exc:
                        raise FetchError("CREDENTIAL_ORIGIN_DENIED") from exc
                    safe_headers["Authorization"] = f"Bearer {token}"
                self._record(
                    "OUTBOUND_REQUEST",
                    "FETCH_DISPATCHED",
                    request_id,
                    source_id,
                    current,
                    credential_alias=credential.alias
                    if isinstance(credential, SecretHandle)
                    else None,
                )
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise FetchError("FETCH_TIMEOUT")
                conn = None
                if target.scheme != "https":
                    raise FetchError("HTTP_TRANSPORT_DISABLED")
                conn = self._connector(target, target.addresses[0], min(remaining, self._timeout))
                path = urlsplit(target.url).path or "/"
                if urlsplit(target.url).query:
                    path += "?" + urlsplit(target.url).query
                host_header = f"[{target.hostname}]" if ":" in target.hostname else target.hostname
                conn.request("GET", path, headers={**safe_headers, "Host": host_header})
                response = conn.getresponse()
                location = response.getheader("Location")
                if response.status in {301, 302, 303, 307, 308}:
                    response.read(min(self._max_bytes, 65536))
                    if redirect_count >= self._max_redirects or not location:
                        raise FetchError("REDIRECT_LIMIT_OR_LOCATION")
                    next_url = self._policy.redirect(current, location)
                    next_target = self._policy.validate(next_url)
                    if credential is not None and (
                        next_target.hostname != target.hostname or next_target.port != target.port
                    ):
                        raise FetchError("CROSS_ORIGIN_CREDENTIAL_REDIRECT_BLOCKED")
                    self._record(
                        "REDIRECT_ALLOWED", "REDIRECT_REVALIDATED", request_id, source_id, next_url
                    )
                    conn.close()
                    conn = None
                    current = next_url
                    continue
                length = response.getheader("Content-Length")
                if length is not None and (not length.isdecimal() or int(length) > self._max_bytes):
                    raise FetchError("RESPONSE_SIZE_LIMIT")
                chunks: list[bytes] = []
                total = 0
                while True:
                    remaining = deadline - self._clock()
                    if remaining <= 0:
                        raise FetchError("FETCH_TIMEOUT")
                    if conn.sock is not None:
                        conn.sock.settimeout(remaining)
                    chunk = response.read(min(65536, self._max_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self._max_bytes:
                        raise FetchError("RESPONSE_SIZE_LIMIT")
                    chunks.append(chunk)
                safe_response_headers = {
                    key.lower(): value[:2048]
                    for key, value in response.getheaders()
                    if key.lower()
                    in {
                        "content-type",
                        "content-length",
                        "etag",
                        "last-modified",
                        "content-language",
                    }
                }
                body = b"".join(chunks)
                try:
                    quarantine = self._quarantine.store(
                        body,
                        declared_mime=response.getheader("Content-Type"),
                    )
                except QuarantineError as exc:
                    quarantine_category: SecurityCategory = (
                        "SECRET_ACCESS_DENIED"
                        if exc.reason == "SECRET_CONTENT_BLOCKED"
                        else "DISALLOWED_MIME"
                    )
                    self._record(
                        quarantine_category,
                        exc.reason,
                        request_id,
                        source_id,
                        current,
                        content_sha256=hashlib.sha256(body).hexdigest(),
                        byte_count=len(body),
                    )
                    raise FetchError("RESPONSE_SECURITY_POLICY_BLOCKED") from exc
                if quarantine.status != "STAGED":
                    self._record(
                        "DISALLOWED_MIME",
                        quarantine.reason_code or "MIME_QUARANTINED",
                        request_id,
                        source_id,
                        current,
                        content_sha256=quarantine.sha256,
                        byte_count=quarantine.size_bytes,
                        mime=quarantine.detected_mime,
                    )
                    raise FetchError("RESPONSE_QUARANTINED")
                response_category: SecurityCategory = (
                    "PROVIDER_AUTH_FAILURE" if response.status in {401, 403} else "OUTBOUND_REQUEST"
                )
                self._record(
                    response_category,
                    "FETCH_COMPLETED",
                    request_id,
                    source_id,
                    current,
                    status=response.status,
                    byte_count=len(body),
                    content_sha256=quarantine.sha256,
                    mime=quarantine.detected_mime,
                    credential_alias=credential.alias
                    if isinstance(credential, SecretHandle)
                    else None,
                )
                return FetchResponse(
                    current,
                    response.status,
                    safe_response_headers,
                    body,
                    redirect_count,
                    quarantine,
                )
            raise FetchError("REDIRECT_LIMIT")
        except UnsafeURL as exc:
            unsafe_url_category: SecurityCategory = (
                "DNS_REBINDING_BLOCKED" if "DNS" in exc.reason else "UNSAFE_URL_BLOCKED"
            )
            self._record(unsafe_url_category, exc.reason, request_id, source_id, current)
            raise FetchError(exc.reason) from exc
        except TimeoutError as exc:
            self._record("OUTBOUND_REQUEST", "FETCH_TIMEOUT", request_id, source_id, current)
            raise FetchError("FETCH_TIMEOUT") from exc
        except FetchError as exc:
            fetch_error_category: SecurityCategory = (
                "REDIRECT_BLOCKED"
                if "REDIRECT" in exc.reason or "ORIGIN" in exc.reason
                else "OUTBOUND_REQUEST"
            )
            self._record(fetch_error_category, exc.reason, request_id, source_id, current)
            raise
        except (OSError, http.client.HTTPException, ssl.SSLError) as exc:
            self._record(
                "OUTBOUND_REQUEST", "FETCH_TRANSPORT_FAILED", request_id, source_id, current
            )
            raise FetchError("FETCH_TRANSPORT_FAILED") from exc
        finally:
            if "conn" in locals() and conn is not None:
                conn.close()

    def _record(
        self,
        category: SecurityCategory,
        reason: str,
        request_id: str,
        source_id: str | None,
        url: str,
        *,
        status: int | None = None,
        byte_count: int | None = None,
        credential_alias: str | None = None,
        content_sha256: str | None = None,
        mime: str | None = None,
    ) -> None:
        assert self._event_sink is not None
        try:
            self._event_sink.record_security(
                SecurityObservation(
                    category=category,
                    reason_code=reason,
                    request_id=request_id,
                    source_id=source_id,
                    url_sha256=url_digest(url),
                    status_code=status,
                    byte_count=byte_count,
                    credential_alias=credential_alias,
                    content_sha256=content_sha256,
                    mime=mime,
                )
            )
        except Exception:
            raise FetchError("SECURITY_AUDIT_FAILURE") from None

    @staticmethod
    def _headers(headers: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {
            "User-Agent": "GIDR/0.1 (+research metadata fetch)",
            "Accept": "*/*",
        }
        for key, value in headers.items():
            if (
                not key.isascii()
                or not key.replace("-", "").isalnum()
                or not isinstance(value, str)
            ):
                raise FetchError("REQUEST_HEADER_INVALID")
            if key.lower() in {
                "host",
                "connection",
                "proxy-authorization",
                "cookie",
                "authorization",
            }:
                raise FetchError("REQUEST_HEADER_FORBIDDEN")
            if "\r" in value or "\n" in value or len(value) > 4096:
                raise FetchError("REQUEST_HEADER_INVALID")
            result[key] = value
        return result
