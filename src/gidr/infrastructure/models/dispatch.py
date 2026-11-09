"""Production model POST transport with pinned public DNS and one-use durable permits."""

from __future__ import annotations

import http.client
import ipaddress
import socket
import ssl
from collections.abc import Mapping
from urllib.parse import urlsplit

from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.ports import (
    ModelDispatchPermit,
    ModelHTTPResponse,
)
from gidr.infrastructure.models.state_bridge import StateAuthorityModelBridge
from gidr.infrastructure.network.policy import URLPolicy

_ENDPOINTS: dict[str, tuple[str, str]] = {
    "jev": ("api.typesafe.ai", "/v1/systemone"),
    "gemini": ("generativelanguage.googleapis.com", "/v1beta/openai/chat/completions"),
    "groq": ("api.groq.com", "/openai/v1/chat/completions"),
    "siliconflow": ("api.siliconflow.cn", "/v1/chat/completions"),
}
_SAFE_RESPONSE_HEADERS = {"content-type", "retry-after", "x-typesafe-request-id"}
_ALLOWED_REQUEST_HEADERS = {
    "accept",
    "accept-encoding",
    "authorization",
    "content-type",
    "connection",
    "user-agent",
    "x-stainless-lang",
    "x-stainless-package-version",
    "x-stainless-os",
    "x-stainless-arch",
    "x-stainless-runtime",
    "x-stainless-runtime-version",
    "x-stainless-async",
}


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, timeout: float):
        self._ssl_context = ssl.create_default_context()
        super().__init__(host, 443, timeout=timeout, context=self._ssl_context)
        self._address = address

    def connect(self) -> None:
        raw = socket.create_connection((self._address, 443), self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw, server_hostname=self.host)


class ModelDispatchDenied(RuntimeError):
    """Stable sanitized model dispatch rejection."""


class PinnedModelHTTPDispatch:
    """No proxy, redirect, arbitrary host, unpinned DNS, or reusable permit."""

    def __init__(
        self,
        *,
        authority: StateAuthorityModelBridge,
        url_policy: URLPolicy,
        secret_policy: SecretPolicy,
    ) -> None:
        self._authority = authority
        self._urls = url_policy
        self._secrets = secret_policy

    async def send(
        self,
        *,
        permit: ModelDispatchPermit,
        url: str,
        method: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout_seconds: float,
        max_response_bytes: int,
    ) -> ModelHTTPResponse:
        endpoint = _ENDPOINTS.get(permit.provider_id)
        if (
            endpoint is None
            or method != "POST"
            or not 0 < timeout_seconds <= 120
            or not 1 <= max_response_bytes <= 10_000_000
        ):
            raise ModelDispatchDenied("MODEL_ENDPOINT_OR_METHOD_DENIED")
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname != endpoint[0]
            or parsed.port not in {None, 443}
            or parsed.path != endpoint[1]
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ModelDispatchDenied("MODEL_ENDPOINT_OR_METHOD_DENIED")
        safe_headers = self._headers(headers)
        try:
            self._secrets.validate_text(body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, ValueError):
            raise ModelDispatchDenied("MODEL_REQUEST_POLICY_BLOCKED") from None
        try:
            target = self._urls.validate(url)
        except Exception:
            raise ModelDispatchDenied("MODEL_URL_POLICY_BLOCKED") from None
        if target.hostname != endpoint[0] or target.port != 443 or not target.addresses:
            raise ModelDispatchDenied("MODEL_DNS_PIN_INVALID")
        for address in target.addresses:
            try:
                if not ipaddress.ip_address(address).is_global:
                    raise ModelDispatchDenied("MODEL_NON_PUBLIC_DESTINATION")
            except ValueError:
                raise ModelDispatchDenied("MODEL_DNS_ADDRESS_INVALID") from None

        try:
            await self._authority.begin_dispatch(permit, url, method, body)
        except Exception:
            raise ModelDispatchDenied("MODEL_DURABLE_PERMIT_REJECTED") from None

        connection: _PinnedHTTPSConnection | None = None
        response_status = 0
        response_body = b""
        finished = False
        try:
            connection = _PinnedHTTPSConnection(
                target.hostname, target.addresses[0], timeout_seconds
            )
            connection.request(
                "POST",
                parsed.path or "/",
                body=body,
                headers={**safe_headers, "Host": target.hostname, "Connection": "close"},
            )
            response = connection.getresponse()
            response_status = response.status
            if response.status in {301, 302, 303, 307, 308}:
                response.read(min(max_response_bytes, 65536))
                raise ModelDispatchDenied("MODEL_REDIRECT_BLOCKED")
            length = response.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > max_response_bytes):
                raise ModelDispatchDenied("MODEL_RESPONSE_SIZE_LIMIT")
            chunks: list[bytes] = []
            size = 0
            while True:
                chunk = response.read(min(65536, max_response_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > max_response_bytes:
                    raise ModelDispatchDenied("MODEL_RESPONSE_SIZE_LIMIT")
                chunks.append(chunk)
            response_body = b"".join(chunks)
            try:
                self._secrets.validate_text(response_body.decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, ValueError):
                raise ModelDispatchDenied("MODEL_RESPONSE_POLICY_BLOCKED") from None
            response_headers = {
                name.lower(): value[:2048]
                for name, value in response.getheaders()
                if name.lower() in _SAFE_RESPONSE_HEADERS
            }
            await self._authority.finish_dispatch(permit, response_status, response_body)
            finished = True
            return ModelHTTPResponse(response_status, response_headers, response_body)
        except ModelDispatchDenied:
            if response_status and not finished:
                try:
                    await self._authority.finish_dispatch(permit, response_status, response_body)
                    finished = True
                except Exception:
                    pass
            raise
        except Exception:
            if response_status and not finished:
                try:
                    await self._authority.finish_dispatch(permit, response_status, response_body)
                    finished = True
                except Exception:
                    pass
            raise ModelDispatchDenied("MODEL_HTTP_DISPATCH_FAILED") from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _headers(headers: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in headers.items():
            lower = key.lower()
            if lower not in _ALLOWED_REQUEST_HEADERS or lower in result:
                raise ModelDispatchDenied("MODEL_REQUEST_HEADER_DENIED")
            if lower == "connection":
                continue
            if any(ord(char) < 32 or ord(char) == 127 for char in value):
                raise ModelDispatchDenied("MODEL_REQUEST_HEADER_INVALID")
            result[lower] = value
        if "authorization" not in result or "content-type" not in result:
            raise ModelDispatchDenied("MODEL_AUTHORIZATION_HEADER_REQUIRED")
        return result
