"""One-request HTTPX2 bridge that delegates all I/O to an authorized dispatcher."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx2

from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.ports import ModelDispatchPermit


class ModelTransportDenied(RuntimeError):
    """The SDK attempted an unapproved or unsafe provider request."""


class AuthorizedModelTransport(httpx2.AsyncBaseTransport):
    """No DNS/proxy/socket access here; the durable permit's dispatcher owns I/O."""

    def __init__(
        self,
        permit: ModelDispatchPermit,
        *,
        allowed_origin: str,
        allowed_path: str,
        secret_policy: SecretPolicy,
        max_response_bytes: int = 1_000_000,
        timeout_seconds: float = 45.0,
    ) -> None:
        if max_response_bytes < 1 or not 0 < timeout_seconds <= 120:
            raise ValueError("MODEL_TRANSPORT_LIMIT_INVALID")
        parts = urlsplit(allowed_origin)
        if (
            parts.scheme != "https"
            or not parts.hostname
            or parts.path not in {"", "/"}
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError("MODEL_ORIGIN_INVALID")
        self._permit = permit
        self._origin = f"https://{parts.hostname.encode('idna').decode('ascii').lower()}"
        if parts.port is not None:
            self._origin += f":{parts.port}"
        self._path = allowed_path
        self._secret_policy = secret_policy
        self._max_bytes = max_response_bytes
        self._timeout = timeout_seconds
        self._used = False

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        if self._used:
            raise ModelTransportDenied("MODEL_ATTEMPT_REPLAYED")
        self._used = True
        if request.method != "POST":
            raise ModelTransportDenied("MODEL_METHOD_DENIED")
        url = str(request.url)
        parts = urlsplit(url)
        if (
            f"{parts.scheme}://{parts.netloc}" != self._origin
            or parts.path != self._path
            or parts.query
            or parts.fragment
            or request.headers.get("host", "").lower() != parts.netloc.lower()
        ):
            raise ModelTransportDenied("MODEL_ENDPOINT_DENIED")
        body = await request.aread()
        try:
            self._secret_policy.validate_text(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise ModelTransportDenied("MODEL_REQUEST_SECRET_OR_ENCODING_BLOCKED") from None
        response = await self._permit.dispatch.send(
            permit=self._permit,
            url=url,
            method="POST",
            headers={key: value for key, value in request.headers.items()},
            body=body,
            timeout_seconds=self._timeout,
            max_response_bytes=self._max_bytes,
        )
        if response.status_code in {301, 302, 303, 307, 308}:
            raise ModelTransportDenied("MODEL_REDIRECT_BLOCKED")
        if len(response.body) > self._max_bytes:
            raise ModelTransportDenied("MODEL_RESPONSE_SIZE_LIMIT")
        try:
            self._secret_policy.validate_text(response.body.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, ValueError):
            raise ModelTransportDenied("MODEL_RESPONSE_SECRET_OR_ENCODING_BLOCKED") from None
        headers = {
            key: value
            for key, value in response.headers.items()
            if key.lower() in {"content-type", "retry-after", "x-typesafe-request-id"}
        }
        return httpx2.Response(
            response.status_code,
            headers=headers,
            content=response.body,
            request=request,
        )

    async def aclose(self) -> None:
        return None
