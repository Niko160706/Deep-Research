from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

import pytest

from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.models.dispatch import (
    ModelDispatchDenied,
    PinnedModelHTTPDispatch,
)
from gidr.infrastructure.models.ports import ModelDispatchPermit, ModelHTTPResponse
from gidr.infrastructure.network.policy import ValidatedURL


class AuthorityStub:
    def __init__(self) -> None:
        self.started: list[tuple[str, str, bytes]] = []
        self.finished: list[tuple[int, bytes]] = []

    async def begin_dispatch(
        self, permit: ModelDispatchPermit, url: str, method: str, body: bytes
    ) -> None:
        self.started.append((url, method, body))

    async def finish_dispatch(self, permit: ModelDispatchPermit, status: int, body: bytes) -> None:
        self.finished.append((status, body))


class ResponseStub:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body
        self._read = False

    def getheader(self, name: str) -> str | None:
        return None

    def getheaders(self) -> list[tuple[str, str]]:
        return [("Content-Type", "application/json"), ("Set-Cookie", "ignored")]

    def read(self, size: int = -1) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._body[:size]


class ConnectionStub:
    response = ResponseStub(200, b'{"ok":true}')
    instances: ClassVar[list[ConnectionStub]] = []

    def __init__(self, host: str, address: str, timeout: float) -> None:
        self.host = host
        self.address = address
        self.timeout = timeout
        self.request_args: tuple[object, ...] | None = None
        self.request_headers: Mapping[str, str] | None = None
        self.closed = False
        self.instances.append(self)

    def request(self, method: str, target: str, *, body: bytes, headers: Mapping[str, str]) -> None:
        self.request_args = (method, target, body)
        self.request_headers = headers

    def getresponse(self) -> ResponseStub:
        return self.response

    def close(self) -> None:
        self.closed = True


def make_dispatch(monkeypatch, authority: AuthorityStub, response: ResponseStub):
    from gidr.infrastructure.models import dispatch as transport

    ConnectionStub.instances.clear()
    ConnectionStub.response = response
    monkeypatch.setattr(transport, "_PinnedHTTPSConnection", ConnectionStub)

    class URLStub:
        def validate(self, url: str) -> ValidatedURL:
            return ValidatedURL(
                url=url,
                scheme="https",
                hostname="api.typesafe.ai",
                port=443,
                addresses=("8.8.8.8",),
            )

    return PinnedModelHTTPDispatch(
        authority=authority,  # type: ignore[arg-type]
        url_policy=URLStub(),  # type: ignore[arg-type]
        secret_policy=SecretPolicy(),
    )


def permit() -> ModelDispatchPermit:
    return ModelDispatchPermit(
        attempt_id="ATT-synthetic",
        run_id="RUN-synthetic",
        task_id="TASK-synthetic",
        execution_id="MOD-synthetic",
        provider_id="jev",
        reservation_id="ATT-parent",
        dispatch=object(),  # type: ignore[arg-type]
        opaque_token=object(),
    )


@pytest.mark.asyncio
async def test_pinned_dispatch_records_only_allowlisted_headers_and_response(monkeypatch) -> None:
    authority = AuthorityStub()
    dispatch = make_dispatch(monkeypatch, authority, ResponseStub(200, b'{"ok":true}'))

    result = await dispatch.send(
        permit=permit(),
        url="https://api.typesafe.ai/v1/systemone",
        method="POST",
        headers={
            "Authorization": "Bearer synthetic",
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        },
        body=b'{"input":"synthetic"}',
        timeout_seconds=2,
        max_response_bytes=128,
    )

    assert result == ModelHTTPResponse(200, {"content-type": "application/json"}, b'{"ok":true}')
    assert len(authority.started) == len(authority.finished) == 1
    connection = ConnectionStub.instances[0]
    assert connection.address == "8.8.8.8"
    assert connection.request_headers is not None
    assert "connection" not in connection.request_headers
    assert connection.request_headers["Host"] == "api.typesafe.ai"
    assert connection.closed


@pytest.mark.asyncio
async def test_pinned_dispatch_rejects_redirect_and_audits_status(monkeypatch) -> None:
    authority = AuthorityStub()
    dispatch = make_dispatch(monkeypatch, authority, ResponseStub(302, b""))

    with pytest.raises(ModelDispatchDenied, match="MODEL_REDIRECT_BLOCKED"):
        await dispatch.send(
            permit=permit(),
            url="https://api.typesafe.ai/v1/systemone",
            method="POST",
            headers={"Authorization": "Bearer synthetic", "Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=2,
            max_response_bytes=128,
        )

    assert authority.finished == [(302, b"")]


@pytest.mark.asyncio
async def test_pinned_dispatch_rejects_unapproved_url_before_authority(monkeypatch) -> None:
    authority = AuthorityStub()
    dispatch = make_dispatch(monkeypatch, authority, ResponseStub(200, b"{}"))

    with pytest.raises(ModelDispatchDenied, match="MODEL_ENDPOINT_OR_METHOD_DENIED"):
        await dispatch.send(
            permit=permit(),
            url="https://169.254.169.254/v1/systemone",
            method="POST",
            headers={"Authorization": "Bearer synthetic", "Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=2,
            max_response_bytes=128,
        )

    assert authority.started == []
    assert ConnectionStub.instances == []


def test_pinned_dispatch_rejects_host_and_header_injection() -> None:
    with pytest.raises(ModelDispatchDenied, match="MODEL_REQUEST_HEADER_DENIED"):
        PinnedModelHTTPDispatch._headers(
            {
                "Authorization": "Bearer synthetic",
                "Content-Type": "application/json",
                "Host": "evil",
            }
        )
    with pytest.raises(ModelDispatchDenied, match="MODEL_REQUEST_HEADER_INVALID"):
        PinnedModelHTTPDispatch._headers(
            {"Authorization": "Bearer synthetic\r\nX-Evil: yes", "Content-Type": "application/json"}
        )
