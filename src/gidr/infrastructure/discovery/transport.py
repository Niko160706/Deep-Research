"""Discovery transport boundary; production requests require grant-bound broker authority."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol

from gidr.infrastructure.network.broker import FetchBroker, FetchResponse


class SourceTransport(Protocol):
    def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResponse: ...


class SourceDispatchBlocked(RuntimeError):
    """Stable blocked reason safe to expose in coverage and audit."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


class BrokerSourceTransport:
    def __init__(
        self,
        broker: FetchBroker,
        *,
        authorize: Callable[[str], object] | None,
        source_id: str | None = None,
    ) -> None:
        self._broker = broker
        self._authorize = authorize
        self._source_id = source_id

    def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> FetchResponse:
        if self._authorize is None:
            raise SourceDispatchBlocked("SOURCE_STATE_AUTHORITY_BRIDGE_UNAVAILABLE")
        try:
            capability = self._authorize(url)
        except Exception:
            raise SourceDispatchBlocked("NETWORK_ACTION_NOT_AUTHORIZED") from None
        try:
            return self._broker.fetch(
                url,
                headers=headers,
                authorization=capability,
                source_id=self._source_id,
            )
        except Exception as exc:
            reason = getattr(exc, "reason", "SOURCE_FETCH_FAILED")
            raise SourceDispatchBlocked(str(reason)) from None
