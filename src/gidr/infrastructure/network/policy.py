"""URL, DNS and redirect policy. All resolved addresses must be public."""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast
from urllib.parse import urljoin, urlsplit, urlunsplit

type DNSAnswers = list[tuple[object, object, int, str, tuple[object, ...]]]


class UnsafeURL(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class ValidatedURL:
    url: str
    scheme: str
    hostname: str
    port: int
    addresses: tuple[str, ...]


class BoundedResolver:
    """At most one daemon lookup can be outstanding; a stuck resolver fails closed."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 3.0,
        resolver: Callable[..., object] | None = None,
    ) -> None:
        if not 0 < timeout_seconds <= 30:
            raise ValueError("DNS_TIMEOUT_INVALID")
        self._timeout = timeout_seconds
        self._resolver = resolver or socket.getaddrinfo
        self._lock = threading.Lock()

    def __call__(self, host: str, port: int, *, type: int) -> DNSAnswers:
        if not self._lock.acquire(blocking=False):
            raise TimeoutError("DNS_RESOLVER_BUSY")
        completed = threading.Event()
        result: list[DNSAnswers] = []
        error: list[Exception] = []

        def lookup() -> None:
            try:
                result.append(cast(DNSAnswers, self._resolver(host, port, type=type)))
            except Exception as exc:
                error.append(exc)
            finally:
                completed.set()
                self._lock.release()

        try:
            threading.Thread(target=lookup, name="gidr-dns", daemon=True).start()
        except RuntimeError:
            self._lock.release()
            raise
        if not completed.wait(self._timeout):
            # Keep the lock held until the daemon returns; no unbounded stuck threads.
            raise TimeoutError("DNS_RESOLUTION_TIMEOUT")
        if error:
            raise error[0]
        return result[0]


class URLPolicy:
    def __init__(
        self,
        *,
        resolver: Callable[..., object] | None = None,
        allow_http_hosts: frozenset[str] = frozenset(),
        allowed_ports: frozenset[int] = frozenset({80, 443}),
        dns_timeout_seconds: float = 3.0,
    ) -> None:
        self._resolver = (
            resolver
            if resolver is not None
            else BoundedResolver(timeout_seconds=dns_timeout_seconds)
        )
        self._http_hosts = allow_http_hosts
        self._ports = allowed_ports

    def validate(self, url: str) -> ValidatedURL:
        if (
            not isinstance(url, str)
            or len(url) > 8192
            or any(ord(c) < 32 or ord(c) == 127 for c in url)
        ):
            raise UnsafeURL("URL_INVALID")
        try:
            parsed = urlsplit(url)
            host = parsed.hostname
            port = parsed.port
        except ValueError as exc:
            raise UnsafeURL("URL_INVALID") from exc
        if parsed.scheme.lower() not in {"https", "http"} or not host:
            raise UnsafeURL("URL_SCHEME_OR_HOST_DENIED")
        scheme = parsed.scheme.lower()
        if parsed.username is not None or parsed.password is not None or parsed.fragment:
            raise UnsafeURL("URL_CREDENTIAL_OR_FRAGMENT_DENIED")
        try:
            hostname = host.encode("idna").decode("ascii").lower().rstrip(".")
        except UnicodeError as exc:
            raise UnsafeURL("URL_HOST_INVALID") from exc
        try:
            literal = ipaddress.ip_address(hostname)
        except ValueError:
            literal = None
            if not hostname or not re.fullmatch(r"[a-z0-9.-]+", hostname) or ".." in hostname:
                raise UnsafeURL("URL_HOST_INVALID")
        if hostname == "localhost" or hostname.endswith((".localhost", ".local")):
            raise UnsafeURL("NON_PUBLIC_DESTINATION")
        effective_port = port or (443 if scheme == "https" else 80)
        if effective_port not in self._ports:
            raise UnsafeURL("URL_PORT_DENIED")
        if scheme == "http" and hostname not in self._http_hosts:
            raise UnsafeURL("URL_HTTP_NOT_AUTHORIZED")
        if literal is not None:
            addresses: tuple[str, ...] = (str(literal),)
        else:
            try:
                answers = cast(
                    DNSAnswers,
                    self._resolver(hostname, effective_port, type=socket.SOCK_STREAM),
                )
                addresses = tuple(
                    dict.fromkeys(str(answer[4][0]).split("%", 1)[0] for answer in answers)
                )
            except TimeoutError as exc:
                raise UnsafeURL("DNS_RESOLUTION_TIMEOUT") from exc
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                raise UnsafeURL("DNS_RESOLUTION_FAILED") from exc
        if not addresses:
            raise UnsafeURL("DNS_NO_ADDRESSES")
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address)
            except ValueError as exc:
                raise UnsafeURL("DNS_ADDRESS_INVALID") from exc
            if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
                ip = ip.ipv4_mapped
            if not ip.is_global:
                raise UnsafeURL("NON_PUBLIC_DESTINATION")
        path = parsed.path or "/"
        netloc_host = f"[{hostname}]" if ":" in hostname else hostname
        normalized = urlunsplit((scheme, f"{netloc_host}:{effective_port}", path, parsed.query, ""))
        return ValidatedURL(normalized, scheme, hostname, effective_port, addresses)

    @staticmethod
    def redirect(current: str, location: str) -> str:
        if not location or len(location) > 8192:
            raise UnsafeURL("REDIRECT_LOCATION_INVALID")
        target = urljoin(current, location)
        if target == current:
            raise UnsafeURL("REDIRECT_LOOP")
        return target
