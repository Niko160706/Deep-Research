"""Resolve approved credential aliases without creating a plaintext vault."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.security.events import SecurityEventSink, SecurityObservation

ALIASES: dict[str, str] = {
    "provider/typesafe": "TYPESAFE_API_KEY",
    "provider/tavily": "TAVILY_API_KEY",
    "provider/exa": "EXA_API_KEY",
    "provider/gemini": "GEMINI_API_KEY",
    "provider/groq": "GROQ_API_KEY",
    "provider/siliconflow": "SILICONFLOW_API_KEY",
}


class SecretUnavailable(RuntimeError):
    """A credential cannot be resolved from a trusted source."""


@dataclass(frozen=True, slots=True)
class SecretHandle:
    """Non-serializable-in-practice credential handle; repr never contains value."""

    alias: str
    _value: str
    _origins: frozenset[str]

    def authorization_value(self, url: str) -> str:
        origin = _origin(url)
        if origin not in self._origins:
            raise SecretUnavailable("CREDENTIAL_ORIGIN_DENIED")
        return self._value

    def __repr__(self) -> str:
        return f"SecretHandle(alias={self.alias!r}, value=<redacted>)"


def _origin(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise SecretUnavailable("CREDENTIAL_ORIGIN_DENIED")
    port = parsed.port or 443
    return f"https://{parsed.hostname.encode('idna').decode('ascii').lower()}:{port}"


class SecretManager:
    """Keyring first, explicit environment aliases second; never reads .env or files."""

    def __init__(
        self,
        *,
        secret_policy: SecretPolicy,
        origins: dict[str, frozenset[str]],
        allow_environment: bool = False,
        environ: dict[str, str] | None = None,
        event_sink: SecurityEventSink | None = None,
    ) -> None:
        self._policy = secret_policy
        self._origins = origins
        self._allow_environment = allow_environment
        self._environ = os.environ if environ is None else environ
        self._event_sink = event_sink

    def resolve(self, alias: str) -> SecretHandle:
        if self._event_sink is None:
            raise SecretUnavailable("SECURITY_AUDIT_SINK_REQUIRED")
        variable = ALIASES.get(alias)
        allowed = self._origins.get(alias)
        if variable is None or not allowed:
            self._record(alias, "CREDENTIAL_ALIAS_NOT_APPROVED", denied=True)
            raise SecretUnavailable("CREDENTIAL_ALIAS_NOT_APPROVED")
        value = self._from_keyring(alias)
        source = "OS_KEYRING"
        if value is None and self._allow_environment:
            value = self._environ.get(variable)
            source = "CONTROLLED_ENVIRONMENT"
        if not value or not value.strip():
            self._record(alias, "CREDENTIAL_NOT_CONFIGURED", denied=True)
            raise SecretUnavailable("CREDENTIAL_NOT_CONFIGURED")
        self._policy.register_secret(value)
        self._record(alias, f"CREDENTIAL_RESOLVED_{source}", denied=False)
        return SecretHandle(alias, value, allowed)

    def _record(self, alias: str, reason: str, *, denied: bool) -> None:
        assert self._event_sink is not None
        try:
            self._event_sink.record_security(
                SecurityObservation(
                    category="SECRET_ACCESS_DENIED" if denied else "CREDENTIAL_PROVIDER_CONFIG",
                    reason_code=reason,
                    request_id="CRED-" + secrets.token_hex(12),
                    credential_alias=alias if alias in ALIASES else None,
                )
            )
        except Exception:
            raise SecretUnavailable("SECURITY_AUDIT_FAILURE") from None

    @staticmethod
    def _from_keyring(alias: str) -> str | None:
        try:
            import keyring

            backend = keyring.get_keyring()
            if getattr(backend, "priority", 0) <= 0:
                return None
            backend_module = type(backend).__module__
            if backend_module not in {
                "keyring.backends.SecretService",
                "keyring.backends.kwallet",
                "keyring.backends.Windows",
                "keyring.backends.macOS",
            }:
                return None
            value = keyring.get_password("gidr", alias)
            return value if isinstance(value, str) else None
        except Exception:
            return None
