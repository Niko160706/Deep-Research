"""Configuration is trusted operator input, never downloaded content."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from typing import Literal

from pydantic import Field, ValidationError

from gidr.contracts.common import Contract, CredentialAlias, Nonnegative, Positive
from gidr.contracts.models import Money


class RuntimeSettings(Contract):
    active_run_limit: Literal[1] = 1
    workers: Positive = 3
    max_runtime_seconds: Positive = 1800


class StateSettings(Contract):
    sqlite_path: str = "data/state/gidr.sqlite3"
    journal_mode: Literal["WAL"] = "WAL"
    synchronous: Literal["FULL"] = "FULL"
    fts5: Literal[True] = True
    artifact_root: str = "data/blobs/sha256"


class BudgetSettings(Contract):
    max_tokens: Positive = 60000
    max_requests: Positive = 120
    max_cost_usd: Money = Decimal(0)
    preplan_model_cost_usd: Money = Decimal(0)
    unknown_usage_policy: Literal["CONSERVATIVE_RESERVATION"] = "CONSERVATIVE_RESERVATION"


class SourceSettings(Contract):
    enabled_providers: tuple[str, ...] = ()
    github_clone_default: Literal[False] = False
    firecrawl_enabled: Literal[False] = False


class NetworkSettings(Contract):
    enabled: bool = False
    transport: Literal["HTTPX"] = "HTTPX"
    trust_env: Literal[False] = False
    tor_opt_in: bool = False
    max_concurrent_requests: Positive = 3
    timeout_seconds: Positive = 30
    max_response_bytes: Positive = 10 * 1024 * 1024
    max_redirects: Nonnegative = 5


class ParserSettings(Contract):
    require_isolation: Literal[True] = True
    suspicious_input_policy: Literal["BLOCK"] = "BLOCK"
    timeout_seconds: Positive = 30
    max_pages: Positive = 200
    max_extracted_chars: Positive = 1000000
    allow_source_code_execution: Literal[False] = False


class ModelSettings(Contract):
    decision_primary: Literal["jev"] = "jev"
    total_attempt_limit: Positive = 3
    sdk_retries: Literal[0] = 0
    approved_credential_aliases: tuple[CredentialAlias, ...] = ()


class TranslationSettings(Contract):
    mode: Literal["SELECTIVE"] = "SELECTIVE"
    preserve_original: Literal[True] = True
    allow_corpus_translation: Literal[False] = False


class AuditSettings(Contract):
    level: Literal["STANDARD", "DIAGNOSTIC", "FULL_TRACE"] = "STANDARD"
    canonical_store: Literal["SQLITE_EVENTS"] = "SQLITE_EVENTS"
    component_logs: Literal["DERIVED"] = "DERIVED"
    include_raw_credentials: Literal[False] = False


class SecuritySettings(Contract):
    credential_mode: Literal["TRUSTED_KEYRING_OR_CONTROLLED_ENV"] = (
        "TRUSTED_KEYRING_OR_CONTROLLED_ENV"
    )
    dotenv_mode: Literal["DISABLED"] = "DISABLED"
    plaintext_keyring_fallback: Literal[False] = False


class OutputSettings(Contract):
    locale: Literal["id", "en"] = "id"
    citation_target: Literal["ORIGINAL_SOURCE"] = "ORIGINAL_SOURCE"


class AppSettings(Contract):
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    state: StateSettings = Field(default_factory=StateSettings)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    source: SourceSettings = Field(default_factory=SourceSettings)
    network: NetworkSettings = Field(default_factory=NetworkSettings)
    parser: ParserSettings = Field(default_factory=ParserSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    audit: AuditSettings = Field(default_factory=AuditSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    def public_snapshot(self) -> dict[str, object]:
        """No credential values are accepted or resolved by these settings."""
        return self.model_dump(mode="json")


class SettingsInputError(ValueError):
    """Sanitized error safe for standard diagnostics, including nested extras."""


def load_settings(data: Mapping[str, object] | None = None) -> AppSettings:
    """Explicit mapping only. Never read .env, process environment or keyring."""
    try:
        return AppSettings.model_validate(dict(data) if data is not None else {})
    except ValidationError as exc:
        codes = sorted({error["type"] for error in exc.errors(include_input=False)})
        raise SettingsInputError("Konfigurasi tidak valid: " + ", ".join(codes)) from None
