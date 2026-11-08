"""Source adapter boundary with canonical inputs/outputs, no SDK dependency."""

from typing import Literal, Protocol, Self

from pydantic import Field, model_validator

from gidr.contracts.common import (
    ArtifactRef,
    Contract,
    CredentialAlias,
    GrantId,
    Language,
    Name,
    Positive,
    QueryId,
    QuestionId,
    Record,
    Text,
    UTCDateTime,
)
from gidr.contracts.sources import SourceCandidate


class CanonicalQuery(Record):
    query_id: QueryId
    question_id: QuestionId
    terms: tuple[Text, ...] = ()
    exact_phrases: tuple[Text, ...] = ()
    negative_terms: tuple[Text, ...] = ()
    domain: Text | None = None
    filetype: Name | None = None
    language: Language | None = None
    date_start: UTCDateTime | None = None
    date_end: UTCDateTime | None = None
    author: Text | None = None
    doi: Text | None = None
    region: Text | None = None
    source_types: tuple[Name, ...] = ()
    mandatory_filters: tuple[Name, ...] = ()
    max_results_budget: Positive


class SourceContext(Record):
    grant_id: GrantId
    plan_version: Positive


class ProviderCapabilities(Contract):
    adapter_id: Name
    adapter_version: Text
    source_families: tuple[Name, ...]
    supported_filters: tuple[Name, ...]
    language_codes: tuple[Text, ...]
    allowed_origins: tuple[Text, ...]
    auth_alias: CredentialAlias | None
    pagination: Literal["NONE", "CURSOR", "OFFSET", "PAGE"]
    rights_policy_ref: Text


class ProviderHealth(Contract):
    status: Literal[
        "NOT_IMPLEMENTED", "MOCK_TESTED", "LIVE_VERIFIED", "DEGRADED", "BLOCKED", "DISABLED"
    ]
    checked_at_utc: UTCDateTime | None = None
    reason_code: Name | None = None
    verification_ref: ArtifactRef | None = None

    @model_validator(mode="after")
    def live_requires_evidence(self) -> Self:
        if self.status == "LIVE_VERIFIED" and (
            self.checked_at_utc is None or self.verification_ref is None
        ):
            raise ValueError("LIVE_VERIFIED requires timestamp and live verification evidence")
        return self


class DiscoveryBatch(Record):
    provider_id: Name
    candidates: tuple[SourceCandidate, ...] = ()
    next_cursor: Text | None = None
    warnings: tuple[Name, ...] = ()
    unsupported_filters: tuple[Name, ...] = ()
    health: ProviderHealth = Field(default_factory=lambda: ProviderHealth(status="NOT_IMPLEMENTED"))

    @model_validator(mode="after")
    def coherent_batch(self) -> Self:
        if any(
            c.run_id != self.run_id or c.provider_id != self.provider_id for c in self.candidates
        ):
            raise ValueError("Discovery provenance does not match batch context")
        if len({c.source_id for c in self.candidates}) != len(self.candidates):
            raise ValueError("Duplicate canonical source identity in batch")
        return self


class SourceAdapter(Protocol):
    @property
    def adapter_id(self) -> str: ...

    def capabilities(self) -> ProviderCapabilities: ...

    async def healthcheck(self) -> ProviderHealth: ...

    async def discover(self, query: CanonicalQuery, context: SourceContext) -> DiscoveryBatch: ...
