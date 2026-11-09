"""Validated query, provider and per-attempt coverage contracts."""

from __future__ import annotations

from datetime import date
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from gidr.contracts.common import (
    Contract,
    Name,
    Nonnegative,
    Positive,
    QueryId,
    QuestionId,
    UTCDateTime,
)

ProviderStatus = Literal[
    "NOT_IMPLEMENTED", "MOCK_TESTED", "LIVE_VERIFIED", "DEGRADED", "BLOCKED", "DISABLED"
]
FilterDisposition = Literal["NATIVE", "LOCAL_POSTFILTER", "UNSUPPORTED"]


class CanonicalQuery(Contract):
    query_id: QueryId
    question_id: QuestionId
    text: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    date_from: date | None = None
    date_to: date | None = None
    languages: tuple[Annotated[str, StringConstraints(pattern=r"^[a-z]{2,3}$")], ...] = ()
    domains: tuple[Annotated[str, StringConstraints(min_length=1, max_length=253)], ...] = ()
    required_terms: tuple[Annotated[str, StringConstraints(min_length=1, max_length=128)], ...] = ()
    mandatory_filters: tuple[Name, ...] = ()
    max_results: Positive = Field(le=100)

    @model_validator(mode="after")
    def valid_window_and_filters(self) -> CanonicalQuery:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("QUERY_DATE_WINDOW_INVALID")
        if len(set(self.languages)) != len(self.languages) or len(set(self.domains)) != len(
            self.domains
        ):
            raise ValueError("QUERY_FILTER_DUPLICATE")
        allowed = {"text", "date", "language", "domain", "required_terms"}
        if not set(self.mandatory_filters) <= allowed:
            raise ValueError("UNKNOWN_MANDATORY_FILTER")
        present = {
            "text": bool(self.text),
            "date": self.date_from is not None or self.date_to is not None,
            "language": bool(self.languages),
            "domain": bool(self.domains),
            "required_terms": bool(self.required_terms),
        }
        if any(not present[name] for name in self.mandatory_filters):
            raise ValueError("MANDATORY_FILTER_VALUE_REQUIRED")
        return self


class CompiledFilter(Contract):
    name: Name
    disposition: FilterDisposition
    mandatory: bool
    reason: Name | None = None


class CompiledQuery(Contract):
    provider_id: Name
    original: CanonicalQuery
    filters: tuple[CompiledFilter, ...]
    blocked_reason: Name | None = None

    @model_validator(mode="after")
    def block_unsupported_mandatory(self) -> CompiledQuery:
        unsupported = any(
            item.mandatory and item.disposition == "UNSUPPORTED" for item in self.filters
        )
        if unsupported != (self.blocked_reason is not None):
            raise ValueError("MANDATORY_FILTER_BLOCK_REASON_REQUIRED")
        return self


class ProviderCapabilities(Contract):
    provider_id: Name
    category: Name
    priority: Annotated[int, Field(ge=0, le=1000)] = 100
    status: ProviderStatus
    native_filters: tuple[Name, ...]
    local_filters: tuple[Name, ...] = ()
    pagination: Literal["NONE", "OFFSET", "CURSOR", "PAGE", "TIMESTAMP"]
    max_page_size: Positive
    metadata_first: bool = True
    default_clone: bool = False
    base_hosts: tuple[Name, ...]
    credential_alias: Name | None = None
    status_reason: Name | None = None


class ProviderHealth(Contract):
    provider_id: Name
    status: ProviderStatus
    checked_at_utc: UTCDateTime | None = None
    reason_code: Name | None = None
    adapter_version: str


class SourceCoverageObservation(Contract):
    run_id: Annotated[str, StringConstraints(pattern=r"^RUN-[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
    question_id: QuestionId
    provider_id: Name
    category: Name
    selected: bool
    attempted: bool
    status: Literal[
        "NOT_SELECTED", "NOT_IMPLEMENTED", "BLOCKED", "FAILED", "ZERO_RESULTS", "RESULTS"
    ]
    query_count: Nonnegative = 0
    provider_result_count: Nonnegative = 0
    filtered_out_count: Nonnegative = 0
    candidate_count: Nonnegative = 0
    unique_candidate_count: Nonnegative = 0
    gap_codes: tuple[Name, ...] = ()

    @model_validator(mode="after")
    def count_consistency(self) -> SourceCoverageObservation:
        if not self.attempted and self.query_count:
            raise ValueError("UNATTEMPTED_PROVIDER_CANNOT_HAVE_QUERIES")
        if self.unique_candidate_count > self.candidate_count:
            raise ValueError("UNIQUE_CANDIDATES_EXCEED_CANDIDATES")
        if self.filtered_out_count > self.provider_result_count:
            raise ValueError("FILTERED_RESULTS_EXCEED_PROVIDER_RESULTS")
        if self.status == "NOT_SELECTED" and (self.selected or self.attempted):
            raise ValueError("NOT_SELECTED_STATUS_MISMATCH")
        return self


class FilterDecision(Contract):
    provider_id: Name
    filter_name: Name
    disposition: FilterDisposition
    mandatory: bool
    reason: Name | None = None
