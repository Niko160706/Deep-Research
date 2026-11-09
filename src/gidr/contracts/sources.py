"""Metadata-first source contracts; none perform URL acquisition or normalization."""

from __future__ import annotations

from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, field_validator, model_validator

from gidr.contracts.common import (
    ArtifactRef,
    ChunkId,
    Contract,
    Digest,
    DocumentId,
    ExternalIds,
    Fraction,
    Language,
    Name,
    Nonnegative,
    ObservedValue,
    Positive,
    QueryId,
    QuestionId,
    Record,
    SourceId,
    Text,
    UTCDateTime,
)

Doi = Annotated[str, StringConstraints(pattern=r"^10\.\d{4,9}/\S+$")]


class ArxivMetadata(Contract):
    kind: Literal["arxiv"] = "arxiv"
    arxiv_id: Annotated[
        str,
        StringConstraints(
            pattern=r"^(?:[0-9]{4}\.[0-9]{4,5}|[A-Za-z][A-Za-z.-]*/[0-9]{7})(?:v[1-9][0-9]*)?$"
        ),
    ]
    version: Positive | None = None
    doi: ObservedValue[Doi]
    authors: tuple[Text, ...] = ()
    primary_category: Text | None = None


class GdeltMetadata(Contract):
    kind: Literal["gdelt"] = "gdelt"
    seen_at_utc: UTCDateTime | None = None
    source_country: ObservedValue[str]


class GithubMetadata(Contract):
    kind: Literal["github"] = "github"
    repository: Annotated[str, StringConstraints(pattern=r"^[\w.-]+/[\w.-]+$")]
    ref: Text | None = None
    commit_sha: (
        Annotated[str, StringConstraints(pattern=r"^(?:[a-f0-9]{40}|[a-f0-9]{64})$")] | None
    ) = None
    inspection: Literal["REFERENCE_ONLY", "README", "TARGETED_FILE", "BOUNDED_ARCHIVE"] = (
        "REFERENCE_ONLY"
    )


class WaybackMetadata(Contract):
    kind: Literal["wayback"] = "wayback"
    original_url: Text
    capture_at_utc: UTCDateTime
    capture_url: Text


Extension = Annotated[
    ArxivMetadata | GdeltMetadata | GithubMetadata | WaybackMetadata, Field(discriminator="kind")
]


class DiscoveryProvenance(Contract):
    question_id: QuestionId
    query_id: QueryId
    method: Name
    parent_source_ids: tuple[SourceId, ...] = ()


class SourceCandidate(Record):
    source_id: SourceId
    source_type: Name
    provider_id: Name
    canonical_url: Text
    external_ids: ExternalIds = Field(default_factory=dict)
    title: Text | None
    language: Language
    discovery: DiscoveryProvenance
    raw_record_ref: ArtifactRef
    access_status: Literal["UNKNOWN", "PUBLIC", "RESTRICTED", "BLOCKED", "FAILED"]
    content_status: Literal["NOT_FETCHED", "RETRIEVED", "FAILED", "BLOCKED"] = "NOT_FETCHED"
    extension: Extension | None = None

    @field_validator("canonical_url")
    @classmethod
    def syntax_only(cls, value: str) -> str:
        # This does NOT certify SSRF safety; FetchBroker must validate DNS/redirects later.
        try:
            url = urlsplit(value)
            port = url.port
        except ValueError:
            raise ValueError("Invalid source URL") from None
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
            raise ValueError("Source URL must be HTTP(S) without credentials")
        if any(ord(c) < 33 or ord(c) == 127 for c in value) or port == 0:
            raise ValueError("Invalid source URL syntax")
        return value

    @model_validator(mode="after")
    def matching_extension(self) -> Self:
        compatible = self.extension is None or self.extension.kind == self.provider_id
        if isinstance(self.extension, WaybackMetadata) and self.provider_id == "wayback_cdx":
            compatible = True
        if not compatible:
            raise ValueError("Provider and typed extension disagree")
        return self


class RetrievedDocument(Record):
    document_id: DocumentId
    source_id: SourceId
    observation_id: Annotated[str, StringConstraints(pattern=r"^OBS-[A-Za-z0-9_-]+$")]
    fetched_at_utc: UTCDateTime
    status: Literal["RETRIEVED", "FAILED", "BLOCKED"]
    mime_type: Text | None = None
    content_hash: Digest | None = None
    original_artifact: ArtifactRef | None = None
    error_code: Name | None = None

    @model_validator(mode="after")
    def valid_result(self) -> Self:
        if self.status == "RETRIEVED":
            if self.original_artifact is None or self.mime_type is None:
                raise ValueError("Retrieved document requires original artifact and MIME")
            if self.content_hash != self.original_artifact.sha256 or self.error_code is not None:
                raise ValueError("Retrieved document requires matching hash and no error")
        elif (
            self.error_code is None
            or self.original_artifact is not None
            or self.content_hash is not None
        ):
            raise ValueError("Failed or blocked retrieval cannot masquerade as a document")
        return self


class TextLocator(Contract):
    unit: Literal["CHAR", "BYTE"]
    start: Nonnegative
    end: Positive
    page: Positive | None = None
    section: Text | None = None
    table: Nonnegative | None = None
    row: Nonnegative | None = None
    column: Nonnegative | None = None

    @model_validator(mode="after")
    def valid_span(self) -> Self:
        if self.end <= self.start:
            raise ValueError("Locator must identify a nonempty span")
        return self


class ContentChunk(Record):
    chunk_id: ChunkId
    document_id: DocumentId
    original_locator: TextLocator
    language: Language
    extracted_text_ref: ArtifactRef
    parser_id: Name
    parser_version: Text
    parser_quality: Fraction | None = None
    quality_flags: tuple[Name, ...] = ()
