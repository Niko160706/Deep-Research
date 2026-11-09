"""Atomic publication of retrieved documents, chunks, provenance and lexical rows."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import Field

from gidr.contracts.common import (
    ArtifactRef,
    ChunkId,
    Contract,
    DocumentId,
    RunId,
    SourceId,
    Text,
)
from gidr.contracts.sources import RetrievedDocument, TextLocator
from gidr.infrastructure.artifacts.repository import ArtifactRepository, ReferenceRequest
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.content.identity import (
    NormalizedIdentity,
    normalize_doi,
    normalize_identity,
)
from gidr.infrastructure.db.authority import PersistenceAuthority, WriteContext
from gidr.infrastructure.db.index import IndexResult, remove_rows


class ChunkPublication(Contract):
    chunk_id: ChunkId
    artifact: ArtifactRef
    text: str = Field(min_length=1, max_length=262144)
    locator_json: str = Field(max_length=65536)
    language: Text
    parser_id: Text
    parser_version: Text
    quality: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    quality_flags: tuple[Text, ...] = Field(default=(), max_length=64)


class PublishContent(Contract):
    run_id: RunId
    source_id: SourceId
    document_id: DocumentId
    observation_id: Text
    canonical_url: Text
    title: str = Field(default="", max_length=4096)
    snippet: str = Field(default="", max_length=8192)
    language: Text = "und"
    original_artifact: ArtifactRef
    mime_type: Text
    fetched_at_utc: datetime
    chunks: tuple[ChunkPublication, ...] = Field(max_length=256)


class ContentReceipt(Contract):
    document_id: DocumentId
    observation_id: Text
    chunk_ids: tuple[ChunkId, ...]
    alias_added: bool


class RetrievalFailure(Contract):
    document: RetrievedDocument
    canonical_url: Text


class UnparsedContent(Contract):
    document: RetrievedDocument
    canonical_url: Text
    reason_code: Text


class RegisterSourceIdentity(Contract):
    source_id: SourceId
    identity: NormalizedIdentity


class SourceIdentityReceipt(Contract):
    source_id: SourceId
    related_source_ids: tuple[SourceId, ...]
    snapshot_parent_id: SourceId | None = None


class ContentRegistry:
    """Relational content/provenance and FTS publish through one state transaction."""

    def __init__(self, authority: PersistenceAuthority, artifacts: ArtifactRepository):
        self.authority = authority
        self.artifacts = artifacts

    def publish(
        self,
        command_id: str,
        trace_id: str,
        request: PublishContent,
        *,
        fault: Callable[[str], None] = lambda _: None,
    ) -> ContentReceipt:
        if request.original_artifact.sha256 != request.original_artifact.artifact_id.removeprefix(
            "ART-"
        ):
            raise ValueError("ORIGINAL_ARTIFACT_HASH_MISMATCH")
        if len({chunk.chunk_id for chunk in request.chunks}) != len(request.chunks):
            raise ValueError("DUPLICATE_CHUNK_ID")
        refs = (request.original_artifact, *(chunk.artifact for chunk in request.chunks))
        with self.artifacts.prepared(refs):

            def change(context: WriteContext) -> ContentReceipt:
                source = context.execute(
                    "SELECT run_id,canonical_url,title FROM source_candidates WHERE source_id=?",
                    (request.source_id,),
                ).fetchone()
                if source is None or source["run_id"] != request.run_id:
                    raise ValueError("CONTENT_SOURCE_RUN_MISMATCH")
                self._persist_identity(
                    context,
                    request.source_id,
                    normalize_identity(request.canonical_url),
                    request.fetched_at_utc,
                )
                external_id = context.execute(
                    "SELECT external_id FROM source_candidates WHERE source_id=?",
                    (request.source_id,),
                ).fetchone()[0]
                if external_id:
                    try:
                        doi = normalize_doi(external_id)
                    except ValueError:
                        doi = None
                    if doi is not None:
                        self._persist_identity(
                            context,
                            request.source_id,
                            NormalizedIdentity("DOI", doi),
                            request.fetched_at_utc,
                        )
                self.artifacts.register(context, request.original_artifact)
                context.execute(
                    "INSERT INTO source_observations(observation_id,source_id,run_id,artifact_id,observed_at_utc,access_status) VALUES(?,?,?,?,?,'RETRIEVED')",
                    (
                        request.observation_id,
                        request.source_id,
                        request.run_id,
                        request.original_artifact.artifact_id,
                        request.fetched_at_utc.isoformat(),
                    ),
                )
                context.execute(
                    "INSERT INTO documents(document_id,source_id,observation_id,original_artifact_id,mime_type,sha256,retrieval_status,availability,fetched_at_utc,language) VALUES(?,?,?,?,?,?,'RETRIEVED','AVAILABLE',?,?)",
                    (
                        request.document_id,
                        request.source_id,
                        request.observation_id,
                        request.original_artifact.artifact_id,
                        request.mime_type,
                        request.original_artifact.sha256,
                        request.fetched_at_utc.isoformat(),
                        request.language,
                    ),
                )
                self.artifacts.attach(
                    context,
                    ReferenceRequest(
                        reference=request.original_artifact,
                        owner_type="DOCUMENT",
                        owner_id=request.document_id,
                        role="ORIGINAL_SOURCE",
                    ),
                )
                cursor = context.execute(
                    "INSERT OR IGNORE INTO source_aliases(source_id,alias_url) VALUES(?,?)",
                    (request.source_id, request.canonical_url),
                )
                alias_added = cursor.rowcount > 0
                title = request.title or source["title"] or ""
                for chunk in request.chunks:
                    try:
                        locator_data = json.loads(chunk.locator_json)
                        locator = TextLocator.model_validate(locator_data)
                    except (ValueError, TypeError):
                        raise ValueError("CHUNK_LOCATOR_INVALID") from None
                    if locator.unit != "CHAR":
                        raise ValueError("CHUNK_LOCATOR_UNIT_UNSUPPORTED")
                    if locator.page is None and locator.end > request.original_artifact.size_bytes:
                        raise ValueError("CHUNK_LOCATOR_OUTSIDE_DOCUMENT")
                    self.artifacts.register(context, chunk.artifact)
                    context.execute(
                        "INSERT INTO chunks(chunk_id,document_id,text_artifact_id,locator_json,language,parser_id,parser_version,quality,quality_flags_json) VALUES(?,?,?,?,?,?,?,?,?)",
                        (
                            chunk.chunk_id,
                            request.document_id,
                            chunk.artifact.artifact_id,
                            chunk.locator_json,
                            chunk.language,
                            chunk.parser_id,
                            chunk.parser_version,
                            chunk.quality,
                            json.dumps(chunk.quality_flags, separators=(",", ":")),
                        ),
                    )
                    self.artifacts.attach(
                        context,
                        ReferenceRequest(
                            reference=chunk.artifact,
                            owner_type="CHUNK",
                            owner_id=chunk.chunk_id,
                            role="EXTRACTED_TEXT",
                        ),
                    )
                    context.execute(
                        "INSERT INTO source_index(source_id,chunk_id,title,snippet,extracted_text) VALUES(?,?,?,?,?)",
                        (
                            request.source_id,
                            chunk.chunk_id,
                            title[:4096],
                            request.snippet or chunk.text[:8192],
                            chunk.text,
                        ),
                    )
                    rowid = context.execute("SELECT last_insert_rowid()").fetchone()[0]
                    context.execute(
                        "INSERT INTO source_fts(rowid,title,snippet,extracted_text) VALUES(?,?,?,?)",
                        (rowid, title[:4096], request.snippet or chunk.text[:8192], chunk.text),
                    )
                context.emit(
                    event_for(
                        context,
                        component="content",
                        event_type="content.published",
                        aggregate_id=request.document_id,
                        run_id=request.run_id,
                    )
                )
                return ContentReceipt(
                    document_id=request.document_id,
                    observation_id=request.observation_id,
                    chunk_ids=tuple(chunk.chunk_id for chunk in request.chunks),
                    alias_added=alias_added,
                )

            return self.authority.execute(
                command_id, trace_id, request, ContentReceipt, change, fault=fault
            )

    def record_failure(
        self,
        command_id: str,
        trace_id: str,
        document: RetrievedDocument,
        canonical_url: str,
    ) -> None:
        if document.status == "RETRIEVED":
            raise ValueError("RETRIEVED_DOCUMENT_IS_NOT_FAILURE")
        request = RetrievalFailure(document=document, canonical_url=canonical_url)

        class FailureReceipt(Contract):
            observation_id: Text

        def change(context: WriteContext) -> FailureReceipt:
            source = context.execute(
                "SELECT run_id FROM source_candidates WHERE source_id=?", (document.source_id,)
            ).fetchone()
            if source is None or source["run_id"] != document.run_id:
                raise ValueError("CONTENT_SOURCE_RUN_MISMATCH")
            context.execute(
                "INSERT INTO source_observations(observation_id,source_id,run_id,artifact_id,observed_at_utc,access_status) VALUES(?,?,?,NULL,?,?)",
                (
                    document.observation_id,
                    document.source_id,
                    document.run_id,
                    document.fetched_at_utc.isoformat(),
                    document.status,
                ),
            )
            context.execute(
                "INSERT INTO documents(document_id,source_id,observation_id,original_artifact_id,mime_type,sha256,retrieval_status,availability,fetched_at_utc) VALUES(?,?,?,NULL,NULL,NULL,?,?,?)",
                (
                    document.document_id,
                    document.source_id,
                    document.observation_id,
                    document.status,
                    "QUARANTINED" if document.error_code == "CONTENT_QUARANTINED" else "MISSING",
                    document.fetched_at_utc.isoformat(),
                ),
            )
            context.emit(
                event_for(
                    context,
                    component="retrieval",
                    event_type=f"content.retrieval_{document.status.casefold()}",
                    aggregate_id=document.document_id,
                    run_id=document.run_id,
                )
            )
            return FailureReceipt(observation_id=document.observation_id)

        self.authority.execute(command_id, trace_id, request, FailureReceipt, change)

    def record_unparsed(
        self,
        command_id: str,
        trace_id: str,
        document: RetrievedDocument,
        canonical_url: str,
        reason_code: str,
    ) -> ContentReceipt:
        if document.status != "RETRIEVED" or document.original_artifact is None:
            raise ValueError("UNPARSED_CONTENT_REQUIRES_RETRIEVED_BYTES")
        original_artifact = document.original_artifact
        request = UnparsedContent(
            document=document, canonical_url=canonical_url, reason_code=reason_code
        )
        with self.artifacts.prepared((original_artifact,)):

            def change(context: WriteContext) -> ContentReceipt:
                source = context.execute(
                    "SELECT run_id FROM source_candidates WHERE source_id=?",
                    (document.source_id,),
                ).fetchone()
                if source is None or source["run_id"] != document.run_id:
                    raise ValueError("CONTENT_SOURCE_RUN_MISMATCH")
                self._persist_identity(
                    context,
                    document.source_id,
                    normalize_identity(canonical_url),
                    document.fetched_at_utc,
                )
                self.artifacts.register(context, original_artifact)
                context.execute(
                    "INSERT INTO source_observations(observation_id,source_id,run_id,artifact_id,observed_at_utc,access_status) VALUES(?,?,?,?,?,'RETRIEVED')",
                    (
                        document.observation_id,
                        document.source_id,
                        document.run_id,
                        original_artifact.artifact_id,
                        document.fetched_at_utc.isoformat(),
                    ),
                )
                context.execute(
                    "INSERT INTO documents(document_id,source_id,observation_id,original_artifact_id,mime_type,sha256,retrieval_status,availability,fetched_at_utc) VALUES(?,?,?,?,?,?,'RETRIEVED','AVAILABLE',?)",
                    (
                        document.document_id,
                        document.source_id,
                        document.observation_id,
                        original_artifact.artifact_id,
                        document.mime_type,
                        document.content_hash,
                        document.fetched_at_utc.isoformat(),
                    ),
                )
                self.artifacts.attach(
                    context,
                    ReferenceRequest(
                        reference=original_artifact,
                        owner_type="DOCUMENT",
                        owner_id=document.document_id,
                        role="ORIGINAL_SOURCE",
                    ),
                )
                alias = context.execute(
                    "INSERT OR IGNORE INTO source_aliases(source_id,alias_url) VALUES(?,?)",
                    (document.source_id, canonical_url),
                )
                context.emit(
                    event_for(
                        context,
                        component="content",
                        event_type="content.parser_review_required",
                        aggregate_id=document.document_id,
                        run_id=document.run_id,
                    )
                )
                return ContentReceipt(
                    document_id=document.document_id,
                    observation_id=document.observation_id,
                    chunk_ids=(),
                    alias_added=alias.rowcount > 0,
                )

            return self.authority.execute(command_id, trace_id, request, ContentReceipt, change)

    def record_identity(
        self, command_id: str, trace_id: str, source_id: SourceId, value: str
    ) -> SourceIdentityReceipt:
        """Record DOI/URL/Wayback identity and auditable relationships, never merge rows."""
        self.authority.policy.validate_text(value)
        normalized = normalize_identity(value)
        request = RegisterSourceIdentity(source_id=source_id, identity=normalized)

        def change(context: WriteContext) -> SourceIdentityReceipt:
            source = context.execute(
                "SELECT source_id FROM source_candidates WHERE source_id=?", (source_id,)
            ).fetchone()
            if source is None:
                raise ValueError("SOURCE_IDENTITY_SOURCE_UNKNOWN")
            related, snapshot_parent_id = self._persist_identity(
                context, source_id, normalized, datetime.now(UTC)
            )
            context.emit(
                event_for(
                    context,
                    component="content",
                    event_type="source.identity_recorded",
                    aggregate_id=source_id,
                )
            )
            return SourceIdentityReceipt(
                source_id=source_id,
                related_source_ids=related,
                snapshot_parent_id=snapshot_parent_id,
            )

        return self.authority.execute(command_id, trace_id, request, SourceIdentityReceipt, change)

    @staticmethod
    def _persist_identity(
        context: WriteContext,
        source_id: str,
        identity: NormalizedIdentity,
        created_at: datetime,
    ) -> tuple[tuple[SourceId, ...], SourceId | None]:
        related = tuple(
            row[0]
            for row in context.execute(
                "SELECT source_id FROM source_identities WHERE identity_type=? AND identity_value=? AND source_id<>? ORDER BY source_id",
                (identity.kind, identity.value, source_id),
            ).fetchall()
        )
        context.execute(
            "INSERT OR IGNORE INTO source_identities(identity_type,identity_value,source_id,created_at_utc) VALUES(?,?,?,?)",
            (identity.kind, identity.value, source_id, created_at.isoformat()),
        )
        for other_source_id in related:
            context.execute(
                "INSERT OR IGNORE INTO source_dependencies(from_source_id,to_source_id,kind,checking_method,certainty) VALUES(?,?,'SAME_IDENTITY',?,'CONFIRMED')",
                (source_id, other_source_id, f"IDENTITY_{identity.kind}"),
            )

        snapshot_parent_id = None
        if identity.snapshot_parent_url is not None:
            parent = context.execute(
                "SELECT source_id FROM source_candidates WHERE canonical_url=? AND source_id<>? UNION SELECT source_id FROM source_aliases WHERE alias_url=? AND source_id<>? ORDER BY source_id LIMIT 1",
                (
                    identity.snapshot_parent_url,
                    source_id,
                    identity.snapshot_parent_url,
                    source_id,
                ),
            ).fetchone()
            if parent is not None:
                snapshot_parent_id = parent[0]
                context.execute(
                    "INSERT OR IGNORE INTO source_dependencies(from_source_id,to_source_id,kind,checking_method,certainty) VALUES(?,?,'SNAPSHOT_OF','WAYBACK_CAPTURE_URL','CONFIRMED')",
                    (source_id, snapshot_parent_id),
                )
        return related, snapshot_parent_id

    def remove_index(self, command_id: str, trace_id: str, source_id: SourceId) -> IndexResult:
        """Delete lexical rows by source atomically while preserving content/provenance."""

        class Removal(Contract):
            source_id: SourceId

        request = Removal(source_id=source_id)

        def change(context: WriteContext) -> IndexResult:
            rows = context.execute(
                "SELECT DISTINCT chunk_id FROM source_index WHERE source_id=?", (source_id,)
            ).fetchall()
            for row in rows:
                remove_rows(context, source_id, row["chunk_id"])
            context.emit(
                event_for(
                    context,
                    component="indexing",
                    event_type="index.removed",
                    aggregate_id=source_id,
                )
            )
            return IndexResult(count=0)

        return self.authority.execute(command_id, trace_id, request, IndexResult, change)
