"""Policy-mediated retrieval. No network client is constructed here."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from gidr.contracts.common import new_id
from gidr.contracts.sources import RetrievedDocument
from gidr.infrastructure.artifacts.quarantine import QuarantineRecord, QuarantineStore
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.discovery.transport import SourceDispatchBlocked, SourceTransport
from gidr.infrastructure.network.broker import FetchResponse


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    document: RetrievedDocument
    canonical_url: str
    quarantine: QuarantineRecord | None = None


def retry_recommendation(document: RetrievedDocument) -> str:
    """Classify retryable transport errors; scheduler remains the sole retry owner."""
    return (
        "RETRYABLE"
        if document.status == "FAILED"
        and document.error_code
        in {"FETCH_TIMEOUT", "FETCH_TRANSPORT_FAILED", "HTTP_STATUS_RETRYABLE"}
        else "TERMINAL"
    )


def canonicalize_url(url: str) -> str:
    if any(ord(char) < 33 or ord(char) == 127 for char in url):
        raise ValueError("URL_NOT_ALLOWED")
    parts = urlsplit(url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
    ):
        raise ValueError("URL_NOT_ALLOWED")
    host = (
        parts.hostname.lower()
        if ":" in parts.hostname
        else parts.hostname.encode("idna").decode("ascii").lower()
    )
    port = parts.port
    formatted_host = f"[{host}]" if ":" in host else host
    netloc = (
        formatted_host
        if port is None
        or (parts.scheme == "https" and port == 443)
        or (parts.scheme == "http" and port == 80)
        else f"{formatted_host}:{port}"
    )
    return urlunsplit((parts.scheme.lower(), netloc, parts.path or "/", parts.query, ""))


class Retriever:
    """Fetch through SourceTransport, sniff MIME, quarantine mismatches, persist bytes."""

    def __init__(
        self, transport: SourceTransport, artifacts: ArtifactStore, quarantine: QuarantineStore
    ):
        self.transport = transport
        self.artifacts = artifacts
        self.quarantine = quarantine

    def retrieve(
        self, run_id: str, source_id: str, url: str, *, headers: dict[str, str] | None = None
    ) -> RetrievalOutcome:
        canonical = canonicalize_url(url)
        try:
            response = self.transport.get(canonical, headers=headers)
        except SourceDispatchBlocked as exc:
            reason = (
                exc.reason
                if exc.reason in {"FETCH_TIMEOUT", "FETCH_TRANSPORT_FAILED"}
                else "NETWORK_DISPATCH_BLOCKED"
            )
            doc = RetrievedDocument(
                run_id=run_id,
                created_at_utc=datetime.now(UTC),
                document_id=new_id("DOC"),
                source_id=source_id,
                observation_id="OBS-" + uuid4().hex,
                fetched_at_utc=datetime.now(UTC),
                status="BLOCKED",
                error_code=reason,
            )
            return RetrievalOutcome(doc, canonical)
        except Exception:
            doc = RetrievedDocument(
                run_id=run_id,
                created_at_utc=datetime.now(UTC),
                document_id=new_id("DOC"),
                source_id=source_id,
                observation_id="OBS-" + uuid4().hex,
                fetched_at_utc=datetime.now(UTC),
                status="FAILED",
                error_code="SOURCE_FETCH_FAILED",
            )
            return RetrievalOutcome(doc, canonical)
        if not isinstance(response, FetchResponse):
            raise TypeError("TRANSPORT_RESPONSE_INVALID")
        if not 200 <= response.status < 300:
            error_code = (
                "HTTP_STATUS_RETRYABLE"
                if response.status == 429 or 500 <= response.status < 600
                else "HTTP_STATUS_NOT_SUCCESS"
            )
            doc = RetrievedDocument(
                run_id=run_id,
                created_at_utc=datetime.now(UTC),
                document_id=new_id("DOC"),
                source_id=source_id,
                observation_id="OBS-" + uuid4().hex,
                fetched_at_utc=datetime.now(UTC),
                status="FAILED",
                error_code=error_code,
            )
            return RetrievalOutcome(doc, canonical)
        declared = next(
            (v for k, v in response.headers.items() if k.casefold() == "content-type"), None
        )
        record = self.quarantine.store(response.body, declared_mime=declared)
        if record.status != "STAGED":
            doc = RetrievedDocument(
                run_id=run_id,
                created_at_utc=datetime.now(UTC),
                document_id=new_id("DOC"),
                source_id=source_id,
                observation_id="OBS-" + uuid4().hex,
                fetched_at_utc=datetime.now(UTC),
                status="BLOCKED",
                error_code="CONTENT_QUARANTINED",
            )
            return RetrievalOutcome(doc, canonical, record)
        ref = self.artifacts.publish(response.body)
        doc = RetrievedDocument(
            run_id=run_id,
            created_at_utc=datetime.now(UTC),
            document_id=new_id("DOC"),
            source_id=source_id,
            observation_id="OBS-" + uuid4().hex,
            fetched_at_utc=datetime.now(UTC),
            status="RETRIEVED",
            mime_type=record.detected_mime,
            content_hash=hashlib.sha256(response.body).hexdigest(),
            original_artifact=ref,
        )
        return RetrievalOutcome(doc, canonical, record)
