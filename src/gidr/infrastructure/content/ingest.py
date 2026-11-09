"""Bounded retrieval-to-content orchestration; all egress stays in SourceTransport."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from gidr.contracts.common import new_id
from gidr.contracts.sources import TextLocator
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.content.registry import (
    ChunkPublication,
    ContentReceipt,
    ContentRegistry,
    PublishContent,
)
from gidr.infrastructure.network.retrieval import RetrievalOutcome, Retriever
from gidr.infrastructure.parsing.chunks import chunk_text, chunk_text_by_page
from gidr.infrastructure.parsing.html import extract_html, extract_plain_text
from gidr.infrastructure.parsing.pdf import PdfParserUnavailable, extract_pdf
from gidr.infrastructure.parsing.sandbox import ParserSandbox


@dataclass(frozen=True, slots=True)
class IngestionResult:
    status: str
    retrieval: RetrievalOutcome
    content: ContentReceipt | None = None
    reason_code: str | None = None


class ContentIngestor:
    """Fetch, parse and publish an immutable source snapshot and lexical chunks."""

    def __init__(
        self,
        retriever: Retriever,
        artifacts: ArtifactStore,
        registry: ContentRegistry,
        *,
        pdf_sandbox: ParserSandbox | None = None,
        pdf_executable: Path = Path("/usr/bin/pdftotext"),
    ):
        self.retriever = retriever
        self.artifacts = artifacts
        self.registry = registry
        self.pdf_sandbox = pdf_sandbox
        self.pdf_executable = pdf_executable

    def _record_unparsed(self, retrieval: RetrievalOutcome, reason: str) -> ContentReceipt:
        document = retrieval.document
        return self.registry.record_unparsed(
            new_id("CMD"),
            new_id("TRACE"),
            document,
            retrieval.canonical_url,
            reason,
        )

    def ingest(self, run_id: str, source_id: str, url: str) -> IngestionResult:
        retrieval = self.retriever.retrieve(run_id, source_id, url)
        document = retrieval.document
        if document.status != "RETRIEVED" or document.original_artifact is None:
            self.registry.record_failure(
                new_id("CMD"), new_id("TRACE"), document, retrieval.canonical_url
            )
            return IngestionResult(status="RETRIEVAL_FAILURE_RECORDED", retrieval=retrieval)
        raw = self.artifacts.read(document.original_artifact)
        try:
            if document.mime_type == "text/html":
                extracted = extract_html(raw)
            elif document.mime_type == "text/plain":
                extracted = extract_plain_text(raw)
            elif document.mime_type == "application/pdf":
                if retrieval.quarantine is None:
                    raise PdfParserUnavailable("PDF_QUARANTINE_RECORD_MISSING")
                extracted = extract_pdf(
                    raw,
                    sandbox=self.pdf_sandbox,
                    input_path=retrieval.quarantine.path,
                    executable=str(self.pdf_executable),
                )
            else:
                receipt = self._record_unparsed(retrieval, "MIME_NOT_SUPPORTED_FOR_TEXT_EXTRACTION")
                return IngestionResult(
                    status="PARSER_UNAVAILABLE",
                    retrieval=retrieval,
                    content=receipt,
                    reason_code="MIME_NOT_SUPPORTED_FOR_TEXT_EXTRACTION",
                )
        except PdfParserUnavailable as exc:
            receipt = self._record_unparsed(retrieval, exc.reason)
            return IngestionResult(
                status="PARSER_UNAVAILABLE",
                retrieval=retrieval,
                content=receipt,
                reason_code=exc.reason,
            )
        if not extracted.text:
            receipt = self._record_unparsed(retrieval, "EMPTY_EXTRACTED_TEXT")
            return IngestionResult(
                status="PARSER_REVIEW_REQUIRED",
                retrieval=retrieval,
                content=receipt,
                reason_code="EMPTY_EXTRACTED_TEXT",
            )
        if len(extracted.source_spans) != len(extracted.text):
            receipt = self._record_unparsed(retrieval, "SOURCE_LOCATOR_UNAVAILABLE")
            return IngestionResult(
                status="PARSER_REVIEW_REQUIRED",
                retrieval=retrieval,
                content=receipt,
                reason_code="SOURCE_LOCATOR_UNAVAILABLE",
            )
        drafts = (
            chunk_text_by_page(extracted.text, extracted.source_pages)
            if extracted.source_pages
            else chunk_text(extracted.text)
        )
        publications: list[ChunkPublication] = []
        for draft in drafts:
            if not draft.text.strip():
                continue
            text_ref = self.artifacts.publish(draft.text.encode("utf-8"))
            mapped = [
                span
                for span in extracted.source_spans[draft.start : draft.end]
                if span[1] > span[0]
            ]
            if not mapped:
                continue
            locator_start, locator_end = mapped[0][0], mapped[-1][1]
            page = (
                extracted.source_pages[draft.start]
                if extracted.source_pages and draft.start < len(extracted.source_pages)
                else None
            )
            locator = TextLocator(unit="CHAR", start=locator_start, end=locator_end, page=page)
            publications.append(
                ChunkPublication(
                    chunk_id=new_id("CHK"),
                    artifact=text_ref,
                    text=draft.text,
                    locator_json=json.dumps(locator.model_dump(mode="json"), separators=(",", ":")),
                    language=draft.language,
                    parser_id=extracted.parser_id,
                    parser_version=extracted.parser_version,
                    quality=extracted.quality,
                    quality_flags=extracted.quality_flags,
                )
            )
        if not publications:
            receipt = self._record_unparsed(retrieval, "NO_INDEXABLE_CHUNKS")
            return IngestionResult(
                status="PARSER_REVIEW_REQUIRED",
                retrieval=retrieval,
                content=receipt,
                reason_code="NO_INDEXABLE_CHUNKS",
            )
        receipt = self.registry.publish(
            new_id("CMD"),
            new_id("TRACE"),
            PublishContent(
                run_id=run_id,
                source_id=source_id,
                document_id=document.document_id,
                observation_id=document.observation_id,
                canonical_url=retrieval.canonical_url,
                original_artifact=document.original_artifact,
                mime_type=document.mime_type,
                language=extracted.language,
                fetched_at_utc=document.fetched_at_utc,
                chunks=tuple(publications),
            ),
        )
        return IngestionResult(status="INDEXED", retrieval=retrieval, content=receipt)
