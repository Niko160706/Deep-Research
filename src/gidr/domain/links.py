"""Pure referential validation; no storage, extraction or evidence promotion."""

from gidr.contracts.common import ArtifactRef, Record
from gidr.contracts.evidence import Claim, EvidenceRecord
from gidr.contracts.models import TranslationArtifact
from gidr.contracts.sources import ContentChunk, RetrievedDocument, SourceCandidate


def validate_artifact_links(
    *,
    sources: tuple[SourceCandidate, ...],
    documents: tuple[RetrievedDocument, ...],
    chunks: tuple[ContentChunk, ...],
    claims: tuple[Claim, ...],
    evidence: tuple[EvidenceRecord, ...],
    translations: tuple[TranslationArtifact, ...] = (),
) -> None:
    """Validate one run's complete candidate snapshot. Raises without mutating it."""
    records: tuple[Record, ...] = (*sources, *documents, *chunks, *claims, *evidence, *translations)
    if len({record.run_id for record in records}) > 1:
        raise ValueError("Cross-run snapshot requires an explicit reuse mapping")

    def unique(values: tuple[str, ...]) -> set[str]:
        if len(set(values)) != len(values):
            raise ValueError("Duplicate identity in snapshot")
        return set(values)

    source_ids = unique(tuple(s.source_id for s in sources))
    document_ids = unique(tuple(d.document_id for d in documents))
    chunk_ids = unique(tuple(c.chunk_id for c in chunks))
    claim_ids = unique(tuple(c.claim_id for c in claims))
    unique(tuple(e.evidence_id for e in evidence))
    unique(tuple(t.translation_id for t in translations))
    if any(d.source_id not in source_ids for d in documents):
        raise ValueError("Orphan document")
    valid_documents = {d.document_id for d in documents if d.status == "RETRIEVED"}
    if any(
        c.document_id not in document_ids or c.document_id not in valid_documents for c in chunks
    ):
        raise ValueError("Chunk requires successfully retrieved original document")
    by_chunk = {c.chunk_id: c for c in chunks}
    for item in evidence:
        if item.claim_id not in claim_ids or item.chunk_id not in chunk_ids:
            raise ValueError("Orphan evidence")
        locator = by_chunk[item.chunk_id].original_locator
        if (
            item.original_locator.unit != locator.unit
            or item.original_locator.start < locator.start
            or item.original_locator.end > locator.end
            or item.original_locator.page != locator.page
        ):
            raise ValueError("Evidence locator lies outside original chunk")
        if not set(item.dependency_source_ids) <= source_ids:
            raise ValueError("Evidence has unknown dependency source")
    for translation in translations:
        if translation.chunk_id not in chunk_ids:
            raise ValueError("Orphan translation")
        original: ArtifactRef = by_chunk[translation.chunk_id].extracted_text_ref
        if (
            original != translation.original_text_ref
            or by_chunk[translation.chunk_id].language != translation.source_language
        ):
            raise ValueError("Translation lost original chunk provenance")
