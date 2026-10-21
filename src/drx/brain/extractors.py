"""Phase 3 — grounded triple extraction (LLM Map-Reduce, 0 hallucinated facts).

Every triple MUST carry:

* ``source_id`` — the chunk it came from (``S1``, ``S2``, ...);
* ``short_quote`` — a verbatim snippet from that chunk.

A pure-Python validator then proves that the subject, object AND quote really
occur in the chunk text. Triples that fail are dropped **before** they reach
the graph, costing zero extra tokens and preventing name-collision
contamination (``Sri Lestari`` the nurse vs. the caterer).

Map-Reduce strategy: parquet documents are split into chunks, chunks are packed
into token-bounded batches (``batch_max_input_tokens``, default 3200), each
batch is extracted independently (map), then all triples are deduplicated and
merged (reduce). A provider failure never crashes the run.
"""
from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from drx.brain.batching import count_tokens, truncate_tokens
from drx.brain.llm_synthesizer import (
    LLMClient,
    ProviderSpec,
    extract_json_block,
    generate_with_failover,
    resolve_providers,
)
from drx.config import LLMConfig
from drx.models import Document, Fact, SourceRef

log = logging.getLogger("drx.brain.extractors")

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_MAX_SUBJECT_OBJECT = 180
_MAX_QUOTE = 400
_MIN_QUOTE = 8


# ---- schemas -----------------------------------------------------------------


class GroundedTriple(BaseModel):
    """Triple + mandatory grounding fields validated by Python."""

    subject: str
    predicate: str
    object: str
    source_id: str
    short_quote: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    source_url: str = ""  # filled from the chunk map, never trusted from the LLM


class ExtractionResult(BaseModel):
    """Outcome of one Map-Reduce extraction pass (never raises to caller)."""

    triples: list[GroundedTriple] = Field(default_factory=list)
    rejected: int = 0
    batches: int = 0
    documents: int = 0
    input_tokens_est: int = 0
    total_tokens: int = 0
    provider: str = ""
    model: str = ""
    retries: int = 0
    skipped_reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return not (self.error or self.skipped_reason)


@dataclass(frozen=True)
class EvidenceChunk:
    """One extractable evidence block with a stable in-batch source id."""

    source_id: str
    url: str
    title: str
    text: str


# ---- chunking / prompting ------------------------------------------------------


def _document_segments(document: Document, max_chars: int) -> list[str]:
    """Sentence-aware, lossless split (same behaviour as synthesis packing)."""
    flat = _WS_RE.sub(" ", document.text or "").strip()
    if not flat:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(flat) if p.strip()] or [flat]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                part[i : i + max_chars] for i in range(0, len(part), max_chars)
            )
            continue
        if current and len(current) + 1 + len(part) > max_chars:
            chunks.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        chunks.append(current)
    return chunks


def build_evidence_chunks(
    documents: Sequence[Document],
    *,
    max_chars: int = 1200,
) -> list[EvidenceChunk]:
    """Split documents into globally-numbered chunks (``S1``...``Sn``)."""
    chunks: list[EvidenceChunk] = []
    for document in documents:
        for text in _document_segments(document, max_chars):
            chunks.append(
                EvidenceChunk(
                    source_id=f"S{len(chunks) + 1}",
                    url=document.url,
                    title=document.title or "",
                    text=text,
                )
            )
    return chunks


def _batch_chunks(
    chunks: Sequence[EvidenceChunk],
    max_tokens: int,
) -> list[list[EvidenceChunk]]:
    """Greedy token-bounded Map batches (chunk overhead included)."""
    batches: list[list[EvidenceChunk]] = []
    current: list[EvidenceChunk] = []
    used = 0
    for chunk in chunks:
        cost = count_tokens(chunk.text) + 24  # URL/title/source-id overhead
        if cost > max_tokens:
            text = truncate_tokens(chunk.text, max(64, max_tokens - 24))
            chunk = EvidenceChunk(chunk.source_id, chunk.url, chunk.title, text)
            cost = count_tokens(text) + 24
        if current and used + cost > max_tokens:
            batches.append(current)
            current = []
            used = 0
        current.append(chunk)
        used += cost
    if current:
        batches.append(current)
    return batches


_EXTRACTOR_SYSTEM = """Anda adalah OSINT Triple Extractor yang bekerja HANYA dari bukti.
Tugas: ekstrak fakta subject-predicate-object dari chunk yang diberikan.

SKEMA JSON (wajib):
{"triples": [{"subject": "<entitas>", "predicate": "<relasi>",
              "object": "<nilai/entitas>", "source_id": "S1",
              "short_quote": "<kutipan verbatim 5-25 kata dari chunk>",
              "confidence": 0.0-1.0}]}

ATURAN KETAT:
1. Setiap triple WAJIB merujuk TEPAT SATU chunk lewat source_id yang ada.
2. short_quote HARUS potongan verbatim dari teks chunk tersebut.
3. subject dan object HARUS benar-benar muncul di chunk yang sama.
4. Jangan memakai pengetahuan luar, jangan mengarang nama/tanggal/jabatan.
5. Maksimum 12 triple per batch; lebih baik sedikit tapi valid.
6. Jawab HANYA JSON valid, tanpa code fence/teks lain."""


def build_extraction_system_prompt(language: str = "id") -> str:
    if (language or "id").lower().startswith("id"):
        return _EXTRACTOR_SYSTEM
    return _EXTRACTOR_SYSTEM + "\n\nWrite entity names exactly as in the chunk."


def build_extraction_prompt(
    target_entity: str,
    context: str,
    batch: Sequence[EvidenceChunk],
) -> str:
    lines = [f"TARGET INVESTIGASI: {target_entity}"]
    if context.strip():
        lines.append(f"KONTEKS TAMBAHAN: {context.strip()}")
    lines.append("")
    for chunk in batch:
        lines.append(f"=== CHUNK {chunk.source_id} ===")
        lines.append(f"URL: {chunk.url}")
        if chunk.title:
            lines.append(f"Judul: {chunk.title}")
        lines.append(f"Teks: {chunk.text}")
        lines.append("")
    lines.append(
        "Ekstrak triple yang relevan dengan target/konteks. "
        "Setiap triple wajib punya source_id + short_quote verbatim. Jawab JSON."
    )
    return "\n".join(lines)


# ---- grounding validation ------------------------------------------------------


def _ground_key(text: str) -> str:
    """Casefold + punctuation-insensitive key for substring grounding."""
    flat = (text or "").casefold()
    flat = re.sub(r"[^\w\s]", " ", flat, flags=re.UNICODE)
    return _WS_RE.sub(" ", flat).strip()


def validate_triple(
    triple: GroundedTriple,
    chunk: EvidenceChunk,
) -> tuple[bool, str]:
    """Prove subject, object and quote all exist in the chunk (0 tokens)."""
    subject = _WS_RE.sub(" ", (triple.subject or "").strip())
    obj = _WS_RE.sub(" ", (triple.object or "").strip())
    predicate = _WS_RE.sub(" ", (triple.predicate or "").strip())
    quote = _WS_RE.sub(" ", (triple.short_quote or "").strip())
    if not subject or not obj or not predicate:
        return False, "empty_field"
    if len(subject) > _MAX_SUBJECT_OBJECT or len(obj) > _MAX_SUBJECT_OBJECT:
        return False, "entity_too_long"
    if len(predicate) > 120:
        return False, "predicate_too_long"
    if triple.source_id != chunk.source_id:
        return False, "source_mismatch"
    if len(quote) < _MIN_QUOTE or len(quote) > _MAX_QUOTE:
        return False, "quote_length"
    haystack = _ground_key(chunk.text)
    if not haystack:
        return False, "empty_chunk"
    if _ground_key(subject) not in haystack:
        return False, "subject_not_in_chunk"
    if _ground_key(obj) not in haystack:
        return False, "object_not_in_chunk"
    if _ground_key(quote) not in haystack:
        return False, "quote_not_in_chunk"
    return True, ""


def _coerce_triples(data: object) -> list[dict]:
    if isinstance(data, dict):
        raw = data.get("triples") or data.get("facts") or []
    elif isinstance(data, list):
        raw = data
    else:
        raw = []
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _accumulate_meta(result: ExtractionResult, meta: dict, *, prompt_est: int) -> None:
    if not meta:
        result.input_tokens_est += prompt_est
        return
    result.model = str(meta.get("model") or result.model)
    result.input_tokens_est += int(meta.get("input_tokens") or prompt_est)
    output = int(meta.get("output_tokens") or 0)
    total = int(meta.get("total_tokens") or 0) or (prompt_est + output)
    result.total_tokens += total


# ---- public API ----------------------------------------------------------------


async def extract_triples(
    documents: Sequence[Document],
    target_entity: str,
    context: str = "",
    *,
    config: LLMConfig | None = None,
    client: LLMClient | None = None,
    env: Mapping[str, str] | None = None,
    language: str = "id",
    max_chars_per_chunk: int = 1200,
) -> ExtractionResult:
    """Map-Reduce triple extraction with strict Python grounding.

    Never raises: missing key/SDK/provider failures become ``skipped_reason``
    or ``error`` so the mechanical pipeline keeps running.
    """
    cfg = config or LLMConfig()
    result = ExtractionResult()
    if not cfg.enabled or not cfg.extraction_enabled:
        result.skipped_reason = "ekstraksi triple LLM nonaktif"
        return result

    specs: list[ProviderSpec] = []
    if client is None:
        specs = resolve_providers(cfg, env)
        if not specs:
            result.skipped_reason = "API key LLM tidak ditemukan"
            return result
        result.provider, result.model = specs[0].name, specs[0].model
    else:
        result.provider = getattr(client, "name", "injected")
        result.model = getattr(client, "model", "")

    chunks = build_evidence_chunks(
        documents, max_chars=min(max_chars_per_chunk, cfg.max_chars_per_segment)
    )
    if not chunks:
        result.skipped_reason = "tidak ada chunk yang bisa diekstrak"
        return result
    result.documents = len(documents)

    cfg_call = cfg.model_copy(
        update={
            # Groq OTPM free tier 1000 -> cap output ekstraksi 350-400 token.
            "max_output_tokens": max(350, min(400, cfg.extraction_max_output_tokens)),
            "temperature": 0.1,
        }
    )
    system = build_extraction_system_prompt(language)
    batches = _batch_chunks(chunks, cfg.batch_max_input_tokens)
    result.batches = len(batches)
    seen: set[tuple[str, str, str, str]] = set()

    for batch in batches:
        by_id = {chunk.source_id: chunk for chunk in batch}
        user = build_extraction_prompt(target_entity, context, batch)
        # HARD GUARD: system + user + retry-suffix ≤ batch_max_input_tokens.
        user_budget = max(64, cfg.batch_max_input_tokens - count_tokens(system) - 64)
        user = truncate_tokens(user, user_budget)
        prompt_est = count_tokens(system) + count_tokens(user)
        text = ""
        meta: dict = {}
        last_exc: Exception | None = None
        for attempt in (0, 1):
            prompt = user
            if attempt == 1:
                prompt += (
                    "\n\nPENTING: balas HANYA objek JSON valid "
                    '{"triples": [...]}, tanpa code fence.'
                )
            try:
                text, meta, spec = await generate_with_failover(
                    specs,
                    cfg_call,
                    system,
                    prompt,
                    json_mode=(attempt == 0),
                    client=client,
                )
                result.provider = spec.name
                result.retries += attempt
                break
            except Exception as exc:
                last_exc = exc
                if attempt == 0 and _looks_like_json_mode_error(exc):
                    log.warning(
                        "ekstraksi: json_mode ditolak provider -> retry tanpa JSON mode"
                    )
                    continue
                result.error = f"{type(last_exc).__name__}: {last_exc}"
                log.warning("ekstraksi triple gagal: %s", last_exc)
                return result

        _accumulate_meta(result, meta, prompt_est=prompt_est)
        data = extract_json_block(text)
        if data is None:
            log.warning(
                "ekstraksi: jawaban bukan JSON valid di batch %d -> dilewati",
                len(by_id),
            )
            continue
        for raw in _coerce_triples(data):
            try:
                triple = GroundedTriple(
                    subject=str(raw.get("subject") or "").strip(),
                    predicate=str(raw.get("predicate") or "").strip(),
                    object=str(raw.get("object") or "").strip(),
                    source_id=str(raw.get("source_id") or "").strip(),
                    short_quote=str(raw.get("short_quote") or "").strip(),
                    confidence=max(0.0, min(1.0, float(raw.get("confidence") or 0.6))),
                )
            except Exception:
                result.rejected += 1
                continue
            chunk = by_id.get(triple.source_id)
            if chunk is None:
                result.rejected += 1
                continue
            valid, reason = validate_triple(triple, chunk)
            if not valid:
                result.rejected += 1
                log.debug("triple dibuang (%s): %s", reason, triple.subject)
                continue
            triple.source_url = chunk.url
            key = (
                _ground_key(triple.subject),
                _ground_key(triple.predicate),
                _ground_key(triple.object),
                triple.source_url,
            )
            if key in seen:
                continue
            seen.add(key)
            result.triples.append(triple)

    log.info(
        "ekstraksi triple: %d valid, %d ditolak, %d batch, %d token",
        len(result.triples),
        result.rejected,
        result.batches,
        result.total_tokens,
    )
    return result


def _looks_like_json_mode_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("response_format", "json mode", "json_object", "response format")
    )


def triples_to_facts(triples: Sequence[GroundedTriple]) -> list[Fact]:
    """Convert grounded triples to graph facts with quote provenance."""
    facts: list[Fact] = []
    for triple in triples:
        facts.append(
            Fact(
                subject=triple.subject,
                predicate=triple.predicate,
                object=triple.object,
                confidence_score=triple.confidence,
                provenance=[
                    SourceRef(
                        url=triple.source_url,
                        source_id=triple.source_id,
                        quote=triple.short_quote,
                    )
                ],
            )
        )
    return facts
