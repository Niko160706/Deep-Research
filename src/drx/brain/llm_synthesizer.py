"""Phase 2 — LLM Report Synthesis Engine (OSINT intelligence analyst).

The mechanical pipeline (Phase 1) stores cleaned documents in
``runs/<run_id>/parquet/raw_documents.parquet``. This module lifts the most
relevant segments out of that parquet, fits them into a bounded prompt
(15–20 segments **or** ~30k tokens, whichever comes first) and asks a hosted
LLM for the final structured Markdown investigation report
(``runs/<run_id>/report_llm.md``).

Provider hierarchy (extraction, critic, and synthesis share it):

* **Primary** — Groq ``openai/gpt-oss-120b`` via the OpenAI-compatible API
  (``GROQ_API_KEY``).
* **Secondary** — Groq ``qwen/qwen3.8-27b`` as the first model fallback.
* **Tertiary** — Google AI Studio via ``google-genai`` (``GEMINI_API_KEY``),
  model ``gemini-3.8-flash`` with its own comma-separated fallback chain.
  Gemini is last because its free daily quota is only ~20 requests/day.
* **OpenRouter** is available only when selected explicitly; it is not part
  of the ``auto`` chain.

Design constraints (project-wide):

* **Token efficiency** — scoring/selection/prompt building are pure Python
  (0 tokens); only the bounded evidence block ever leaves disk.
* **Graceful degradation** — missing key, missing SDK, or a network failure
  never raises: the caller gets an :class:`LLMResult` carrying
  ``skipped_reason`` / ``error`` instead, and the Phase 1 run stays valid.
* **Deterministic core** — no hidden global state; clients are injectable so
  the whole selection → prompt → report flow is unit-testable offline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from drx.brain.batching import count_tokens, mmr_indices, truncate_tokens
from drx.config import LLMConfig
from drx.models import Document
from drx.textutil import has_all_tokens, name_tokens, split_aliases, tokenize

log = logging.getLogger("drx.brain.llm_synthesizer")

#: Lima bagian wajib pada laporan final (urutannya juga wajib).
REQUIRED_SECTIONS: tuple[str, ...] = (
    "## Profil & Identitas Utama",
    "## Ringkasan Eksekutif",
    "## Rekam Jejak & Temuan Kunci",
    "## Potensi Ambigu / Entitas Serupa",
    "## Daftar Sumber / Sitasi",
)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CONTEXT_SPLIT_RE = re.compile(r"[,;|]")
_WS_RE = re.compile(r"\s+")


# ---- token estimation -------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) for budgeting, no tokenizer needed."""
    return (len(text or "") + 3) // 4


# ---- data containers --------------------------------------------------------

@dataclass(frozen=True)
class ProviderSpec:
    """A resolved LLM endpoint: which provider, model and key to use."""

    name: str  # "gemini" | "openrouter" | "groq"
    model: str
    api_key: str
    base_url: str = ""
    fallback_model: str = ""


@dataclass
class Segment:
    """One relevant evidence block with its provenance."""

    text: str
    url: str
    title: str = ""
    score: float = 0.0
    order: int = 0


class VerifiedFact(BaseModel):
    """Satu triple fakta hasil verifikasi LLM (subject-predicate-object)."""

    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_url: str = ""


class ReportSidecar(BaseModel):
    """Data terstruktur pendamping laporan (``report_sidecar.json``).

    Empat field pertama adalah skema wajib PRD; ``VerifiedFact.source_url``
    selalu di-clamp ke URL bukti yang benar-benar dipakai prompt (anti-halusinasi).
    """

    target_entity: str = ""
    verified_facts: list[VerifiedFact] = Field(default_factory=list)
    disambiguated_entities: list[str] = Field(default_factory=list)
    sources_used: list[str] = Field(default_factory=list)


@dataclass
class LLMResult:
    """Outcome of one synthesis attempt (never raises to the caller)."""

    provider: str = ""
    model: str = ""
    markdown: str = ""
    documents_used: int = 0
    segments_used: int = 0
    input_chars: int = 0
    input_tokens_est: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    sections_complete: bool = False
    used_fallback_model: bool = False
    retries: int = 0
    sidecar: ReportSidecar | None = None
    sidecar_complete: bool = False
    skipped_reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True bila laporan benar-benar dihasilkan tanpa error."""
        return bool(self.markdown) and not self.error


# ---- provider resolution ----------------------------------------------------

def _env_value(env: Mapping[str, str], *names: str) -> str:
    for name in names:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return ""


def resolve_providers(
    config: LLMConfig,
    env: Mapping[str, str] | None = None,
) -> list[ProviderSpec]:
    """Semua provider LLM yang tersedia, urut prioritas (rantai failover).

    ``auto`` (default) mengembalikan **groq → gemini** untuk setiap provider
    yang key-nya ada. Spesifikasi Groq membawa rantai model
    ``openai/gpt-oss-120b → qwen/qwen3.8-27b`` (``groq_fallback_model``);
    Gemini baru dicoba setelah seluruh model Groq gagal, karena kuota
    hariannya hanya ~20 request. OpenRouter tidak dimasukkan ke mode auto
    (hanya dipakai bila ``provider: openrouter`` dipilih eksplisit). Daftar
    kosong berarti tidak ada provider tersedia (caller skip mulus).
    """
    env = os.environ if env is None else env
    choice = (config.provider or "auto").strip().lower()
    if choice not in {"auto", "gemini", "google", "google-ai", "aistudio",
                      "openrouter", "groq"}:
        log.warning("provider LLM %r tidak dikenal -> pakai mode auto", config.provider)
        choice = "auto"

    gemini_key = _env_value(
        env, "GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"
    )
    openrouter_key = _env_value(env, "OPENROUTER_API_KEY")
    groq_key = _env_value(env, "GROQ_API_KEY")

    def _gemini() -> ProviderSpec:
        return ProviderSpec(
            name="gemini",
            model=config.model,
            api_key=gemini_key,
            fallback_model=config.fallback_model,
        )

    def _openrouter() -> ProviderSpec:
        return ProviderSpec(
            name="openrouter",
            model=config.openrouter_model,
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
        )

    def _groq() -> ProviderSpec:
        return ProviderSpec(
            name="groq",
            model=config.groq_model,
            api_key=groq_key,
            base_url=GROQ_BASE_URL,
            # PRIMARY gpt-oss-120b -> SECONDARY qwen3.8-27b.
            fallback_model=config.groq_fallback_model,
        )

    if choice in {"gemini", "google", "google-ai", "aistudio"}:
        return [_gemini()] if gemini_key else []
    if choice == "openrouter":
        return [_openrouter()] if openrouter_key else []
    if choice == "groq":
        return [_groq()] if groq_key else []

    specs: list[ProviderSpec] = []
    if groq_key:
        specs.append(_groq())
    # Gemini is the final fallback (small daily quota: ~20 requests/day).
    if gemini_key:
        specs.append(_gemini())
    # OpenRouter is deliberately excluded from auto; use provider="openrouter".
    return specs


def resolve_provider(
    config: LLMConfig,
    env: Mapping[str, str] | None = None,
) -> ProviderSpec | None:
    """Provider pertama dari :func:`resolve_providers` (API backward-compatible)."""
    specs = resolve_providers(config, env)
    return specs[0] if specs else None


# ---- parquet loading --------------------------------------------------------

def _resolve_parquet_path(path: str | Path) -> Path | None:
    """Accept the run dir, its ``parquet/`` dir, or the parquet file itself."""
    candidate = Path(path)
    if candidate.is_dir():
        for nested in (
            candidate / "parquet" / "raw_documents.parquet",
            candidate / "raw_documents.parquet",
        ):
            if nested.exists():
                return nested
        return None
    return candidate


def load_documents(parquet_path: str | Path) -> list[Document]:
    """Read the run's cleaned documents back (Phase 1 output → Phase 2 input)."""
    path = _resolve_parquet_path(parquet_path)
    if path is None or not path.exists():
        log.warning("parquet tidak ditemukan: %s", parquet_path)
        return []
    try:
        import pandas as pd

        frame = pd.read_parquet(path)
    except Exception as exc:
        log.warning("tidak bisa membaca parquet %s: %s", path, exc)
        return []
    documents: list[Document] = []
    for record in frame.to_dict("records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        try:
            documents.append(Document(**clean))
        except Exception as exc:
            log.debug("baris parquet rusak dilewati: %s", exc)
    return documents


# ---- evidence selection -----------------------------------------------------

def _split_context(context: str) -> list[str]:
    """``"Dinas Kesehatan, Kab. Sleman"`` -> two clean terms."""
    return [t.strip() for t in _CONTEXT_SPLIT_RE.split(context or "") if t.strip()]


def _document_segments(document: Document, max_chars: int) -> list[str]:
    """Split one document into ≤ ``max_chars`` chunks (lossless, deterministic)."""
    flat = _WS_RE.sub(" ", document.text or "").strip()
    if not flat:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(flat) if p.strip()]
    if not parts:
        parts = [flat]
    chunks: list[str] = []
    current = ""
    for part in parts:
        if len(part) > max_chars:
            # Blok raksasa (mis. baris PDF): pecah keras, lalu lanjut.
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


def _score_segment(text: str, aliases: list[str], context_terms: list[str]) -> float:
    """Relevance score: target name (3.0) > context term (1.5) > term overlap."""
    score = 0.0
    tokens = set(tokenize(text))
    for alias in aliases:
        alias_tokens = name_tokens(alias)
        if alias_tokens and has_all_tokens(text, alias_tokens):
            score += 3.0
            break
    for term in context_terms:
        term_tokens = tokenize(term)
        if term_tokens and has_all_tokens(text, term_tokens):
            score += 1.5
    focus: set[str] = set()
    for alias in aliases:
        focus.update(name_tokens(alias))
    for term in context_terms:
        focus.update(tokenize(term))
    if focus:
        score += len(focus & tokens) / len(focus)
    score += min(len(tokens), 50) / 100.0
    return score


def select_segments(
    documents: list[Document],
    target_entity: str,
    context: str = "",
    *,
    max_segments: int = 18,
    max_input_tokens: int = 30000,
    max_chars_per_segment: int = 2000,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
) -> list[Segment]:
    """Pick the most relevant evidence blocks within a double budget.

    Ranking is deterministic: score desc, then document/chunk order. Two hard
    caps apply — ``max_segments`` (15–20 by config) and ``max_input_tokens``
    (~30k) — whichever is reached first. When ``use_mmr`` is set, the candidate
    order is diversified with Maximal Marginal Relevance before budgeting
    (Phase 3: fewer near-duplicate chunks fit under the 3.2k guard).
    """
    aliases = split_aliases(target_entity) or [target_entity]
    context_terms = _split_context(context)

    candidates: list[Segment] = []
    order = 0
    for document in documents:
        for chunk in _document_segments(document, max_chars_per_segment):
            candidates.append(
                Segment(
                    text=chunk,
                    url=document.url,
                    title=document.title or "",
                    score=_score_segment(chunk, aliases, context_terms),
                    order=order,
                )
            )
            order += 1

    return _budget_segments(
        candidates,
        max_segments=max_segments,
        max_input_tokens=max_input_tokens,
        use_mmr=use_mmr,
        mmr_lambda=mmr_lambda,
    )


def _budget_segments(
    candidates: list[Segment],
    *,
    max_segments: int,
    max_input_tokens: int,
    use_mmr: bool = False,
    mmr_lambda: float = 0.7,
) -> list[Segment]:
    """Rank + diversify + trim a candidate list under the double budget."""
    candidates.sort(key=lambda seg: (-seg.score, seg.order))
    if use_mmr and candidates:
        order_by_mmr = mmr_indices(
            [seg.score for seg in candidates],
            [seg.text for seg in candidates],
            max_items=len(candidates),
            lambda_=mmr_lambda,
        )
        candidates = [candidates[index] for index in order_by_mmr]

    selected: list[Segment] = []
    seen: set[str] = set()
    used_tokens = 0
    for segment in candidates:
        if len(selected) >= max_segments:
            break
        key = segment.text[:80].lower()
        if key in seen:
            continue
        cost = estimate_tokens(segment.text)
        if used_tokens + cost > max_input_tokens:
            continue  # tidak cukup anggaran untuk segmen ini; coba yang lebih kecil
        seen.add(key)
        selected.append(segment)
        used_tokens += cost
    return selected


def build_evidence_segments(
    documents: list[Document],
    facts: Sequence[Mapping[str, Any]],
    *,
    target_entity: str = "",
    context: str = "",
    max_chars_per_segment: int = 2000,
) -> list[Segment]:
    """Segments HANYA dari fakta KG + chunk yang melandasinya.

    Anti-halusinasi untuk sintesis final: dokumen/chunk mentah yang tidak
    pernah menghasilkan triple tervalidasi tidak boleh masuk prompt. Untuk
    setiap fakta:

    1. cari kutipan pendukungnya di dokumen sumber dan ambil jendela teks
       di sekitar kutipan (bukti asli), atau
    2. bila kutipan tak lagi ditemukan (parquet berubah), pakai kalimat fakta
       Knowledge Graph itu sendiri — tetap berasal dari triple tervalidasi.

    Fakta tanpa URL sumber dilewati karena tidak bisa disitasi.
    """
    aliases = split_aliases(target_entity) or ([target_entity] if target_entity else [])
    context_terms = _split_context(context)
    by_url: dict[str, Document] = {}
    for document in documents:
        for key in (document.url, document.url.rstrip("/")):
            by_url.setdefault(key, document)

    segments: list[Segment] = []
    seen: set[str] = set()

    def _add(text: str, url: str, title: str) -> bool:
        clean = re.sub(r"\s+", " ", text or "").strip()
        if not clean:
            return False
        key = clean[:80].lower()
        if key in seen:
            return False
        seen.add(key)
        segments.append(
            Segment(
                text=clean,
                url=url,
                title=title,
                score=_score_segment(clean, aliases, context_terms),
                order=len(segments),
            )
        )
        return True

    def _locate(document: Document, quote: str) -> str:
        quote = re.sub(r"\s+", " ", quote or "").strip()
        if not quote:
            return ""
        text = re.sub(r"\s+", " ", document.text or "").strip()
        if not text:
            return ""
        index = text.lower().find(quote.lower())
        if index < 0:
            return ""
        start = max(0, index - 220)
        end = min(len(text), index + len(quote) + 220)
        window = text[start:end].strip()
        if max_chars_per_segment and len(window) > max_chars_per_segment:
            window = window[:max_chars_per_segment].rstrip() + " …"
        prefix = "… " if start > 0 else ""
        suffix = " …" if end < len(text) and not window.endswith("…") else ""
        return f"{prefix}{window}{suffix}"

    for fact in facts:
        subject = str(fact.get("subject") or "").strip()
        predicate = str(fact.get("predicate") or "").strip()
        obj = str(fact.get("object") or "").strip()
        if not (subject and obj):
            continue
        urls = [str(url).strip() for url in (fact.get("sources") or []) if str(url).strip()]
        if not urls:
            continue
        quotes = [str(q).strip() for q in (fact.get("quotes") or []) if str(q).strip()]
        fact_text = (
            f"{subject} — {predicate.replace('_', ' ')} — {obj}."
            if predicate
            else f"{subject} — {obj}."
        )
        grounded = False
        for quote in quotes:
            for url in urls:
                document = by_url.get(url) or by_url.get(url.rstrip("/"))
                if document is None:
                    continue
                chunk = _locate(document, quote)
                if chunk and _add(chunk, url, document.title or f"Kutipan fakta: {subject}"):
                    grounded = True
                    break
            if grounded:
                break
        if not grounded:
            _add(fact_text, urls[0], "Knowledge Graph (fakta terverifikasi)")

    return segments


# ---- prompt building --------------------------------------------------------

def build_system_prompt(language: str = "id") -> str:
    """System prompt: OSINT Intelligence Analyst + struktur laporan wajib."""
    lang_line = (
        "Bahasa laporan: Bahasa Indonesia (formal, padat)."
        if (language or "id").lower().startswith("id")
        else "Report language: English (formal, concise)."
    )
    return "\n".join(
        [
            "Anda adalah OSINT Intelligence Analyst senior.",
            "Tugas Anda: menyusun LAPORAN INVESTIGASI Markdown FINAL tentang satu",
            "target, SEMATA-MATA berdasarkan bukti yang disediakan.",
            "",
            "ATURAN KETAT:",
            "1. Hanya gunakan fakta dari blok BUKTI. Jangan menambah pengetahuan",
            "   luar, jangan mengarang nama/tanggal/angka.",
            "2. Setiap klaim WAJIB diakhiri sitasi label sumbernya, mis. [S3]",
            "   (boleh gabungan: [S1][S4]).",
            "3. Bila suatu aspek tidak didukung bukti, tulis eksplisit",
            '   "(tidak ditemukan bukti)" — jangan dikosongkan, jangan diduga.',
            "4. Nama pasaran: entri yang terbukti milik orang BERBEDA jangan",
            '   dimasukkan sebagai fakta target — catat di bagian "Potensi Ambigu".',
            "5. Jangan menyalin URL di luar daftar bukti. Jangan menulis preamble.",
            f"6. {lang_line}",
            "",
            "STRUKTUR WAJIB (judul persis seperti ini, urut, semuanya harus ada):",
            "## Profil & Identitas Utama",
            "   - poin: nama lengkap, gelar, jabatan/profesi, instansi/lokasi",
            "## Ringkasan Eksekutif",
            "   - narasi 2-3 paragraf: siapa target ini berdasarkan data",
            "## Rekam Jejak & Temuan Kunci",
            "   - poin kronologis/aktivitas penting, tiap poin bersitasi",
            "## Potensi Ambigu / Entitas Serupa",
            "   - nama/entitas mirip yang terdeteksi & alasan pembedaannya",
            "     (tulis \"Tidak ada indikasi ambiguitas\" bila bersih)",
            "## Daftar Sumber / Sitasi",
            "   - daftar [S#] URL yang benar-benar dipakai di laporan",
            "",
            "Awali laporan dengan H1: # Laporan Investigasi: <nama target>",
            "",
            "SETELAH laporan markdown selesai, tulis TANPA judul/heading tambahan,",
            "satu blok kode berpagar ```json berisi data terstruktur (sidecar):",
            '  {"target_entity": "<nama target>",',
            '   "verified_facts": [{"subject": "<entitas>", "predicate": "<relasi>",',
            '                       "object": "<nilai>", "source_url": "<URL [S#] yang mendukung>"}],',
            '   "disambiguated_entities": ["<nama/entitas mirip yang DIBEDAKAN>"],',
            '   "sources_used": ["<URL yang benar-benar dipakai>"]}',
            "Aturan sidecar: hanya fakta/URL yang muncul di BUKTI; setiap source_url",
            "harus persis salah satu URL bukti; boleh verified_facts: [] bila tidak ada",
            "triple yang terverifikasi.",
        ]
    )


def build_user_prompt(
    target_entity: str,
    context: str,
    segments: Sequence[Segment],
    *,
    retry_missing: Sequence[str] = (),
) -> str:
    """Evidence block + instructions; ``retry_missing`` menandai bagian yang bolong."""
    lines = [f"TARGET INVESTIGASI: {target_entity}"]
    if context.strip():
        lines.append(f"KONTEKS TAMBAHAN: {context.strip()}")
    lines.append("")
    lines.append("=== BUKTI ===")
    for index, segment in enumerate(segments, start=1):
        lines.append(f"[S{index}] URL: {segment.url}")
        if segment.title:
            lines.append(f"      Judul: {segment.title}")
        lines.append(f"      Teks: {segment.text}")
        lines.append("")
    lines.append("=== AKHIR BUKTI ===")
    if retry_missing:
        lines.append("")
        lines.append(
            "PERINGATAN: jawaban sebelumnya TIDAK memuat bagian wajib berikut: "
            + ", ".join(retry_missing)
            + ". Tulis ULANG laporan lengkap dengan SEMUA bagian wajib, urut, "
            "lengkap dengan sitasi [S#]."
        )
    lines.append(
        "Susun laporan investigasi final sesuai struktur wajib di system prompt."
    )
    return "\n".join(lines)


def fit_user_prompt_to_token_cap(
    system: str,
    target_entity: str,
    context: str,
    segments: list[Segment],
    max_input_tokens: int,
    *,
    retry_missing: list[str] | None = None,
) -> tuple[list[Segment], str, int]:
    """Fit evidence + prompt under the hard token cap (tiktoken-exact).

    Phase 3 guard: every LLM request must stay in the 3.0–3.5k input range
    (anti Groq 413 / Gemini 503). Segments are dropped from the tail first;
    the final fallback truncates the prompt so the cap is never exceeded.
    Returns ``(kept_segments, user_prompt, exact_token_estimate)``.
    """
    if max_input_tokens <= 0:
        user = build_user_prompt(
            target_entity, context, segments, retry_missing=retry_missing
        )
        return segments, user, count_tokens(system) + count_tokens(user)
    kept = list(segments)
    while kept:
        user = build_user_prompt(
            target_entity, context, kept, retry_missing=retry_missing
        )
        est = count_tokens(system) + count_tokens(user)
        if est <= max_input_tokens:
            return kept, user, est
        kept.pop()
    user = build_user_prompt(
        target_entity,
        context,
        segments[:1] if segments else [],
        retry_missing=retry_missing,
    )
    budget = max(120, max_input_tokens - count_tokens(system) - 8)
    user = truncate_tokens(user, budget)
    return segments[:1], user, count_tokens(system) + count_tokens(user)


def _sections_complete(markdown: str) -> bool:
    return all(section in (markdown or "") for section in REQUIRED_SECTIONS)


# ---- clients ----------------------------------------------------------------

class LLMClient(Protocol):
    """Minimal async protocol so tests can inject a fake provider."""

    name: str
    model: str

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]: ...


def model_chain(spec: ProviderSpec) -> list[str]:
    """Model yang dicoba berurutan: ``model`` dulu, lalu ``fallback_model``.

    ``fallback_model`` boleh berisi beberapa nama dipisah koma — dipakai untuk
    Gemini (``"gemini-3.7-flash, gemini-3.5-flash"``) maupun Groq
    (``"qwen/qwen3.8-27b"``). Setiap model yang 404 (retire), 503 (high
    demand), atau ditolak otomatis dilewati ke model berikutnya.
    """
    models: list[str] = []
    for candidate in [spec.model, *(spec.fallback_model or "").split(",")]:
        name = (candidate or "").strip()
        if name and name not in models:
            models.append(name)
    return models


def gemini_thinking_config(types_module: Any, level: str) -> Any:
    """``ThinkingConfig`` Gemini 3.x dari ``thinking_level`` (None bila kosong).

    Token "thinking" pada Gemini 3.x ikut memakan jatah ``max_output_tokens``
    (terbukti bisa membuat jawaban kosong saat cap kecil) — level rendah
    menjaga jatah output tetap untuk laporan. Nilai kosong/tak dikenal
    dikembalikan None supaya perilaku provider default dipakai.
    """
    value = (level or "").strip().upper()
    if not value:
        return None
    known = {
        member.name
        for member in getattr(types_module, "ThinkingLevel", ())
        if "UNSPECIFIED" not in member.name
    }
    if known and value not in known:
        log.warning(
            "thinking_level %r tidak dikenal %s -> pakai default provider",
            level,
            sorted(known),
        )
        return None
    try:
        return types_module.ThinkingConfig(thinking_level=value)
    except Exception:
        log.debug("ThinkingConfig(thinking_level=%r) tidak didukung SDK", level)
        return None


# Error provider yang berarti prompt melebihi batas token/TPM (bukan transient).
_PROMPT_TOO_LARGE_MARKERS = (
    "413",
    "request too large",
    "reduce your message size",
    "context_length_exceeded",
    "maximum context length",
    "context length",
    "itpm",
    "tokens per minute",
)


def is_prompt_too_large(error: BaseException | str) -> bool:
    """True bila error provider menandakan prompt melebihi batas token/TPM.

    Groq free tier mis. membalas 413 ``Request too large ... ITPM: Limit 7000``
    untuk satu request yang terlalu besar — solusinya memangkas prompt, bukan
    menunggu (berbeda dari 429 RPM yang sifatnya sementara).
    """
    text = str(error).lower()
    return any(marker in text for marker in _PROMPT_TOO_LARGE_MARKERS)


def prompt_size_ladder(
    base_segments: int,
    base_tokens: int,
    base_chars: int,
    *,
    token_caps: tuple[int, int] = (5000, 3000),
    min_segments: int = 3,
) -> list[tuple[int, int, int]]:
    """Tangga ukuran prompt ``(max_segments, max_input_tokens, max_chars)``.

    Tier 0 = konfigurasi penuh; tier berikutnya memangkas segmen/token/karakter
    agar provider dengan batas ITPM/TPM kecil (Groq free tier 7k) tetap bisa
    memproses. Tier duplikat (konfigurasi sudah kecil) dibuang.
    """
    ladder = [(base_segments, base_tokens, base_chars)]
    for divisor, token_cap, char_cap in (
        (2, token_caps[0], 1400),
        (3, token_caps[1], 1000),
    ):
        ladder.append(
            (
                max(1, min(base_segments, max(min_segments, base_segments // divisor))),
                min(base_tokens, token_cap),
                min(base_chars, char_cap),
            )
        )
    unique: list[tuple[int, int, int]] = []
    for tier in ladder:
        if tier not in unique:
            unique.append(tier)
    return unique


class GeminiClient:
    """Google AI Studio client (``google-genai``) with model fallback."""

    name = "gemini"

    def __init__(self, spec: ProviderSpec, config: LLMConfig) -> None:
        self.spec = spec
        self.model = spec.model
        self.temperature = config.temperature
        self.timeout = config.timeout_seconds
        self.max_output_tokens = config.max_output_tokens
        self.thinking_level = config.thinking_level
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "google-genai belum terpasang — jalankan: pip install google-genai"
                ) from exc
            kwargs: dict = {"api_key": self.spec.api_key}
            try:
                kwargs["http_options"] = types.HttpOptions(
                    timeout=int(self.timeout * 1000)
                )
            except Exception:
                log.debug("google-genai: HttpOptions(timeout) tidak didukung")
            self._client = genai.Client(**kwargs)
        return self._client

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]:
        from google.genai import types

        client = self._get_client()
        models = model_chain(self.spec)
        if not models:
            raise RuntimeError("nama model Gemini kosong — isi llm.model di settings.yaml")

        gen_config: dict = {
            "system_instruction": system,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
        }
        if json_mode:
            # Gemini native JSON mode: output dijamin objek JSON valid.
            gen_config["response_mime_type"] = "application/json"
        thinking = gemini_thinking_config(types, self.thinking_level)
        if thinking is not None:
            gen_config["thinking_config"] = thinking

        last_exc: Exception | None = None
        for model in models:
            # Bila model/versi menolak thinking_config, ulangi model yang sama
            # sekali tanpa config itu (bukan langsung membuang model).
            for attempt in (0, 1):
                config = dict(gen_config)
                if attempt == 1:
                    config.pop("thinking_config", None)
                try:
                    response = await client.aio.models.generate_content(
                        model=model,
                        contents=user,
                        config=types.GenerateContentConfig(**config),
                    )
                except Exception as exc:
                    last_exc = exc
                    if (
                        attempt == 0
                        and "thinking_config" in config
                        and "thinking" in str(exc).lower()
                    ):
                        log.info(
                            "gemini(%s): thinking_config ditolak -> retry tanpa itu",
                            model,
                        )
                        continue
                    log.warning("gemini(%s) gagal: %s", model, exc)
                    break
                try:
                    text = response.text or ""
                except Exception:
                    text = ""
                usage = getattr(response, "usage_metadata", None)
                meta = {
                    "model": model,
                    "used_fallback_model": model != self.spec.model,
                    "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
                    "output_tokens": int(
                        getattr(usage, "candidates_token_count", 0) or 0
                    ),
                    "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
                }
                return text, meta
        raise RuntimeError(
            f"Gemini gagal untuk model {models}: {last_exc}"
        ) from last_exc


class OpenAICompatClient:
    """OpenRouter / Groq client (OpenAI-compatible chat completions)."""

    def __init__(self, spec: ProviderSpec, config: LLMConfig) -> None:
        self.spec = spec
        self.name = spec.name
        self.model = spec.model
        self.temperature = config.temperature
        self.timeout = config.timeout_seconds
        self.max_output_tokens = config.max_output_tokens

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "openai client belum terpasang — jalankan: pip install openai"
            ) from exc
        models = model_chain(self.spec)
        if not models:
            raise RuntimeError(
                f"nama model {self.name} kosong — isi llm.{self.name}_model di settings.yaml"
            )
        headers = (
            {"HTTP-Referer": "https://localhost/drx", "X-Title": "DRX Deep Research"}
            if self.name == "openrouter"
            else None
        )
        client = AsyncOpenAI(
            api_key=self.spec.api_key,
            base_url=self.spec.base_url,
            timeout=self.timeout,
            max_retries=1,
            default_headers=headers,
        )
        extra: dict = {}
        if json_mode:
            # Didukung OpenRouter & Groq untuk model chat modern; bila provider
            # menolak, pemanggil (critic) otomatis retry tanpa json_mode.
            extra["response_format"] = {"type": "json_object"}
        last_exc: Exception | None = None
        for model in models:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_output_tokens,
                    **extra,
                )
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "LLM %s model %s gagal: %s -> model fallback berikutnya",
                    self.name,
                    model,
                    exc,
                )
                continue
            choice = response.choices[0] if response.choices else None
            text = (getattr(choice.message, "content", "") or "") if choice else ""
            usage = getattr(response, "usage", None)
            meta = {
                "model": model,
                "used_fallback_model": model != self.model,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
                "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
            }
            return text, meta
        raise RuntimeError(
            f"{self.name} gagal untuk model {models}: {last_exc}"
        ) from last_exc


def build_client(spec: ProviderSpec, config: LLMConfig) -> LLMClient:
    """Provider factory: Groq/OpenRouter (OpenAI-compatible) or Gemini."""
    if spec.name == "gemini":
        return GeminiClient(spec, config)
    return OpenAICompatClient(spec, config)


# ---- provider rate guard (Groq OTPM 1000) ------------------------------------

# Groq free tier membatasi output-token/menit (OTPM ~1000). Jeda ini adalah
# hard floor: konfigurasi tidak boleh menurunkannya untuk request Groq riil.
_GROQ_OTPM_MIN_INTERVAL_SECONDS = 2.0
_provider_last_call: dict[str, float] = {}


async def _throttle_provider(name: str, min_interval: float) -> None:
    """Jeda minimum antar request provider yang sama (default Groq: 2 detik).

    Groq free tier membatasi output-token per menit (OTPM ~1000) — tanpa jeda,
    request critic/ekstraksi beruntun bisa langsung 429. Timestamp disimpan per
    nama provider; ``client`` injeksi (unit test) tidak pernah melewati guard
    ini karena ``generate_with_failover`` hanya memanggilnya untuk client riil.
    """
    interval = float(min_interval or 0.0)
    if interval <= 0:
        return
    last = _provider_last_call.get(name)
    now = time.monotonic()
    if last is not None:
        wait = interval - (now - last)
        if wait > 0:
            log.info("throttle %s: jeda %.2fs (OTPM guard)", name, wait)
            await asyncio.sleep(wait)
    _provider_last_call[name] = time.monotonic()


async def generate_with_failover(
    specs: Sequence[ProviderSpec],
    config: LLMConfig,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    client: LLMClient | None = None,
) -> tuple[str, dict, ProviderSpec]:
    """Coba provider dalam ``specs`` berurutan sampai ada yang berhasil.

    Ini lapisan failover di atas rantai model Gemini: bila seluruh model Gemini
    gagal (404 sama sekali / 503 high-demand), sintesis otomatis pindah ke
    OpenRouter lalu Groq (selama key-nya ada di ``auto`` mode). ``client`` yang
    diinjeksi (test) memakai jalan pintas satu elemen. Raise ``RuntimeError``
    hanya bila *semua* percobaan gagal — pesannya memuat setiap kegagalan.
    """
    chain: list[tuple[ProviderSpec, LLMClient]]
    if client is not None:
        injected = ProviderSpec(
            name=getattr(client, "name", "injected"),
            model=getattr(client, "model", ""),
            api_key="",
        )
        chain = [(injected, client)]
    else:
        chain = [(spec, build_client(spec, config)) for spec in specs]

    errors: list[str] = []
    for spec, candidate in chain:
        if client is None and spec.name == "groq":
            # Groq free tier: OTPM ~1000 -> hard floor jeda 2 detik antar request.
            interval = max(
                _GROQ_OTPM_MIN_INTERVAL_SECONDS,
                float(config.provider_min_interval_seconds or 0.0),
            )
            await _throttle_provider(spec.name, interval)
        try:
            if json_mode:
                text, meta = await candidate.generate(system, user, json_mode=True)
            else:
                text, meta = await candidate.generate(system, user)
            if errors:
                log.info(
                    "LLM berhasil via %s setelah %d percobaan gagal",
                    spec.name,
                    len(errors),
                )
            return text, meta, spec
        except Exception as exc:
            errors.append(f"{spec.name}({spec.model}): {exc}")
            log.warning("LLM gagal via %s/%s: %s", spec.name, spec.model, exc)
    raise RuntimeError("semua provider LLM gagal -> " + " | ".join(errors))


# ---- public API -------------------------------------------------------------

async def synthesize_report(
    documents_parquet_path: str | Path,
    target_entity: str,
    context: str = "",
    *,
    config: LLMConfig | None = None,
    client: LLMClient | None = None,
    env: Mapping[str, str] | None = None,
    language: str = "id",
    evidence_facts: Sequence[Mapping[str, Any]] | None = None,
) -> LLMResult:
    """Phase 2 entry point: parquet Phase 1 → laporan investigasi Markdown LLM.

    Reads the cleaned documents, selects ≤ ``config.max_segments`` segments
    within ~``config.max_input_tokens`` tokens, and asks the configured LLM
    for the final report. Never raises — failures land in ``LLMResult.error``,
    unavailable providers in ``LLMResult.skipped_reason``.

    ``evidence_facts`` (Phase 5): bila diisi, prompt HANYA memuat fakta
    Knowledge Graph yang sudah tervalidasi + chunk yang mengutipnya. Chunk
    parquet mentah yang tidak pernah diekstrak/diverifikasi tidak dikirim.
    """
    cfg = config or LLMConfig()
    result = LLMResult()

    documents = load_documents(documents_parquet_path)
    if not documents:
        result.skipped_reason = f"tidak ada dokumen di {documents_parquet_path}"
        return result

    specs: list[ProviderSpec] = []
    if client is None:
        specs = resolve_providers(cfg, env)
        if not specs:
            result.skipped_reason = (
                "API key LLM tidak ditemukan — set GROQ_API_KEY (primary) "
                "atau GEMINI_API_KEY (fallback terakhir) di .env"
            )
            return result
        result.provider, result.model = specs[0].name, specs[0].model
    else:
        result.provider = getattr(client, "name", "injected")
        result.model = getattr(client, "model", "")

    # Phase 5 anti-halusinasi: sintesis final hanya boleh membaca fakta KG
    # yang tervalidasi + chunk yang mengutipnya (bukan seluruh parquet).
    evidence_mode = evidence_facts is not None
    evidence_segments: list[Segment] = []
    if evidence_mode:
        evidence_segments = build_evidence_segments(
            documents,
            evidence_facts or [],
            target_entity=target_entity,
            context=context,
            max_chars_per_segment=cfg.max_chars_per_segment,
        )
        if not evidence_segments:
            result.skipped_reason = (
                "tidak ada fakta Knowledge Graph yang memiliki bukti teks — "
                "sintesis final dibatasi pada fakta terverifikasi"
            )
            return result

    system = build_system_prompt(language)
    hard_cap = int(cfg.batch_max_input_tokens or 0)
    plans: list[tuple[list[Segment], str, int]] = []
    for max_segments, max_tokens, max_chars in prompt_size_ladder(
        cfg.max_segments, cfg.max_input_tokens, cfg.max_chars_per_segment
    ):
        if evidence_mode:
            tier_segments = _budget_segments(
                list(evidence_segments),
                max_segments=max_segments,
                max_input_tokens=max_tokens,
                use_mmr=cfg.use_mmr,
                mmr_lambda=cfg.mmr_lambda,
            )
        else:
            tier_segments = select_segments(
                documents,
                target_entity,
                context,
                max_segments=max_segments,
                max_input_tokens=max_tokens,
                max_chars_per_segment=max_chars,
                use_mmr=cfg.use_mmr,
                mmr_lambda=cfg.mmr_lambda,
            )
        if not tier_segments:
            continue
        tier_segments, tier_user, est = fit_user_prompt_to_token_cap(
            system, target_entity, context, tier_segments, hard_cap
        )
        if plans and est >= plans[-1][2]:
            continue  # tier tidak benar-benar lebih kecil (artefak chunking/dedupe)
        plans.append((tier_segments, tier_user, est))
    if not plans:
        result.skipped_reason = "tidak ada segmen relevan untuk dikirim ke LLM"
        return result

    # Provider non-Gemini (Groq/OpenRouter) sering berbatas ITPM/TPM kecil:
    # mulai dari tier lebih ramping agar tidak membuang satu request 413.
    start_index = 0
    if client is None and specs and specs[0].name != "gemini":
        start_index = min(1, len(plans) - 1)

    text = ""
    meta: dict = {}
    winning_spec: ProviderSpec | None = None
    segments: list[Segment] = plans[start_index][0]
    tier = start_index
    for tier in range(start_index, len(plans)):
        segments, user, est = plans[tier]
        result.input_chars = len(system) + len(user)
        result.input_tokens_est = est
        try:
            text, meta, winning_spec = await generate_with_failover(
                specs, cfg, system, user, client=client
            )
        except Exception as exc:
            if tier + 1 < len(plans) and is_prompt_too_large(exc):
                log.warning(
                    "prompt ~%d token ditolak provider (batas TPM) -> "
                    "pangkas konteks & retry",
                    est,
                )
                continue
            result.error = f"{type(exc).__name__}: {exc}"
            log.warning("LLM request gagal: %s", exc)
            return result
        result.provider = winning_spec.name
        result.retries = tier
        result.documents_used = len({segment.url for segment in segments})
        result.segments_used = len(segments)
        _apply_meta(result, meta)
        break

    if winning_spec is None:
        result.skipped_reason = "tidak ada segmen relevan untuk dikirim ke LLM"
        return result

    if not _sections_complete(text):
        missing = [s for s in REQUIRED_SECTIONS if s not in (text or "")]
        log.warning("jawaban LLM tidak memuat bagian wajib %s -> retry sekali", missing)
        result.retries = tier + 1
        _, retry_user, _ = fit_user_prompt_to_token_cap(
            system,
            target_entity,
            context,
            segments,
            hard_cap,
            retry_missing=missing,
        )
        try:
            retry_text, retry_meta, retry_spec = await generate_with_failover(
                specs,
                cfg,
                system,
                retry_user,
                client=client,
            )
            if _sections_complete(retry_text):
                text = retry_text
                result.provider = retry_spec.name
            else:
                log.warning("retry tetap tidak lengkap — laporan dipakai apa adanya")
            _apply_meta(result, retry_meta)  # token retry tetap tercatat
        except Exception as exc:
            log.warning("retry LLM gagal: %s", exc)

    body, raw_sidecar = split_llm_answer(text)
    result.sidecar, result.sidecar_complete = build_sidecar(
        raw_sidecar, target_entity, segments
    )
    result.markdown = (body or "").strip()
    result.sections_complete = _sections_complete(result.markdown)
    if not result.markdown:
        result.error = "jawaban LLM kosong"
    return result


def _apply_meta(result: LLMResult, meta: dict) -> None:
    """Copy token/model metadata into the result (akumulatif antar percobaan)."""
    if not meta:
        return
    model = str(meta.get("model") or "")
    if model:
        result.model = model
    result.used_fallback_model = bool(meta.get("used_fallback_model"))
    output_tokens = int(meta.get("output_tokens") or 0)
    result.output_tokens += output_tokens
    total = int(meta.get("total_tokens") or 0)
    if not total:
        total = result.input_tokens_est + output_tokens
    result.total_tokens += total


# ---- sidecar (structured output) --------------------------------------------

#: Batas jumlah triple di sidecar (jaga-jaga jawaban LLM membengkak).
SIDECAR_MAX_FACTS = 50

_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _parse_json_object(candidate: str) -> dict | None:
    """``json.loads`` defensif: kembalikan dict bila valid, else ``None``."""
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def extract_json_block(text: str) -> dict | None:
    """Ambil objek JSON pertama yang valid dari jawaban LLM (fenced atau polos)."""
    raw = text or ""
    candidates: list[str] = [block.strip() for block in _FENCED_BLOCK_RE.findall(raw)]
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end > start:
        candidates.append(raw[start : end + 1])
    for candidate in candidates:
        data = _parse_json_object(candidate)
        if data is not None:
            return data
    return None


def split_llm_answer(text: str) -> tuple[str, dict | None]:
    """Pisahkan laporan markdown dari blok JSON sidecar di akhir jawaban."""
    raw = (text or "").rstrip()
    for match in reversed(list(_FENCED_BLOCK_RE.finditer(raw))):
        data = _parse_json_object(match.group(1))
        if data is not None and ("verified_facts" in data or "target_entity" in data):
            body = (raw[: match.start()] + raw[match.end() :]).rstrip()
            return body, data
    return raw, None


def _clean_list(value: object, *, limit: int) -> list[str]:
    """Normalisasi list-of-str dari LLM: strip, buang kosong & duplikat, cap."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text or text in out:
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def build_sidecar(
    data: dict | None,
    target_entity: str,
    segments: Sequence[Segment],
) -> tuple[ReportSidecar, bool]:
    """Validasi sidecar LLM; semua source_url di-clamp ke URL bukti asli.

    Returns ``(sidecar, complete)`` — ``complete=False`` berarti jawaban LLM
    tidak memuat blok JSON yang bisa dipakai (sidecar fallback tetap dibuat dari
    daftar URL bukti supaya file luaran tidak pernah kosong).
    """
    evidence: list[str] = []
    for segment in segments:
        if segment.url and segment.url not in evidence:
            evidence.append(segment.url)
    evidence_set = {url.rstrip("/") for url in evidence}

    if not isinstance(data, dict):
        fallback = ReportSidecar(target_entity=target_entity, sources_used=evidence)
        return fallback, False

    facts: list[VerifiedFact] = []
    dropped = 0
    raw_facts = data.get("verified_facts")
    for item in raw_facts if isinstance(raw_facts, list) else []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        source_url = str(item.get("source_url") or "").strip()
        if source_url.rstrip("/") not in evidence_set or not (subject and predicate and obj):
            dropped += 1
            continue
        facts.append(
            VerifiedFact(
                subject=subject, predicate=predicate, object=obj, source_url=source_url
            )
        )
        if len(facts) >= SIDECAR_MAX_FACTS:
            break
    if dropped:
        log.warning(
            "sidecar: %d fakta tanpa source_url bukti dibuang (anti-halusinasi)",
            dropped,
        )

    sources = [
        url
        for url in _clean_list(data.get("sources_used"), limit=50)
        if url.rstrip("/") in evidence_set
    ]
    if not sources:
        sources = evidence
    sidecar = ReportSidecar(
        target_entity=target_entity or str(data.get("target_entity") or "").strip(),
        verified_facts=facts,
        disambiguated_entities=_clean_list(data.get("disambiguated_entities"), limit=10),
        sources_used=sources,
    )
    return sidecar, True


# ---- output helpers ---------------------------------------------------------

def format_llm_report(result: LLMResult) -> str:
    """Markdown final + footer provenance (dipakai terminal dan file)."""
    footer = (
        f"_LLM: {result.provider}/{result.model} · {result.segments_used} segmen "
        f"dari {result.documents_used} dokumen · ±{result.input_tokens_est} token "
        f"input · {result.total_tokens} token total"
    )
    if result.sidecar is not None:
        footer += f" · {len(result.sidecar.verified_facts)} fakta terverifikasi"
    footer += "_"
    return f"{result.markdown.rstrip()}\n\n---\n{footer}"


def write_llm_report(
    run_dir: str | Path,
    result: LLMResult,
    *,
    filename: str = "report_llm.md",
) -> Path | None:
    """Write ``runs/<run_id>/report_llm.md``; ``None`` bila tidak ada markdown."""
    if not result.markdown:
        return None
    path = Path(run_dir) / filename
    path.write_text(format_llm_report(result) + "\n", encoding="utf-8")
    return path


def write_report_sidecar(
    run_dir: str | Path,
    sidecar: ReportSidecar,
    *,
    filename: str = "report_sidecar.json",
) -> Path:
    """Tulis ``runs/<run_id>/report_sidecar.json`` (data Phase 1 dipertahankan).

    Bila file sudah ada (ditulis ``main.py`` dari laporan ekstraktif Phase 1),
    isi lamanya dipindahkan ke key ``phase1`` supaya tidak ada data yang hilang.
    """
    path = Path(run_dir) / filename
    data = sidecar.model_dump()
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = None
        if isinstance(existing, dict) and "verified_facts" not in existing:
            data["phase1"] = existing
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
