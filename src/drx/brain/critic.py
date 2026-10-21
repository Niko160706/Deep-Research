"""Phase 2 — LLM Critic & Gap Analyzer (OSINT Fact Verifier).

Setelah Phase 1 mengumpulkan dan membersihkan dokumen, critic menilai apakah
bukti sudah CUKUP untuk laporan final: apakah identitas target jelas, apakah ada
kontradiksi/alibi antar segmen, dan fakta apa yang masih bolong. Hasilnya adalah
verdict JSON (pydantic) + daftar sub-kueri gap yang dijalankan orchestrator
dalam *agentic critic loop* sebelum sintesis.

Prinsip yang sama dengan ``llm_synthesizer``:

* **Token efficiency** — segmen dipilih deterministik di Python; hanya
  ``critic_max_segments`` (default 12) segmen / ~``critic_max_input_tokens``
  yang dikirim LLM.
* **Graceful degradation** — tanpa API key / error jaringan, fungsi tidak
  pernah melempar exception: kembalian membawa ``skipped_reason``/``error``
  sehingga loop investigasi tetap selesai.
* **Anti-loop mati** — verdict yang tidak bisa diparse sebagai JSON membuat
  orchestrator berhenti beriterasi, bukan mengulang tanpa arah.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, Field

from drx.brain.batching import count_tokens, truncate_tokens
from drx.brain.llm_synthesizer import (
    LLMClient,
    ProviderSpec,
    Segment,
    extract_json_block,
    generate_with_failover,
    is_prompt_too_large,
    load_documents,
    prompt_size_ladder,
    resolve_providers,
    select_segments,
)
from drx.config import LLMConfig

log = logging.getLogger("drx.brain.critic")


class SuggestedNode(BaseModel):
    """Saran pivot dari critic: node_id harus ada di graph (grounded)."""

    node_id: str
    label: str = ""
    reason: str = ""


class CriticResult(BaseModel):
    """Verdict kecukupan bukti (JSON LLM) + metadata eksekusi Python.

    Tiga field pertama adalah skema wajib yang diminta dari LLM; sisanya
    diisi runtime untuk observability (token, provider, alasan skip/error).
    """

    is_sufficient: bool = False
    critique_summary: str = ""
    suggested_sub_queries: list[str] = Field(default_factory=list)
    # ---- Phase 3: triple-JSON critic extras ----
    contradictions: list[str] = Field(default_factory=list)
    suggested_nodes: list[SuggestedNode] = Field(default_factory=list)
    # ---- metadata (bukan bagian skema LLM) ----
    provider: str = ""
    model: str = ""
    documents_used: int = 0
    segments_used: int = 0
    batches: int = 0
    nodes_used: int = 0
    input_tokens_est: int = 0
    total_tokens: int = 0
    retries: int = 0
    skipped_reason: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        """True bila verdict LLM benar-benar diperoleh tanpa error."""
        return not (self.error or self.skipped_reason)


_SYSTEM_PROMPT = """Anda adalah OSINT Fact Verifier senior dalam pipeline investigasi.
Tugas Anda: menilai apakah bukti yang terkumpul sudah CUKUP untuk menulis
laporan investigasi final tentang satu target, dan bila belum, sebutkan gap
datanya.

Periksa tiga hal:
1. IDENTITAS — apakah target jelas dan konsisten (nama lengkap, gelar,
   jabatan/profesi, instansi/lokasi)? Apakah ada entri milik orang lain
   (nama pasaran) yang ikut tercampur?
2. KONTRADIKSI / ALIBI — apakah ada fakta antar segmen yang saling bertentangan
   (tanggal, jabatan, lokasi, kronologi)?
3. GAP DATA — fakta penting apa yang masih bolong dan layak dicari: tanggal
   lahir, jabatan saat ini, riwayat, relasi, verifikasi silang?

ATURAN:
- Jawab HANYA satu objek JSON valid; tanpa teks pembuka/penutup, tanpa code
  fence, tanpa penjelasan tambahan.
- is_sufficient bernilai true hanya bila identitas target jelas, tidak ada
  kontradiksi menggantung, dan ada bukti jabatan/instansi/kronologi yang bisa
  dirujuk di laporan.
- critique_summary: ringkas 1-3 kalimat (Bahasa Indonesia): apa yang sudah
  kuat, apa yang bolong, kontradiksi apa yang terlihat.
- suggested_sub_queries: 0-3 kueri pencarian spesifik (Bahasa Indonesia) yang
  memuat nama target + pembeda dari konteks (instansi/kota/jabatan), maksimum
  ~12 kata per kueri, tanpa tanda kurung. Kosongkan bila cukup.
- Skema: {"is_sufficient": <bool>, "critique_summary": "<str>",
  "suggested_sub_queries": ["<str>", "..."]}"""


def build_critic_system_prompt(language: str = "id") -> str:
    """System prompt fact-verifier; bahasa ringkasan mengikuti ``language``."""
    if (language or "id").lower().startswith("id"):
        return _SYSTEM_PROMPT
    return (
        _SYSTEM_PROMPT
        + "\n\nWrite critique_summary and suggested_sub_queries in English."
    )


def build_critic_prompt(
    target_entity: str,
    context: str,
    segments: Sequence[Segment],
) -> str:
    """Blok bukti + instruksi verdict untuk LLM critic."""
    lines = [f"TARGET INVESTIGASI: {target_entity}"]
    if context.strip():
        lines.append(f"KONTEKS TAMBAHAN: {context.strip()}")
    lines.append("")
    lines.append("=== BUKTI YANG TERKUMPUL ===")
    for index, segment in enumerate(segments, start=1):
        lines.append(f"[S{index}] URL: {segment.url}")
        if segment.title:
            lines.append(f"      Judul: {segment.title}")
        lines.append(f"      Teks: {segment.text}")
        lines.append("")
    lines.append("=== AKHIR BUKTI ===")
    lines.append(
        "Nilai kecukupan bukti di atas terhadap target dan jawab HANYA JSON "
        "sesuai skema."
    )
    return "\n".join(lines)


def _clean_sub_queries(value: object, *, limit: int) -> list[str]:
    """Normalisasi usulan sub-kueri: rapikan spasi, buang kosong/duplikat, cap."""
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        query = " ".join(str(item or "").split()).strip()
        if not query:
            continue
        if len(query) > 160:
            query = query[:160].rstrip()
        if any(query.lower() == existing.lower() for existing in out):
            continue
        out.append(query)
        if len(out) >= limit:
            break
    return out


def _parse_verdict(text: str, *, limit: int) -> CriticResult | None:
    """Parse jawaban LLM menjadi verdict; ``None`` bila bukan JSON valid."""
    data = extract_json_block(text)
    if not isinstance(data, dict) or "is_sufficient" not in data:
        return None
    return CriticResult(
        is_sufficient=bool(data.get("is_sufficient")),
        critique_summary=" ".join(str(data.get("critique_summary") or "").split()),
        suggested_sub_queries=_clean_sub_queries(
            data.get("suggested_sub_queries"), limit=limit
        ),
    )


def _accumulate_meta(result: CriticResult, meta: dict) -> None:
    """Tambahkan penggunaan token dari satu percobaan ke total."""
    if not meta:
        return
    model = str(meta.get("model") or "")
    if model:
        result.model = model
    tokens = int(meta.get("total_tokens") or 0)
    if not tokens:
        tokens = result.input_tokens_est + int(meta.get("output_tokens") or 0)
    result.total_tokens += tokens


async def analyze_fact_gaps(
    documents_parquet_path: str | Path,
    target_entity: str,
    context: str = "",
    *,
    config: LLMConfig | None = None,
    client: LLMClient | None = None,
    env: Mapping[str, str] | None = None,
    language: str = "id",
) -> CriticResult:
    """Phase 2 entry point: nilai kecukupan bukti dan usulkan sub-kueri gap.

    Membaca dokumen Phase 1 dari parquet, memilih segmen paling relevan, lalu
    meminta verdict JSON ke LLM (Gemini/OpenRouter/Groq). Dua percobaan: dengan
    *native JSON mode* lalu tanpa (fallback untuk provider yang menolaknya).
    Tidak pernah melempar exception — kegagalan menjadi ``error`` /
    ``skipped_reason``.
    """
    cfg = config or LLMConfig()
    result = CriticResult()

    documents = load_documents(documents_parquet_path)
    if not documents:
        result.skipped_reason = f"tidak ada dokumen di {documents_parquet_path}"
        return result

    system = build_critic_system_prompt(language)

    specs: list[ProviderSpec] = []
    if client is None:
        specs = resolve_providers(cfg, env)
        if not specs:
            result.skipped_reason = (
                "API key LLM tidak ditemukan — set GEMINI_API_KEY (Google AI "
                "Studio), OPENROUTER_API_KEY, atau GROQ_API_KEY di .env"
            )
            return result
        result.provider, result.model = specs[0].name, specs[0].model
    else:
        result.provider = getattr(client, "name", "injected")
        result.model = getattr(client, "model", "")

    cfg_call = cfg.model_copy(
        update={
            # Groq OTPM free tier 1000 -> cap output critic (legacy path) 350-400.
            "max_output_tokens": max(350, min(400, cfg.critic_max_output_tokens)),
        }
    )
    retry_suffix = (
        "\n\nPENTING: balas HANYA JSON valid satu objek dengan key "
        '"is_sufficient", "critique_summary", "suggested_sub_queries" — '
        "tanpa code fence dan tanpa teks lain."
    )
    plans: list[tuple[list[Segment], str, int]] = []
    for max_segments, max_tokens, max_chars in prompt_size_ladder(
        cfg.critic_max_segments,
        cfg.critic_max_input_tokens,
        cfg.max_chars_per_segment,
        token_caps=(4000, 2500),
    ):
        tier_segments = select_segments(
            documents,
            target_entity,
            context,
            max_segments=max_segments,
            max_input_tokens=max_tokens,
            max_chars_per_segment=max_chars,
        )
        if not tier_segments:
            continue
        tier_user = build_critic_prompt(target_entity, context, tier_segments)
        # HARD GUARD (legacy path): system + user + retry-suffix ≤
        # critic_max_input_tokens (default 3.4k, anti 413/503).
        user_budget = max(
            200, cfg.critic_max_input_tokens - count_tokens(system) - 64
        )
        tier_user = truncate_tokens(tier_user, user_budget)
        est = count_tokens(system) + count_tokens(tier_user)
        if plans and est >= plans[-1][2]:
            continue  # tier tidak benar-benar lebih kecil (artefak chunking/dedupe)
        plans.append((tier_segments, tier_user, est))
    if not plans:
        result.skipped_reason = "tidak ada segmen relevan untuk dinilai"
        return result

    last_text = ""
    last_exc: Exception | None = None
    for tier, (segments, user, est) in enumerate(plans):
        result.documents_used = len({segment.url for segment in segments})
        result.segments_used = len(segments)
        result.input_tokens_est = est
        if tier > 0:
            log.warning(
                "critic: prompt dipangkas ke ~%d token (batas TPM provider) -> retry",
                est,
            )
        attempts: list[tuple[bool, str]] = [(True, user), (False, user + retry_suffix)]
        size_blocked = False
        for index, (json_mode, prompt) in enumerate(attempts):
            try:
                text, meta, spec = await generate_with_failover(
                    specs, cfg_call, system, prompt, json_mode=json_mode, client=client
                )
            except Exception as exc:
                last_exc = exc
                if is_prompt_too_large(exc):
                    size_blocked = True
                    break
                if index == 0:
                    log.warning(
                        "critic percobaan JSON gagal di semua provider (%s) -> "
                        "retry tanpa json_mode",
                        exc,
                    )
                    continue
                result.error = f"{type(exc).__name__}: {exc}"
                log.warning("critic request gagal: %s", exc)
                return result
            result.provider = spec.name
            last_text = text or ""
            _accumulate_meta(result, meta)
            verdict = _parse_verdict(last_text, limit=cfg.critic_max_sub_queries)
            if verdict is not None:
                result.is_sufficient = verdict.is_sufficient
                result.critique_summary = verdict.critique_summary
                result.suggested_sub_queries = verdict.suggested_sub_queries
                result.retries = tier + index
                if not result.is_sufficient and not result.suggested_sub_queries:
                    log.info("critic: gap terdeteksi tanpa usulan sub-kueri")
                return result
            if index == 0:
                log.warning("jawaban critic bukan JSON valid -> retry tanpa json_mode")
        if not size_blocked:
            result.error = "jawaban critic tidak bisa diparse sebagai JSON"
            log.warning("critic: %s (tail=%r)", result.error, last_text[-120:])
            return result
        if tier + 1 < len(plans):
            log.warning(
                "critic: prompt ~%d token ditolak provider (batas TPM) -> "
                "pangkas konteks & retry",
                est,
            )
            continue
        assert last_exc is not None
        result.error = f"{type(last_exc).__name__}: {last_exc}"
        log.warning("critic request gagal (batas token provider): %s", last_exc)
        return result

    result.skipped_reason = "tidak ada segmen relevan untuk dinilai"
    return result


# ---- Phase 3: triple-JSON critic (NO raw text) -------------------------------

_TRIPLE_SYSTEM_PROMPT = """Anda adalah OSINT Graph Critic. Anda HANYA menerima
daftar triple JSON (bukan teks mentah) dari knowledge graph investigasi.
Tugas: menilai apakah triple sudah cukup untuk laporan final, temukan
kontradiksi, dan sarankan ID node untuk pivot berikutnya.

Periksa:
1. KELENGKAPAN — apakah identitas target (nama, jabatan, instansi, lokasi)
   terwakili? apakah relasi penting masih bolong?
2. KONTRADIKSI — triple yang saling bertentangan (jabatan/tanggal/lokasi).
3. PIVOT — node mana yang layak diperdalam? Pilih dari node_id yang ADA di
   daftar triple, prioritaskan individu/instansi yang belum banyak dieksplorasi.

Jawab HANYA JSON valid (tanpa code fence/teks lain):
{"is_sufficient": <bool>,
 "critique_summary": "<1-3 kalimat Bahasa Indonesia>",
 "contradictions": ["<deskripsi kontradiksi>", "..."],
 "suggested_sub_queries": ["<kueri pencarian spesifik>", "..."],
 "suggested_nodes": [{"node_id": "n12", "label": "<label>",
                      "reason": "<alasan pivot 1 kalimat>"}]}

Aturan: suggested_nodes maksimum 5 ("Saran ID Node untuk Pivot"), node_id HARUS
persis salah satu subject_id/object_id pada triple; suggested_sub_queries
maksimum 3, memuat nama target + pembeda konteks; kosongkan list bila tidak ada."""


def build_triple_critic_system_prompt(language: str = "id") -> str:
    if (language or "id").lower().startswith("id"):
        return _TRIPLE_SYSTEM_PROMPT
    return _TRIPLE_SYSTEM_PROMPT + "\n\nWrite all strings in English."


def build_triple_critic_prompt(
    target_entity: str,
    context: str,
    triples: Sequence[dict],
) -> str:
    """Compact JSON triple block + verdict instruction (token-bounded)."""
    lines = [f"TARGET INVESTIGASI: {target_entity}"]
    if context.strip():
        lines.append(f"KONTEKS TAMBAHAN: {context.strip()}")
    lines.append("")
    lines.append("=== TRIPLE GRAPH (JSON) ===")
    for item in triples:
        lines.append(
            json.dumps(
                {
                    "subject_id": item.get("subject_id", ""),
                    "subject": item.get("subject", ""),
                    "predicate": item.get("predicate", ""),
                    "object_id": item.get("object_id", ""),
                    "object": item.get("object", ""),
                    "source_count": item.get("source_count", 0),
                    "verified": item.get("verified", False),
                    "contradiction": item.get("contradiction", False),
                },
                ensure_ascii=False,
            )
        )
    lines.append("=== AKHIR TRIPLE ===")
    lines.append("Nilai kecukupan bukti dan jawab HANYA JSON sesuai skema.")
    return "\n".join(lines)


def _clean_str_list(value: object, *, limit: int, max_len: int = 240) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = " ".join(str(item or "").split()).strip()
        if not text or len(text) > max_len:
            continue
        if any(text.lower() == existing.lower() for existing in out):
            continue
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _clean_nodes(
    value: object,
    *,
    labels: dict[str, str],
    limit: int = 5,
) -> list[SuggestedNode]:
    """Keep only suggested nodes whose id really exists in the input triples."""
    if not isinstance(value, list):
        return []
    out: list[SuggestedNode] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        node_id = str(item.get("node_id") or "").strip()
        if node_id not in labels or node_id in seen:
            continue
        seen.add(node_id)
        out.append(
            SuggestedNode(
                node_id=node_id,
                label=labels.get(node_id) or str(item.get("label") or ""),
                reason=" ".join(str(item.get("reason") or "").split())[:200],
            )
        )
        if len(out) >= limit:
            break
    return out


def _parse_triple_verdict(
    text: str,
    *,
    labels: dict[str, str],
    limit: int,
) -> CriticResult | None:
    data = extract_json_block(text)
    if not isinstance(data, dict) or "is_sufficient" not in data:
        return None
    return CriticResult(
        is_sufficient=bool(data.get("is_sufficient")),
        critique_summary=" ".join(str(data.get("critique_summary") or "").split()),
        suggested_sub_queries=_clean_sub_queries(
            data.get("suggested_sub_queries"), limit=limit
        ),
        contradictions=_clean_str_list(data.get("contradictions"), limit=5),
        suggested_nodes=_clean_nodes(data.get("suggested_nodes"), labels=labels),
    )


def _group_triples(
    triples: Sequence[dict],
    max_tokens: int,
) -> list[list[dict]]:
    """Token-bounded triple batches (Map step of the critic)."""
    batches: list[list[dict]] = []
    current: list[dict] = []
    used = 0
    for item in triples:
        cost = max(8, count_tokens(json.dumps(item, ensure_ascii=False))) + 6
        if current and used + cost > max_tokens:
            batches.append(current)
            current = []
            used = 0
        current.append(item)
        used += cost
        if used >= max_tokens:  # satu triple raksasa tetap jalan sendiri
            batches.append(current)
            current = []
            used = 0
    if current:
        batches.append(current)
    return batches


async def analyze_graph_gaps(
    triples: Sequence[dict],
    target_entity: str,
    context: str = "",
    *,
    config: LLMConfig | None = None,
    client: LLMClient | None = None,
    env: Mapping[str, str] | None = None,
    language: str = "id",
) -> CriticResult:
    """Phase 3 entry point: critic membaca JSON triple, bukan teks mentah.

    Map-Reduce: triple dipecah ke batch ≤ ``batch_max_input_tokens``; tiap batch
    dinilai LLM (map), lalu verdict digabung (reduce). ``is_sufficient`` hanya
    true bila SEMUA batch dinilai cukup. ``suggested_nodes`` | ``node_id`` yang
    tidak ada di triple otomatis dibuang (grounding).
    """
    cfg = config or LLMConfig()
    result = CriticResult()
    triples = list(triples)
    if not cfg.enabled:
        result.skipped_reason = "critic nonaktif (llm.enabled=false)"
        return result
    if not triples:
        result.skipped_reason = "tidak ada triple di graph untuk dinilai"
        return result

    labels: dict[str, str] = {}
    for item in triples:
        for id_key, label_key in (("subject_id", "subject"), ("object_id", "object")):
            node_id = str(item.get(id_key) or "").strip()
            label = str(item.get(label_key) or "").strip()
            if node_id and node_id not in labels:
                labels[node_id] = label
    result.nodes_used = len(labels)
    result.documents_used = len({sid for item in triples for sid in item.get("sources") or []})

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

    cfg_call = cfg.model_copy(
        update={
            # Groq OTPM free tier 1000 -> cap output critic 350-400 token.
            "max_output_tokens": max(350, min(400, cfg.critic_max_output_tokens)),
            "temperature": 0.2,
        }
    )
    system = build_triple_critic_system_prompt(language)
    batches = _group_triples(triples, cfg.batch_max_input_tokens)
    result.batches = len(batches)

    sufficient_all = True
    summaries: list[str] = []
    queries: list[str] = []
    contradictions: list[str] = []
    suggested: list[SuggestedNode] = []

    for batch in batches:
        user = build_triple_critic_prompt(target_entity, context, batch)
        # HARD GUARD: system + user + retry-suffix ≤ batch_max_input_tokens.
        user_budget = max(64, cfg.batch_max_input_tokens - count_tokens(system) - 64)
        user = truncate_tokens(user, user_budget)
        result.input_tokens_est += count_tokens(system) + count_tokens(user)
        text = ""
        meta: dict = {}
        verdict: CriticResult | None = None
        for attempt in (0, 1):
            prompt = user
            if attempt == 1:
                prompt += (
                    "\n\nPENTING: balas HANYA JSON valid satu objek dengan key "
                    '"is_sufficient", "critique_summary", "suggested_nodes".'
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
            except Exception as exc:
                if attempt == 0 and _json_mode_error(exc):
                    log.warning(
                        "critic(triple): json_mode ditolak provider -> retry tanpa itu"
                    )
                    continue
                result.error = f"{type(exc).__name__}: {exc}"
                log.warning("critic(triple) request gagal: %s", exc)
                return result
            result.provider = spec.name
            result.retries += attempt
            verdict = _parse_triple_verdict(
                text, labels=labels, limit=cfg.critic_max_sub_queries
            )
            if verdict is not None:
                break
            log.warning("jawaban critic(triple) bukan JSON valid -> retry tanpa json_mode")
        _accumulate_meta(result, meta)
        if verdict is None:
            result.error = "jawaban critic(triple) tidak bisa diparse sebagai JSON"
            return result
        sufficient_all = sufficient_all and verdict.is_sufficient
        if verdict.critique_summary:
            summaries.append(verdict.critique_summary)
        for query in verdict.suggested_sub_queries:
            if query not in queries:
                queries.append(query)
        for item in verdict.contradictions:
            if item not in contradictions:
                contradictions.append(item)
        for node in verdict.suggested_nodes:
            if all(existing.node_id != node.node_id for existing in suggested):
                suggested.append(node)

    result.is_sufficient = sufficient_all
    result.critique_summary = " ".join(summaries)[:600]
    result.suggested_sub_queries = queries[: cfg.critic_max_sub_queries]
    result.contradictions = contradictions[:5]
    result.suggested_nodes = suggested[:5]
    if not result.is_sufficient and not result.suggested_sub_queries:
        log.info("critic(triple): gap terdeteksi tanpa usulan sub-kueri")
    log.info(
        "critic(triple): sufficient=%s, %d batch, %d node, %d triple, %d token",
        result.is_sufficient,
        result.batches,
        result.nodes_used,
        len(triples),
        result.total_tokens,
    )
    return result


def _json_mode_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("response_format", "json mode", "json_object", "response format")
    )
