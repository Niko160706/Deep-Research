"""P0 — seeded query planner (ONE small LLM call, ~400 tokens, best-effort).

Before any search happens, the planner asks the LLM for 3-4 *seeded sub-query
variations* of the target query: identity, role/position, affiliation and
official-source probes. Those tasks are pushed into the frontier ahead of the
mechanical decomposition so the first search cycles already cover angles the
template decomposer would miss.

Graceful degradation rules (project-wide):

* prompt is hard-capped (``PLANNER_MAX_INPUT_TOKENS`` ≈ 400) so the call can
  never trip Groq's 413 / Gemini 503 size limits;
* no key / provider error / invalid JSON -> deterministic seed fallback;
* never raises.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence

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

log = logging.getLogger("drx.brain.query_planner")

#: Hard cap for the planner prompt (small call by design).
PLANNER_MAX_INPUT_TOKENS = 400


class QueryPlanResult(BaseModel):
    """Seeded sub-query variations + execution metadata."""

    queries: list[str] = Field(default_factory=list)
    rationale: str = ""
    provider: str = ""
    model: str = ""
    input_tokens_est: int = 0
    total_tokens: int = 0
    retries: int = 0
    skipped_reason: str = ""
    error: str = ""
    used_fallback: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.queries) and not self.error


_SYSTEM = """Anda adalah OSINT Query Planner. Dari satu target investigasi,
hasilkan variasi sub-kueri pencarian yang berbenih (seeded) untuk mencari
identitas, jabatan/profesi, afiliasi instansi, dan sumber resmi.

Jawab HANYA JSON valid: {"queries": ["...", "..."], "rationale": "<1 kalimat>"}
Aturan: 3-4 kueri, maksimum 12 kata per kueri, Bahasa Indonesia, tanpa tanda
kutip ganda di dalam string, sertakan pembeda konteks (kota/instansi) bila ada.
Jangan membuat kueri tentang orang/entitas lain yang tidak diminta."""


def seeded_queries(target_entity: str, context: str = "", limit: int = 4) -> list[str]:
    """Deterministic fallback seeds (also shown to the LLM as examples)."""
    target = " ".join((target_entity or "").split())
    if not target:
        return []
    terms = [" ".join(t.strip().split()) for t in (context or "").split(",") if t.strip()]
    ctx = terms[0] if terms else ""
    seeds = [
        target,
        f"{target} profil lengkap",
        f"{target} jabatan karier",
    ]
    if ctx:
        seeds.append(f"{target} {ctx}")
    out: list[str] = []
    for seed in seeds:
        if seed and seed.lower() not in {x.lower() for x in out}:
            out.append(seed)
        if len(out) >= limit:
            break
    return out


def build_planner_prompt(target_entity: str, context: str, seeds: Sequence[str]) -> str:
    lines = [f"TARGET: {target_entity}"]
    if context.strip():
        lines.append(f"KONTEKS: {context.strip()}")
    if seeds:
        lines.append("BENIH AWAL: " + " | ".join(seeds))
    lines.append("Hasilkan 3-4 variasi sub-kueri berbenih. Jawab JSON.")
    return "\n".join(lines)


def _clean_queries(value: object, *, target: str, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    target_tokens = {t.lower() for t in target.split() if len(t) > 2}
    out: list[str] = []
    for item in value:
        query = " ".join(str(item or "").split()).strip().strip('"').strip()
        if not query or len(query) > 160:
            continue
        tokens = {t.lower().strip(".,") for t in query.split()}
        if target_tokens and not (tokens & target_tokens):
            continue  # tanpa nama target -> bukan kueri untuk investigasi ini
        if any(query.lower() == existing.lower() for existing in out):
            continue
        out.append(query)
        if len(out) >= limit:
            break
    return out


async def plan_queries(
    target_entity: str,
    context: str = "",
    *,
    config: LLMConfig | None = None,
    client: LLMClient | None = None,
    env: Mapping[str, str] | None = None,
    language: str = "id",
) -> QueryPlanResult:
    """One small LLM call for seeded variations; deterministic fallback on error."""
    cfg = config or LLMConfig()
    limit = max(1, int(cfg.planner_max_queries))
    result = QueryPlanResult()
    seeds = seeded_queries(target_entity, context, limit=limit)
    if not cfg.enabled or not cfg.planner_enabled:
        result.skipped_reason = "query planner nonaktif"
        return result

    specs: list[ProviderSpec] = []
    if client is None:
        specs = resolve_providers(cfg, env)
        if not specs:
            result.skipped_reason = "API key LLM tidak ditemukan"
            result.queries = seeds
            result.used_fallback = True
            return result
        result.provider, result.model = specs[0].name, specs[0].model
    else:
        result.provider = getattr(client, "name", "injected")
        result.model = getattr(client, "model", "")

    system = _SYSTEM
    user = build_planner_prompt(target_entity, context, seeds)
    budget = max(120, PLANNER_MAX_INPUT_TOKENS - count_tokens(system) - 24)
    user = truncate_tokens(user, budget)
    result.input_tokens_est = count_tokens(system) + count_tokens(user)

    cfg_call = cfg.model_copy(
        update={
            "max_output_tokens": 300,
            "temperature": 0.4,
            "thinking_level": "minimal",
        }
    )
    text = ""
    meta: dict = {}
    for attempt in (0, 1):
        prompt = user
        if attempt == 1:
            prompt += (
                '\n\nPENTING: balas HANYA JSON valid {"queries": [...]}, '
                "tanpa code fence dan tanpa teks lain."
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
            result.retries = attempt
            break
        except Exception as exc:
            if attempt == 0 and _json_mode_error(exc):
                log.warning("planner: json_mode ditolak provider -> retry tanpa itu")
                continue
            result.error = f"{type(exc).__name__}: {exc}"
            log.warning("query planner gagal: %s", exc)
            result.queries = seeds
            result.used_fallback = True
            return result

    output = int(meta.get("output_tokens") or 0)
    result.model = str(meta.get("model") or result.model)
    result.total_tokens = int(meta.get("total_tokens") or 0) or (
        result.input_tokens_est + output
    )
    data = extract_json_block(text)
    queries = _clean_queries((data or {}).get("queries"), target=target_entity, limit=limit)
    if not queries:
        queries = seeds
        result.used_fallback = True
    result.queries = queries
    if data:
        result.rationale = " ".join(str(data.get("rationale") or "").split())[:200]
    log.info(
        "query planner: %d variasi (%d token, fallback=%s)",
        len(result.queries),
        result.total_tokens,
        result.used_fallback,
    )
    return result


def _json_mode_error(exc: BaseException | str) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("response_format", "json mode", "json_object", "response format")
    )
