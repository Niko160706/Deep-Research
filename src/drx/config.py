"""Configuration loading with Pydantic v2 validation.

Reads `config/settings.yaml` into a typed :class:`Settings` object. Every
hard limit, threshold and path used by the orchestrator lives here so it can
be tuned without touching code.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from pydantic import BaseModel, Field

log = logging.getLogger("drx.config")


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load ``KEY=VALUE`` pairs from ``.env`` into ``os.environ``.

    Real environment variables always win over ``.env`` values, and lines
    starting with ``#`` or without ``=`` are ignored. Used to inject secrets
    (e.g. ``BRAVE_SEARCH_API_KEY``) without hard-coding them in YAML or code.
    """
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class RoleConfig(BaseModel):
    """Per-role LLM settings (see `config/models.yaml`)."""

    model: str = "deepseek-chat"
    temperature: float = 0.2


class BudgetLimits(BaseModel):
    max_tokens: int = 150000
    max_depth: int = 4
    max_runtime_seconds: int = 600
    max_urls: int = 200
    playwright_max_tabs: int = 2


class NoveltyConfig(BaseModel):
    floor: float = 0.10
    stagnation_limit: int = 3


class SearchConfig(BaseModel):
    backend: str = "hybrid"
    region: str = "global"
    results_per_query: int = 10
    timeout_seconds: float = 15.0
    # Lazy decomposition: run the next sub-query only while the run has fewer
    # than this many valid documents (0 = always run every sub-query).
    lazy_min_docs: int = 10


class CleaningConfig(BaseModel):
    min_text_length: int = 400
    bm25_keep_top_k: int = 12
    target_lang: str = ""
    # Strict disambiguation: with --context, ordinary documents must contain at
    # least one context keyword (profiles are exempt).
    require_context: bool = True
    # ASN/employee tables: NIP/PNS/Golongan hits above this -> keep only the
    # line window around the target name.
    asn_noise_threshold: int = 10


class ScrapingConfig(BaseModel):
    min_text_chars: int = 500
    # High-value domains (ac.id/go.id/wikipedia/imdb/letterboxd/sinta) escalate
    # to Playwright when the static text is under this many chars.
    high_value_min_chars: int = 200


class ChunkingConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 200


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-small-en-v1.5"
    quantize_int8: bool = True


class SynthesisConfig(BaseModel):
    max_context_tokens: int = 30000
    language: str = "id"


class LLMConfig(BaseModel):
    """Phase 2 — grounded extraction, critic, and report synthesis.

    Hierarki default (``provider="auto"``): **Groq GPT-OSS 120B → Groq Qwen
    → Gemini**. Gemini diletakkan terakhir karena kuota hariannya kecil
    (20 request/hari); OpenRouter hanya dipakai bila dipilih eksplisit.
    """

    enabled: bool = True
    # auto: Groq (gpt-oss-120b -> qwen3.8-27b) -> Gemini.
    provider: str = "auto"  # auto | groq | gemini | openrouter
    # Gemini: cadangan terakhir (model + rantai fallback, dipisah koma).
    model: str = "gemini-3.8-flash"
    fallback_model: str = "gemini-3.7-flash,gemini-3.5-flash"
    openrouter_model: str = "google/gemini-3.8-flash"
    # Groq: PRIMARY gpt-oss-120b, lalu SECONDARY qwen3.8-27b.
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str = "qwen/qwen3.8-27b"
    # Jeda minimum antar request provider (detik) — Groq free tier OTPM 1000.
    provider_min_interval_seconds: float = 2.0
    temperature: float = 0.25
    # Gemini 3.x: level berpikir (minimal|low|medium|high; "" = default provider).
    # Token "thinking" ikut memakan max_output_tokens -> "low" menjaga laporan utuh.
    thinking_level: str = "low"
    timeout_seconds: float = 120.0
    # Konteks prompt: 15–20 segmen ATAU sampai ~30k token (mana yang lebih dulu).
    max_segments: int = 18
    max_input_tokens: int = 30000
    max_chars_per_segment: int = 2000
    max_output_tokens: int = 8192
    # ---- Phase 3: token-accurate batching guard (anti 413/503) ----
    # Setiap request LLM (ekstraksi/planner/critic/sintesis) dibatasi 3000-3500
    # token input; dokumen lebih besar dipotong via MMR / Map-Reduce.
    batch_max_input_tokens: int = 3200
    use_mmr: bool = True  # diversitas bukti (Maximal Marginal Relevance)
    mmr_lambda: float = 0.7  # 1.0 = relevance-only, 0.0 = diversity-only
    extraction_enabled: bool = True  # triple LLM + grounding (source_id/short_quote)
    extraction_max_output_tokens: int = 400  # 350-400 (Groq OTPM 1000 / anti truncate)
    planner_enabled: bool = True  # P0 query planner (~400 token, 1 panggilan awal)
    planner_max_queries: int = 4
    # Critic hanya membaca JSON triple (bukan teks mentah parquet).
    critic_reads_triples: bool = True
    critic_max_triples: int = 120  # batas triple JSON per panggilan critic
    critic_max_output_tokens: int = 400  # 350-400 (Groq OTPM 1000)
    # Harga estimasi biaya checkpoint (USD per 1 juta token; 0 = free tier).
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0
    # ---- Agentic critic loop (gap analysis sebelum sintesis) ----
    critic_enabled: bool = True
    critic_max_loops: int = 2  # maksimum ronde eksekusi sub-kueri gap
    critic_max_sub_queries: int = 3  # sub-kueri gap yang dieksekusi per ronde
    critic_max_segments: int = 8  # aman untuk Groq free tier (ITPM 7k)
    critic_max_input_tokens: int = 3400  # hard guard 3.0–3.5k token (anti 413/503)


class VisualizerConfig(BaseModel):
    """Phase 3 — interactive graph canvas (``runs/<run_id>/graph_interactive.html``)."""

    enabled: bool = True
    max_nodes: int = 600  # batas node kanvas (bundle & RAM browser tetap wajar)
    cdn_resources: str = "remote"  # remote | in_line (bundle < 1.5 MB)


class WebConfig(BaseModel):
    """Mode Web App — server lokal kanvas + endpoint pivot/expand/stop."""

    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8000
    auto_open_browser: bool = True
    # Fallback port bila 8000 sedang dipakai (dicoba berurutan).
    port_attempts: int = 10


class PathsConfig(BaseModel):
    runs_dir: str = "runs"


class Settings(BaseModel):
    """Top-level configuration tree."""

    roles: dict[str, RoleConfig] = Field(default_factory=dict)
    budget_limits: BudgetLimits = Field(default_factory=BudgetLimits)
    novelty: NoveltyConfig = Field(default_factory=NoveltyConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    cleaning: CleaningConfig = Field(default_factory=CleaningConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    synthesis: SynthesisConfig = Field(default_factory=SynthesisConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    visualizer: VisualizerConfig = Field(default_factory=VisualizerConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    gemini_api_key: str | None = None
    openrouter_api_key: str | None = None
    groq_api_key: str | None = None
    brave_search_api_key: str | None = None
    tavily_api_key: str | None = None
    exa_api_key: str | None = None
    searxng_url: str = "http://127.0.0.1:8080"
    tor_proxy: str = "socks5://127.0.0.1:9050"

    @classmethod
    def load(cls, path: str | Path) -> Settings:
        """Load YAML config, injecting secrets from ``.env`` / env vars."""
        _load_dotenv()
        p = Path(path)
        data: dict = {}
        if p.exists():
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML is required to load config. Install requirements-base.txt"
                ) from exc
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        else:
            log.warning("Config not found at %s; using defaults", p)

        settings = cls.model_validate(data)

        # Secrets come from the environment (never committed to YAML).
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if gemini_key:
            settings.gemini_api_key = gemini_key
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if openrouter_key:
            settings.openrouter_api_key = openrouter_key
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            settings.groq_api_key = groq_key
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY")
        if brave_key:
            settings.brave_search_api_key = brave_key
        tavily_key = os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            settings.tavily_api_key = tavily_key
        exa_key = os.environ.get("EXA_API_KEY")
        if exa_key:
            settings.exa_api_key = exa_key
        searxng_url = os.environ.get("SEARXNG_URL")
        if searxng_url:
            settings.searxng_url = searxng_url
        tor_proxy = os.environ.get("TOR_PROXY")
        if tor_proxy:
            settings.tor_proxy = tor_proxy
        return settings

    def role(self, name: str) -> RoleConfig:
        return self.roles.get(name, RoleConfig())
