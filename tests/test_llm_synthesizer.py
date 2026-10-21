"""Tests for Phase 2 LLM report synthesis (``drx.brain.llm_synthesizer``).

Everything runs offline: provider resolution is pure (env mapping injected),
segment selection and prompt building are pure Python, and the LLM call itself
is exercised through an injected fake client — no network, no API key.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

from drx.brain import llm_synthesizer as syn
from drx.config import LLMConfig
from drx.models import Document

FULL_REPORT = """# Laporan Investigasi: Sri Lestari

## Profil & Identitas Utama
- Nama lengkap: Sri Lestari, A.Md.Kep. [S1]
- Jabatan/Profesi: perawat [S1]

## Ringkasan Eksekutif
Sri Lestari adalah perawat di Kab. Sleman berdasarkan sumber resmi. [S1]

## Rekam Jejak & Temuan Kunci
- Bekerja di lingkungan Dinas Kesehatan Kab. Sleman. [S1]

## Potensi Ambigu / Entitas Serupa
Tidak ada indikasi ambiguitas. [S1]

## Daftar Sumber / Sitasi
- [S1] https://news.contoh.id/sri-lestari
"""


class _FakeClient:
    """Deterministic stand-in for GeminiClient / OpenAICompatClient."""

    name = "fake"
    model = "fake-1"

    def __init__(self, replies: list[str], *, boom: bool = False) -> None:
        self.replies = list(replies)
        self.boom = boom
        self.calls: list[tuple[str, str]] = []

    async def generate(self, system: str, user: str) -> tuple[str, dict]:
        self.calls.append((system, user))
        if self.boom:
            raise RuntimeError("network down")
        text = self.replies.pop(0) if self.replies else ""
        return text, {
            "model": self.model,
            "input_tokens": 100,
            "output_tokens": 50,
            "total_tokens": 150,
        }


def _docs() -> list[Document]:
    return [
        Document(
            url="https://news.contoh.id/sri-lestari",
            title="Sri Lestari perawat Sleman",
            text=(
                "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan "
                "Kab. Sleman dengan dedikasi tinggi. "
            )
            * 6,
        ),
        Document(
            url="https://blog.contoh.id/resep",
            title="Resep masakan rumahan",
            text="Artikel ini membahas resep masakan rumahan yang mudah. " * 6,
        ),
    ]


def _write_parquet(tmp_path: Path, docs: list[Document], *, nested: bool = False) -> Path:
    if nested:
        target = tmp_path / "parquet" / "raw_documents.parquet"
    else:
        target = tmp_path / "raw_documents.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([d.model_dump() for d in docs]).to_parquet(target, index=False)
    return target


# ---- token estimation & provider resolution --------------------------------

def test_estimate_tokens_is_chars_over_four() -> None:
    assert syn.estimate_tokens("") == 0
    assert syn.estimate_tokens("abcd") == 1
    assert syn.estimate_tokens("a" * 400) == 100


def test_resolve_provider_auto_prefers_gemini() -> None:
    spec = syn.resolve_provider(
        LLMConfig(),
        {"GEMINI_API_KEY": "gk", "OPENROUTER_API_KEY": "ok", "GROQ_API_KEY": "qk"},
    )
    assert spec is not None
    assert spec.name == "gemini"
    assert spec.model == "gemini-3.8-flash"
    assert spec.fallback_model == "gemini-3.7-flash,gemini-3.5-flash"
    assert spec.api_key == "gk"
    assert spec.base_url == ""  # Google SDK default


def test_resolve_provider_accepts_google_api_key_alias() -> None:
    spec = syn.resolve_provider(LLMConfig(), {"GOOGLE_API_KEY": "alias"})
    assert spec is not None
    assert spec.name == "gemini"
    assert spec.api_key == "alias"


def test_resolve_provider_auto_falls_back_to_openrouter_then_groq() -> None:
    spec = syn.resolve_provider(LLMConfig(), {"OPENROUTER_API_KEY": "ok"})
    assert spec is not None
    assert spec.name == "openrouter"
    assert spec.model == "google/gemini-3.8-flash"
    assert spec.base_url == syn.OPENROUTER_BASE_URL

    spec = syn.resolve_provider(LLMConfig(), {"GROQ_API_KEY": "qk"})
    assert spec is not None
    assert spec.name == "groq"
    assert spec.model == "qwen/qwen3.8-27b"
    assert spec.base_url == syn.GROQ_BASE_URL
    # Groq multi-model fallback (OTPM guard): qwen -> gpt-oss-120b -> llama-3.3.
    assert syn.model_chain(spec) == [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
    ]


def test_resolve_provider_returns_none_without_any_key() -> None:
    assert syn.resolve_provider(LLMConfig(), {}) is None


def test_explicit_provider_requires_its_own_key() -> None:
    # openrouter key tersedia, tapi provider dipaksa "gemini" -> None (skip)
    cfg = LLMConfig(provider="gemini")
    assert syn.resolve_provider(cfg, {"OPENROUTER_API_KEY": "ok"}) is None
    spec = syn.resolve_provider(cfg, {"GEMINI_API_KEY": "gk"})
    assert spec is not None and spec.name == "gemini"


def test_unknown_provider_falls_back_to_auto() -> None:
    spec = syn.resolve_provider(LLMConfig(provider="bogus"), {"GROQ_API_KEY": "qk"})
    assert spec is not None and spec.name == "groq"


def test_build_client_selects_provider_class() -> None:
    cfg = LLMConfig()
    gem = syn.build_client(syn.ProviderSpec("gemini", "m", "k"), cfg)
    assert isinstance(gem, syn.GeminiClient)
    assert gem.name == "gemini"
    oai = syn.build_client(
        syn.ProviderSpec("openrouter", "m", "k", base_url=syn.OPENROUTER_BASE_URL), cfg
    )
    assert isinstance(oai, syn.OpenAICompatClient)
    assert oai.name == "openrouter"


# ---- provider failover chain (404/503 Gemini -> OpenRouter -> Groq) ---------

def test_resolve_providers_returns_priority_chain() -> None:
    specs = syn.resolve_providers(
        LLMConfig(),
        {"GEMINI_API_KEY": "gk", "OPENROUTER_API_KEY": "ok", "GROQ_API_KEY": "qk"},
    )
    assert [s.name for s in specs] == ["gemini", "openrouter", "groq"]

    specs = syn.resolve_providers(LLMConfig(), {"GROQ_API_KEY": "qk"})
    assert [s.name for s in specs] == ["groq"]

    assert syn.resolve_providers(LLMConfig(), {}) == []


def test_resolve_providers_explicit_provider_is_single_or_empty() -> None:
    cfg = LLMConfig(provider="groq")
    specs = syn.resolve_providers(cfg, {"GEMINI_API_KEY": "gk", "GROQ_API_KEY": "qk"})
    # provider eksplisit: tidak ada failover lintas-provider
    assert [s.name for s in specs] == ["groq"]
    assert syn.resolve_providers(cfg, {"GEMINI_API_KEY": "gk"}) == []


def test_model_chain_parses_comma_separated_fallbacks() -> None:
    spec = syn.ProviderSpec("gemini", "a", "k", fallback_model="b, c , a, ")
    assert syn.model_chain(spec) == ["a", "b", "c"]  # dedupe + trim
    assert syn.model_chain(syn.ProviderSpec("gemini", "", "k", fallback_model="")) == []


# ---- Groq multi-model fallback (qwen -> gpt-oss-120b -> llama-3.3) ----------


class _FakeCompletions:
    def __init__(self, owner: _FakeAsyncOpenAI) -> None:
        self.owner = owner

    async def create(self, **kwargs):
        model = kwargs["model"]
        self.owner.calls.append(model)
        if model != "llama-3.3-70b-versatile":
            raise RuntimeError(f"404 model {model} tidak tersedia untuk Groq")
        from types import SimpleNamespace

        usage = SimpleNamespace(prompt_tokens=3, completion_tokens=4, total_tokens=7)
        message = SimpleNamespace(content='{"queries": []}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class _FakeAsyncOpenAI:
    instances: ClassVar[list] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.chat = type("_Chat", (), {"completions": _FakeCompletions(self)})()
        _FakeAsyncOpenAI.instances.append(self)


@pytest.mark.asyncio
async def test_openai_compat_client_falls_back_through_groq_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import openai

    _FakeAsyncOpenAI.instances = []
    monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)

    spec = syn.ProviderSpec(
        "groq",
        "qwen/qwen3.8-27b",
        "qk",
        base_url=syn.GROQ_BASE_URL,
        fallback_model="openai/gpt-oss-120b,llama-3.3-70b-versatile",
    )
    client = syn.OpenAICompatClient(spec, LLMConfig())
    text, meta = await client.generate("sys", "user")

    assert text == '{"queries": []}'
    assert meta["model"] == "llama-3.3-70b-versatile"
    assert meta["used_fallback_model"] is True
    assert meta["total_tokens"] == 7
    assert _FakeAsyncOpenAI.instances[0].calls == [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "llama-3.3-70b-versatile",
    ]


class _FakeTime:
    def __init__(self, ticks: list[float]) -> None:
        self._ticks = iter(ticks)

    def monotonic(self) -> float:
        return next(self._ticks)


@pytest.mark.asyncio
async def test_throttle_provider_sleeps_between_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(syn, "asyncio", SimpleNamespace(sleep=fake_sleep))
    monkeypatch.setattr(syn, "time", _FakeTime([100.0, 100.0, 101.0, 101.0]))
    syn._provider_last_call.clear()

    await syn._throttle_provider("groq", 2.0)  # request pertama: tanpa jeda
    assert sleeps == []
    await syn._throttle_provider("groq", 2.0)  # 1s berlalu -> jeda 1s lagi
    assert sleeps == [1.0]

    sleeps.clear()
    await syn._throttle_provider("groq", 0)  # guard nonaktif -> tanpa sleep
    assert sleeps == []


@pytest.mark.asyncio
async def test_generate_with_failover_throttles_groq_real_clients_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    throttled: list[tuple[str, float]] = []

    async def fake_throttle(name: str, interval: float) -> None:
        throttled.append((name, interval))

    monkeypatch.setattr(syn, "_throttle_provider", fake_throttle)
    cfg = LLMConfig(provider_min_interval_seconds=2.0)
    spec = syn.ProviderSpec("groq", "m", "k", base_url=syn.GROQ_BASE_URL)

    # client injeksi (unit test) -> tidak pernah kena throttle
    injected = _FakeClient(["ok"])
    text, _meta, _spec = await syn.generate_with_failover(
        [spec], cfg, "sys", "user", client=injected
    )
    assert text == "ok"
    assert throttled == []

    # jalur client riil -> throttle dipanggil dengan interval settings
    monkeypatch.setattr(syn, "build_client", lambda spec, cfg: _FakeClient(["ok"]))
    text, _meta, _spec = await syn.generate_with_failover([spec], cfg, "sys", "user")
    assert text == "ok"
    assert throttled == [("groq", 2.0)]


def test_gemini_thinking_config_helper() -> None:
    from google.genai import types

    conf = syn.gemini_thinking_config(types, "low")
    assert conf is not None
    assert types.ThinkingLevel(conf.thinking_level) == types.ThinkingLevel.LOW

    assert syn.gemini_thinking_config(types, "") is None
    assert syn.gemini_thinking_config(types, "   ") is None
    assert syn.gemini_thinking_config(types, "turbo-9") is None  # invalid -> None


class _FailingProvider:
    """Provider yang selalu gagal (mis. 404 model retire / 503 high demand)."""

    def __init__(self, spec: syn.ProviderSpec) -> None:
        self.name = spec.name
        self.model = spec.model

    async def generate(self, system: str, user: str, *, json_mode: bool = False):
        raise RuntimeError("503 UNAVAILABLE: high demand")


@pytest.mark.asyncio
async def test_synthesize_report_fails_over_to_next_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini gagal total -> sintesis tetap jalan via provider berikutnya."""
    parquet = _write_parquet(tmp_path, _docs())
    working = _FakeClient([FULL_REPORT])

    def fake_build(spec: syn.ProviderSpec, cfg: LLMConfig):
        if spec.name == "gemini":
            return _FailingProvider(spec)
        return working

    monkeypatch.setattr(syn, "build_client", fake_build)

    result = await syn.synthesize_report(
        parquet,
        "Sri Lestari",
        "Sleman",
        env={"GEMINI_API_KEY": "gk", "GROQ_API_KEY": "qk"},  # tanpa OpenRouter key
    )

    assert result.ok and result.sections_complete
    assert result.provider == "groq"  # jatuh ke qwen di Groq
    assert working.calls  # provider fallback benar-benar dipanggil


# ---- parquet loading --------------------------------------------------------

def test_load_documents_accepts_run_dir_parquet_dir_and_file(tmp_path: Path) -> None:
    docs = _docs()
    file = _write_parquet(tmp_path, docs, nested=True)

    assert len(syn.load_documents(file)) == 2  # file parquet langsung
    assert len(syn.load_documents(file.parent)) == 2  # folder parquet/
    assert len(syn.load_documents(tmp_path)) == 2  # folder run
    assert syn.load_documents(tmp_path / "nope.parquet") == []


# ---- evidence selection -----------------------------------------------------

def test_select_segments_prioritizes_name_and_context() -> None:
    segments = syn.select_segments(
        _docs(),
        "Sri Lestari",
        "Sleman, Kesehatan",
        max_segments=10,
        max_input_tokens=10**9,
    )
    assert segments
    assert "perawat" in segments[0].text
    assert segments[0].url == "https://news.contoh.id/sri-lestari"
    scores = [s.score for s in segments]
    assert scores == sorted(scores, reverse=True)


def test_select_segments_respects_segment_and_token_caps() -> None:
    docs = [
        Document(
            url=f"https://x.id/{i}",
            title="profil",
            text=f"Sri Lestari perawat Sleman nomor {i}. " * 8,
        )
        for i in range(6)
    ]
    by_count = syn.select_segments(
        docs, "Sri Lestari", "Sleman", max_segments=3, max_input_tokens=10**9
    )
    assert len(by_count) == 3

    unlimited = syn.select_segments(
        docs, "Sri Lestari", "Sleman", max_segments=50, max_input_tokens=10**9
    )
    capped = syn.select_segments(
        docs, "Sri Lestari", "Sleman", max_segments=50, max_input_tokens=150
    )
    assert len(capped) < len(unlimited)  # token cap benar-benar memotong
    assert sum(syn.estimate_tokens(s.text) for s in capped) <= 150


def test_select_segments_dedupes_identical_text() -> None:
    body = "Sri Lestari perawat Sleman. " * 10
    docs = [
        Document(url="https://a.id/1", title="satu", text=body),
        Document(url="https://b.id/2", title="dua", text=body),
    ]
    segments = syn.select_segments(docs, "Sri Lestari", "Sleman")
    assert len(segments) == 1


# ---- prompts ----------------------------------------------------------------

def test_prompts_carry_sections_target_and_citation_urls() -> None:
    segments = syn.select_segments(_docs(), "Sri Lestari", "Sleman, Kesehatan")
    system = syn.build_system_prompt("id")
    user = syn.build_user_prompt("Sri Lestari", "Sleman, Kesehatan", segments)

    assert "OSINT Intelligence Analyst" in system
    for section in syn.REQUIRED_SECTIONS:
        assert section in system  # struktur wajib diinstruksikan
    assert "TARGET INVESTIGASI: Sri Lestari" in user
    assert "[S1]" in user
    assert "https://news.contoh.id/sri-lestari" in user


def test_retry_prompt_mentions_missing_sections() -> None:
    segments = syn.select_segments(_docs(), "Sri Lestari", "Sleman")
    user = syn.build_user_prompt(
        "Sri Lestari",
        "Sleman",
        segments,
        retry_missing=[syn.REQUIRED_SECTIONS[1]],
    )
    assert syn.REQUIRED_SECTIONS[1] in user
    assert "TULIS ULANG" in user or "Tulis ULANG" in user


# ---- end-to-end synthesis (fake client) ------------------------------------

@pytest.mark.asyncio
async def test_synthesize_report_with_injected_client(tmp_path: Path) -> None:
    _write_parquet(tmp_path, _docs(), nested=True)
    client = _FakeClient([FULL_REPORT])

    result = await syn.synthesize_report(
        tmp_path,  # folder run langsung (bukan file parquet)
        "Sri Lestari",
        "Sleman, Kesehatan",
        client=client,
        config=LLMConfig(),
    )

    assert result.ok
    assert result.sections_complete
    assert result.markdown.startswith("# Laporan Investigasi")
    assert result.segments_used >= 1
    assert result.documents_used >= 1
    assert result.total_tokens == 150
    system, user = client.calls[0]
    assert "Sri Lestari" in user
    assert "https://news.contoh.id/sri-lestari" in user
    assert syn.REQUIRED_SECTIONS[0] in system


@pytest.mark.asyncio
async def test_synthesize_report_skips_without_api_key(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    result = await syn.synthesize_report(parquet, "Sri Lestari", "Sleman", env={})
    assert not result.ok
    assert "API key" in result.skipped_reason
    assert result.markdown == ""


@pytest.mark.asyncio
async def test_synthesize_report_skips_without_documents(tmp_path: Path) -> None:
    result = await syn.synthesize_report(tmp_path / "missing.parquet", "Sri Lestari", "")
    assert not result.ok
    assert "tidak ada dokumen" in result.skipped_reason


@pytest.mark.asyncio
async def test_synthesize_report_retries_when_sections_missing(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    partial = "# Parsial\n\n## Profil & Identitas Utama\n- Sri Lestari [S1]\n"
    client = _FakeClient([partial, FULL_REPORT])

    result = await syn.synthesize_report(parquet, "Sri Lestari", "", client=client)

    assert len(client.calls) == 2  # ada satu retry
    assert result.retries == 1
    assert result.sections_complete
    assert result.markdown == FULL_REPORT.strip()
    assert "bagian wajib" in client.calls[1][1]  # instruksi retry dikirim


@pytest.mark.asyncio
async def test_synthesize_report_keeps_first_answer_when_retry_also_incomplete(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    first = "# A\n\n## Ringkasan Eksekutif\n- isi [S1]\n"
    second = "# B\n\n## Ringkasan Eksekutif\n- isi lain [S1]\n"
    client = _FakeClient([first, second])

    result = await syn.synthesize_report(parquet, "Sri Lestari", "", client=client)

    assert result.ok  # tidak crash, teks terbaik tetap dikembalikan
    assert not result.sections_complete
    assert result.markdown == first.strip()
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_synthesize_report_client_error_is_returned_not_raised(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeClient([], boom=True)

    result = await syn.synthesize_report(parquet, "Sri Lestari", "", client=client)

    assert not result.ok
    assert "RuntimeError" in result.error
    assert "network down" in result.error


# ---- output helpers ---------------------------------------------------------

def test_write_llm_report_writes_markdown_with_provenance(tmp_path: Path) -> None:
    result = syn.LLMResult(
        provider="gemini",
        model="gemini-2.5-flash",
        markdown=FULL_REPORT,
        documents_used=2,
        segments_used=3,
        input_tokens_est=1234,
        total_tokens=1500,
        sections_complete=True,
    )
    path = syn.write_llm_report(tmp_path, result)
    assert path == tmp_path / "report_llm.md"
    text = path.read_text(encoding="utf-8")
    assert "## Ringkasan Eksekutif" in text
    assert "gemini/gemini-2.5-flash" in text
    assert "3 segmen dari 2 dokumen" in text
    assert text.endswith("\n")

    assert syn.write_llm_report(tmp_path, syn.LLMResult()) is None


def test_format_llm_report_appends_footer() -> None:
    result = syn.LLMResult(
        provider="groq",
        model="llama-3.3-70b-versatile",
        markdown="# Laporan [S1]",
        segments_used=1,
        documents_used=1,
    )
    formatted = syn.format_llm_report(result)
    assert formatted.startswith("# Laporan [S1]")
    assert "groq/llama-3.3-70b-versatile" in formatted


# ---- report sidecar (structured output) --------------------------------------

SIDECAR_DICT = {
    "target_entity": "Sri Lestari",
    "verified_facts": [
        {
            "subject": "Sri Lestari",
            "predicate": "bekerja sebagai",
            "object": "perawat",
            "source_url": "https://news.contoh.id/sri-lestari",
        },
        {
            "subject": "Sri Lestari",
            "predicate": "berdomisili di",
            "object": "Jakarta",
            "source_url": "https://halusinasi.example/x",  # bukan URL bukti
        },
    ],
    "disambiguated_entities": ["Sri Lestari (Jakarta)", "  "],
    "sources_used": [
        "https://news.contoh.id/sri-lestari",
        "https://halusinasi.example/x",
    ],
}


def test_extract_json_block_variants() -> None:
    assert syn.extract_json_block('{"a": 1}') == {"a": 1}
    assert syn.extract_json_block('```json\n{"a": 1}\n```') == {"a": 1}
    assert syn.extract_json_block(
        'teks\n```\n{"a": {"b": 2}}\n```\nteks'
    ) == {"a": {"b": 2}}
    assert syn.extract_json_block("tidak ada json sama sekali") is None
    assert syn.extract_json_block("```json\n{bukan json}\n```") is None


def test_split_llm_answer_strips_sidecar_block() -> None:
    answer = (
        FULL_REPORT
        + "\n\n```json\n"
        + json.dumps(SIDECAR_DICT)
        + "\n```\n"
    )
    body, data = syn.split_llm_answer(answer)

    assert data is not None
    assert data["target_entity"] == "Sri Lestari"
    assert "```json" not in body
    assert body == FULL_REPORT.strip()  # laporan markdown tetap utuh


def test_split_llm_answer_without_block_returns_original() -> None:
    body, data = syn.split_llm_answer(FULL_REPORT)
    assert data is None
    assert body == FULL_REPORT.strip()


def test_build_sidecar_clamps_urls_to_evidence() -> None:
    segments = syn.select_segments(_docs(), "Sri Lestari", "Sleman, Kesehatan")

    sidecar, complete = syn.build_sidecar(SIDECAR_DICT, "Sri Lestari", segments)

    assert complete is True
    assert sidecar.target_entity == "Sri Lestari"
    assert len(sidecar.verified_facts) == 1  # fakta ber-URL halusinasi dibuang
    assert sidecar.verified_facts[0].source_url == "https://news.contoh.id/sri-lestari"
    assert sidecar.sources_used == ["https://news.contoh.id/sri-lestari"]
    assert sidecar.disambiguated_entities == ["Sri Lestari (Jakarta)"]


def test_build_sidecar_fallback_without_json() -> None:
    segments = syn.select_segments(_docs(), "Sri Lestari", "Sleman, Kesehatan")

    sidecar, complete = syn.build_sidecar(None, "Sri Lestari", segments)

    assert complete is False
    assert sidecar.target_entity == "Sri Lestari"
    assert sidecar.verified_facts == []
    # fallback diisi seluruh URL segmen bukti
    assert set(sidecar.sources_used) == {
        "https://news.contoh.id/sri-lestari",
        "https://blog.contoh.id/resep",
    }


@pytest.mark.asyncio
async def test_synthesize_report_extracts_sidecar_from_answer(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    answer = FULL_REPORT + "\n```json\n" + json.dumps(SIDECAR_DICT) + "\n```\n"
    client = _FakeClient([answer])

    result = await syn.synthesize_report(
        parquet, "Sri Lestari", "Sleman, Kesehatan", client=client
    )

    assert result.ok
    assert result.sections_complete  # blok JSON tidak merusak validasi bagian
    assert result.sidecar is not None and result.sidecar_complete
    assert result.sidecar.target_entity == "Sri Lestari"
    assert len(result.sidecar.verified_facts) == 1
    assert "```json" not in result.markdown
    assert result.markdown == FULL_REPORT.strip()


@pytest.mark.asyncio
async def test_synthesize_report_without_sidecar_block_uses_fallback(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeClient([FULL_REPORT])

    result = await syn.synthesize_report(
        parquet, "Sri Lestari", "Sleman", client=client
    )

    assert result.sidecar is not None
    assert result.sidecar_complete is False
    assert result.sidecar.verified_facts == []
    assert result.sidecar.sources_used  # diisi URL bukti sebagai fallback


def test_write_report_sidecar_preserves_phase1(tmp_path: Path) -> None:
    phase1 = {"documents": 3, "entities": ["Sri Lestari"]}
    (tmp_path / "report_sidecar.json").write_text(
        json.dumps(phase1), encoding="utf-8"
    )
    sidecar = syn.build_sidecar(
        SIDECAR_DICT,
        "Sri Lestari",
        syn.select_segments(_docs(), "Sri Lestari", "Sleman"),
    )[0]

    path = syn.write_report_sidecar(tmp_path, sidecar)

    assert path == tmp_path / "report_sidecar.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["target_entity"] == "Sri Lestari"
    assert data["verified_facts"][0]["predicate"] == "bekerja sebagai"
    assert data["phase1"] == phase1  # data Phase 1 tidak hilang


def test_write_report_sidecar_without_existing_file(tmp_path: Path) -> None:
    sidecar = syn.ReportSidecar(
        target_entity="Sri Lestari",
        verified_facts=[
            syn.VerifiedFact(
                subject="a", predicate="b", object="c", source_url="https://x.id"
            )
        ],
        sources_used=["https://x.id"],
    )
    data = json.loads(syn.write_report_sidecar(tmp_path, sidecar).read_text("utf-8"))
    assert "phase1" not in data
    assert data["verified_facts"] == [
        {"subject": "a", "predicate": "b", "object": "c", "source_url": "https://x.id"}
    ]


def test_format_llm_report_footer_mentions_verified_facts() -> None:
    result = syn.LLMResult(
        provider="gemini",
        model="gemini-2.5-flash",
        markdown="# Laporan [S1]",
        sidecar=syn.ReportSidecar(
            verified_facts=[
                syn.VerifiedFact(subject="a", predicate="b", object="c"),
                syn.VerifiedFact(subject="d", predicate="e", object="f"),
            ]
        ),
    )
    assert "2 fakta terverifikasi" in syn.format_llm_report(result)


# ---- prompt terlalu besar (Groq free tier ITPM 7k) ---------------------------

def test_is_prompt_too_large_markers() -> None:
    groq_413 = (
        "Error code: 413 - {'error': {'message': 'Request too large for model "
        "`qwen/qwen3.8-27b` ... ITPM: Limit 7000, Requested 8493 ...', "
        "'code': 'rate_limit_exceeded'}}"
    )
    assert syn.is_prompt_too_large(groq_413)
    assert syn.is_prompt_too_large("context_length_exceeded")
    assert not syn.is_prompt_too_large("503 UNAVAILABLE: high demand")
    assert not syn.is_prompt_too_large("network down")


def test_prompt_size_ladder_monotonic_and_deduped() -> None:
    ladder = syn.prompt_size_ladder(18, 30000, 2000)
    assert ladder[0] == (18, 30000, 2000)
    assert ladder[1] == (9, 5000, 1400)
    assert ladder[2] == (6, 3000, 1000)

    # konfigurasi sudah kecil -> tier tidak pernah membesar
    small = syn.prompt_size_ladder(2, 10**9, 2000)
    assert small[0] == (2, 10**9, 2000)
    assert all(tier[0] <= 2 for tier in small)
    assert small[-1] == (2, 3000, 1000)
    assert syn.prompt_size_ladder(4, 1500, 800)[-1] == (3, 1500, 800)


class _TooLargeOnceClient:
    """Fake provider: request pertama ditolak 413 (ITPM), lalu sukses."""

    name = "groq"
    model = "qwen/fake"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[str] = []

    async def generate(self, system: str, user: str) -> tuple[str, dict]:
        self.calls.append(user)
        if len(self.calls) == 1:
            raise RuntimeError(
                "Error code: 413 - Request too large for model "
                "`qwen/fake` ... ITPM: Limit 7000, Requested 8493 ... "
                "code: rate_limit_exceeded"
            )
        return self.reply, {
            "model": self.model,
            "input_tokens": 500,
            "output_tokens": 100,
            "total_tokens": 600,
        }


def _long_docs() -> list[Document]:
    """Dokumen panjang & bervariasi agar tier 0 benar-benar lebih besar dari tier 1."""
    return [
        Document(
            url=f"https://news.contoh.id/sri-lestari-{i}",
            title=f"Sri Lestari perawat Sleman {i}",
            text=(
                "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan Kab. Sleman. "
                + " ".join(
                    f"Catatan {i}-{j}: program imunisasi dan posyandu di "
                    f"wilayah Sleman utara berjalan baik pada tahun {2000 + j}."
                    for j in range(45)
                )
            ),
        )
        for i in range(6)
    ]


@pytest.mark.asyncio
async def test_synthesize_report_shrinks_prompt_when_provider_rejects_size(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _long_docs())
    client = _TooLargeOnceClient(FULL_REPORT)

    result = await syn.synthesize_report(
        parquet, "Sri Lestari", "Sleman", client=client
    )

    assert result.ok and result.sections_complete
    assert len(client.calls) == 2  # 413 -> pangkas konteks -> retry
    assert len(client.calls[1]) < len(client.calls[0])
    assert result.retries == 1


@pytest.mark.asyncio
async def test_synthesize_report_starts_small_for_non_gemini_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parquet = _write_parquet(tmp_path, _long_docs())
    working = _FakeClient([FULL_REPORT])
    monkeypatch.setattr(syn, "build_client", lambda spec, cfg: working)

    result = await syn.synthesize_report(
        parquet, "Sri Lestari", "Sleman", env={"GROQ_API_KEY": "qk"}
    )

    assert result.ok and result.provider == "groq"
    # Provider non-Gemini langsung memakai tier 1 (prompt lebih ramping), lalu
    # Phase 3 hard guard (MMR + 3.2k token) memangkasnya agar tidak kena 413.
    cfg = LLMConfig()
    system = syn.build_system_prompt("id")
    _, expected, exact_est = syn.fit_user_prompt_to_token_cap(
        system,
        "Sri Lestari",
        "Sleman",
        syn.select_segments(
            _long_docs(),
            "Sri Lestari",
            "Sleman",
            max_segments=9,
            max_input_tokens=5000,
            max_chars_per_segment=1400,
            use_mmr=cfg.use_mmr,
            mmr_lambda=cfg.mmr_lambda,
        ),
        cfg.batch_max_input_tokens,
    )
    assert working.calls[0][1] == expected
    assert exact_est <= cfg.batch_max_input_tokens
    assert result.segments_used <= 9


# ---- Phase 5: evidence-only synthesis (anti-halusinasi) ---------------------


def test_build_evidence_segments_uses_grounding_quote_only() -> None:
    facts = [
        {
            "subject": "Sri Lestari",
            "predicate": "bekerja_sebagai",
            "object": "perawat",
            "sources": ["https://news.contoh.id/sri-lestari"],
            "quotes": ["Sri Lestari bekerja sebagai perawat"],
        }
    ]

    segments = syn.build_evidence_segments(_docs(), facts, max_chars_per_segment=300)

    assert len(segments) == 1
    assert segments[0].url == "https://news.contoh.id/sri-lestari"
    assert "Sri Lestari bekerja sebagai perawat" in segments[0].text
    assert "resep masakan" not in segments[0].text


def test_build_evidence_segments_falls_back_to_kg_fact_text() -> None:
    facts = [
        {
            "subject": "Sri Lestari",
            "predicate": "bekerja_di",
            "object": "Dinas Kesehatan Sleman",
            "sources": ["https://arsip.contoh.id/hilang"],
            "quotes": ["kutipan yang tidak ada di parquet"],
        }
    ]

    segments = syn.build_evidence_segments(_docs(), facts)

    assert len(segments) == 1
    assert segments[0].url == "https://arsip.contoh.id/hilang"
    assert segments[0].title.startswith("Knowledge Graph")
    assert "Dinas Kesehatan Sleman" in segments[0].text


@pytest.mark.asyncio
async def test_synthesize_report_evidence_mode_excludes_unvalidated_chunks(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeClient([FULL_REPORT])
    facts = [
        {
            "subject": "Sri Lestari",
            "predicate": "bekerja_sebagai",
            "object": "perawat",
            "sources": ["https://news.contoh.id/sri-lestari"],
            "quotes": ["Sri Lestari bekerja sebagai perawat"],
        }
    ]

    result = await syn.synthesize_report(
        parquet, "Sri Lestari", "Sleman", client=client, evidence_facts=facts
    )

    assert result.ok
    assert result.documents_used == 1
    assert result.sidecar is not None
    assert result.sidecar.sources_used == ["https://news.contoh.id/sri-lestari"]
    _system, user = client.calls[0]
    assert "Sri Lestari bekerja sebagai perawat" in user
    # Raw chunk from the unrelated parquet doc never leaves disk.
    assert "resep masakan" not in user
    assert "https://blog.contoh.id/resep" not in user


@pytest.mark.asyncio
async def test_synthesize_report_evidence_mode_skips_without_kg_facts(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeClient([FULL_REPORT])

    result = await syn.synthesize_report(
        parquet, "Sri Lestari", "", client=client, evidence_facts=[]
    )

    assert not result.ok
    assert "Knowledge Graph" in result.skipped_reason
    assert client.calls == []
