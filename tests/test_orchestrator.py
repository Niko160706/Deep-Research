"""Integration test: 70/30 context plumbing through the orchestrator.

The network is stubbed on both ends (mock search backend + stub scraper) so the
whole pipeline — decompose, search, clean (Fellegi-Sunter), graph, Louvain
pruning, report — runs offline and deterministically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from drx.brain.critic import CriticResult
from drx.brain.llm_synthesizer import LLMResult
from drx.config import Settings
from drx.models import Document, Fact, SearchResult, SourceRef
from drx.muscle.backends import MockSearchBackend
from drx.orchestrator import Orchestrator


class _StubScraper:
    """Drop-in for the network scraper: one clean document per result URL."""

    async def fetch(self, results):
        body = (
            "Entity A overview. This authoritative page describes entity A "
            "in detail, including its relations to entity B and related work. "
        )
        return [
            Document(url=r.url, title=r.title, text=body * 8)
            for r in results
        ]


def _settings(tmp_path: Path) -> Settings:
    settings = Settings.load(tmp_path / "missing.yaml")
    settings.paths.runs_dir = str(tmp_path)
    settings.cleaning.min_text_length = 50
    # Jalur LLM (agentic critic + sintesis) off secara default supaya test tidak
    # pernah menyentuh jaringan; test yang relevan meng-enable-nya eksplisit.
    settings.llm.enabled = False
    return settings


@pytest.mark.asyncio
async def test_context_plumbing_tree_and_report(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    orch = Orchestrator(
        settings, run_id="itest", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    report = await orch.run("entity")

    # 70/30 decomposition reached the orchestrator
    assert orch.initial_tasks
    anchored = [t for t in orch.initial_tasks if t.is_anchored]
    exploratory = [t for t in orch.initial_tasks if not t.is_anchored]
    assert anchored and exploratory
    assert len(anchored) > len(exploratory)
    # exploratory branch is sampled within the first two sub-queries
    assert any(not t.is_anchored for t in orch.initial_tasks[:2])
    # every initial sub-query is accounted for (no silent starvation)
    assert all(t.status != "pending" for t in orch.initial_tasks)
    assert any(t.status == "searched" for t in orch.initial_tasks)

    assert report is not None
    assert report.markdown  # narrative report produced
    (orch.run_dir / "report.md").write_text(report.markdown, encoding="utf-8")
    assert (orch.run_dir / "report.md").exists()
    assert (orch.run_dir / "manifest.json").exists()
    assert orch.graph_store.node_count() > 0


@pytest.mark.asyncio
async def test_early_stop_defers_until_exploratory_probe_runs(tmp_path: Path) -> None:
    """The 30% probe must run before a "self-answer sufficient" stop can fire."""
    settings = _settings(tmp_path)
    orch = Orchestrator(
        settings, run_id="itest3", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]
    # force the early-stop path on every cycle
    orch.evaluator.should_self_answer = lambda new_nodes: True  # type: ignore[assignment]

    async def _always_sufficient(graph):
        return True

    orch.evaluator.self_answer_sufficient = _always_sufficient  # type: ignore[assignment]

    await orch.run("entity")

    exploratory = [t for t in orch.initial_tasks if not t.is_anchored]
    assert exploratory
    assert all(t.status != "pending" for t in exploratory)


@pytest.mark.asyncio
async def test_no_context_runs_100pct_exploratory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    orch = Orchestrator(settings, run_id="itest2", context="", verbose=False)
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    await orch.run("entity")

    assert orch.initial_tasks
    assert all(not t.is_anchored for t in orch.initial_tasks)


class _RecordingScraper:
    """Records every URL handed to the fetcher (global dedup assertions)."""

    def __init__(self) -> None:
        self.urls: list[str] = []

    async def fetch(self, results):
        self.urls.extend(r.url for r in results)
        body = (
            "Entity A overview. This authoritative page describes entity A "
            "in detail, including its relations to entity B and related work. "
        )
        return [
            Document(url=r.url, title=r.title, text=body * 8) for r in results
        ]


class _CapturingBackend:
    """Records the exact query string sent to the search backend."""

    name = "capture"
    supports_boolean = True

    def __init__(self) -> None:
        self.queries: list[str] = []
        self._inner = MockSearchBackend()

    async def search(self, query: str, limit: int = 10):
        self.queries.append(query)
        return await self._inner.search(query, limit)


class _UniqueUrlBackend:
    """Distinct URLs for every query, so each sub-query finds fresh material."""

    name = "unique"
    supports_boolean = True

    async def search(self, query: str, limit: int = 10):
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
        return [
            SearchResult(
                url=f"https://contoh.id/{key}/{i}",
                title=f"Entity A overview {query}",
                snippet=f"entity A {query}",
                domain="contoh.id",
                score=1.0,
            )
            for i in range(2)
        ][:limit]


@pytest.mark.asyncio
async def test_lazy_decomposition_skips_remaining_subqueries(tmp_path: Path) -> None:
    """Enough valid docs on the first sub-query -> the rest is never executed."""
    settings = _settings(tmp_path)
    settings.search.lazy_min_docs = 2  # the stub yields 2 docs per sub-query
    orch = Orchestrator(
        settings, run_id="lazy", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    await orch.run("entity")

    searched = [t for t in orch.initial_tasks if t.status == "searched"]
    skipped = [t for t in orch.initial_tasks if t.status == "skipped"]
    assert len(searched) == 1  # quota saver: only the first sub-query ran
    assert skipped  # the rest are flagged, not silently pending
    assert orch._valid_docs >= 2


@pytest.mark.asyncio
async def test_lazy_disabled_runs_every_subquery(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.search.lazy_min_docs = 0  # disabled
    orch = Orchestrator(
        settings, run_id="nolazy", context="Surabaya, Informatika", verbose=False
    )
    orch.autosearch.backend = _UniqueUrlBackend()  # type: ignore[assignment]
    orch.survey.autosearch.backend = MockSearchBackend()
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    await orch.run("entity")

    assert all(t.status != "pending" for t in orch.initial_tasks)
    assert len([t for t in orch.initial_tasks if t.status == "searched"]) > 1


@pytest.mark.asyncio
async def test_global_dedup_cache_prevents_rescraping(tmp_path: Path) -> None:
    """A URL fetched on one path is never fetched again on another path."""
    settings = _settings(tmp_path)
    settings.search.lazy_min_docs = 0
    orch = Orchestrator(
        settings, run_id="dedup", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()  # same 2 URLs for every sub-query
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    recorder = _RecordingScraper()
    orch.scraper = recorder  # type: ignore[assignment]

    await orch.run("entity")

    assert recorder.urls  # something was fetched
    assert len(recorder.urls) == len(set(recorder.urls))  # nothing twice
    assert orch.seen_urls  # the global cache is populated
    assert orch.autosearch.seen_urls is orch.seen_urls  # shared by reference


@pytest.mark.asyncio
async def test_bracket_query_sanitized_and_aliases_plumbed(tmp_path: Path) -> None:
    """'Bayu Skak (Bayu Eko Moektito)' -> OR query + alias-aware entity gate."""
    settings = _settings(tmp_path)
    orch = Orchestrator(settings, run_id="alias", context="", verbose=False)
    backend = _CapturingBackend()
    orch.autosearch.backend = backend  # type: ignore[assignment]
    orch.survey.autosearch.backend = backend  # type: ignore[assignment]
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    await orch.run("Bayu Skak (Bayu Eko Moektito)")

    assert orch.main_query == '"Bayu Skak" OR "Bayu Eko Moektito"'
    assert orch.aliases == ["Bayu Skak", "Bayu Eko Moektito"]
    assert backend.queries  # the sanitized form reached the backend
    assert any('"Bayu Skak" OR "Bayu Eko Moektito"' in q for q in backend.queries)
    assert "(" not in backend.queries[0]


class _StrictPathBackend:
    """Homonym pages per sub-query: one social profile + one legit page.

    URLs differ per query (hash key) so the global dedup cache never hides a
    path: every sub-query — anchored 70% and exploratory 30% — is fetched and
    cleaned on its own.
    """

    name = "strict-path"
    supports_boolean = True

    async def search(self, query: str, limit: int = 10):
        key = hashlib.sha1(query.encode("utf-8")).hexdigest()[:6]
        return [
            SearchResult(
                url=f"https://facebook.com/pages/{key}/sri-lestari",
                title="Sri Lestari",
                snippet="Sri Lestari profil media sosial",
                domain="facebook.com",
                score=1.0,
            ),
            SearchResult(
                url=f"https://news.contoh.id/{key}/sri-lestari-perawat",
                title="Sri Lestari perawat Sleman",
                snippet="Sri Lestari bekerja di Sleman.",
                domain="news.contoh.id",
                score=1.0,
            ),
        ][:limit]


class _StrictPathScraper:
    """Body depends on the URL: only the legit page carries a context word."""

    async def fetch(self, results):
        docs = []
        for r in results:
            if "perawat" in r.url:
                body = "Sri Lestari bekerja sebagai perawat di Sleman. " * 8
            else:
                body = "Sri Lestari tinggal di kota lain dan berdagang. " * 8
            docs.append(Document(url=r.url, title=r.title, text=body))
        return docs


@pytest.mark.asyncio
async def test_strict_context_enforced_on_every_path_including_exploratory(
    tmp_path: Path, caplog
) -> None:
    """min_docs=0 runs all paths; homonyms must not survive any of them."""
    import logging

    settings = _settings(tmp_path)
    settings.search.lazy_min_docs = 0
    orch = Orchestrator(
        settings, run_id="strictall", context="Sleman, Kesehatan", verbose=False
    )
    backend = _StrictPathBackend()
    orch.autosearch.backend = backend  # type: ignore[assignment]
    orch.survey.autosearch.backend = backend  # type: ignore[assignment]
    orch.scraper = _StrictPathScraper()  # type: ignore[assignment]

    with caplog.at_level(logging.INFO, logger="drx.orchestrator"):
        report = await orch.run("Sri Lestari")

    # policy is announced as global (anchored 70% + exploratory 30%)
    assert "strict --context enforcement: ON for ALL paths" in caplog.text

    # the broad exploratory probe really ran...
    exploratory = [t for t in orch.initial_tasks if not t.is_anchored]
    assert exploratory
    assert any(t.status == "searched" for t in exploratory)
    assert any(t.status == "searched" for t in orch.initial_tasks if t.is_anchored)

    # ...yet only the page carrying a --context keyword survives, on any path
    kept = {d.url for d in orch.parquet.load_documents()}
    assert kept
    assert all("perawat" in url for url in kept)
    assert not any("facebook.com" in url for url in kept)
    assert report.markdown


# ---- Phase 2: LLM report synthesis ------------------------------------------

LLM_REPORT = """# Laporan Investigasi: Sri Lestari

## Profil & Identitas Utama
- Perawat di Kab. Sleman. [S1]

## Ringkasan Eksekutif
Narasi eksekutif. [S1]

## Rekam Jejak & Temuan Kunci
- Bekerja sejak 2015. [S1]

## Potensi Ambigu / Entitas Serupa
Tidak ada indikasi. [S1]

## Daftar Sumber / Sitasi
- [S1] https://news.contoh.id/a
"""


@pytest.mark.asyncio
async def test_synthesize_llm_writes_report_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drx.brain.llm_synthesizer as syn

    settings = _settings(tmp_path)
    settings.llm.enabled = True  # test ini memang menguji jalur sintesis
    orch = Orchestrator(settings, run_id="llm", context="Sleman", verbose=False)
    orch.entity = "Sri Lestari"  # biasanya diisi oleh run()
    orch.main_query = "Sri Lestari"
    orch.parquet.save(
        [
            Document(
                url="https://news.contoh.id/a",
                title="Sri Lestari perawat",
                text="Sri Lestari perawat Sleman. " * 10,
            )
        ]
    )
    orch.graph_store.resolve_and_add(
        [
            Fact(
                subject="Sri Lestari",
                predicate="bekerja_sebagai",
                object="perawat",
                provenance=[
                    SourceRef(
                        url="https://news.contoh.id/a",
                        quote="Sri Lestari perawat Sleman",
                    )
                ],
            )
        ]
    )

    async def fake_synthesize(
        parquet_path, target, context, *, config, env, language, evidence_facts=None
    ):
        assert str(parquet_path).endswith("raw_documents.parquet")
        assert target == "Sri Lestari"
        assert context == "Sleman"
        # Phase 5: the synthesizer only receives verified KG facts, never the
        # raw parquet corpus.
        assert evidence_facts, "expected verified KG facts"
        assert evidence_facts[0]["subject"] == "Sri Lestari"
        assert evidence_facts[0]["predicate"] == "bekerja_sebagai"
        return LLMResult(
            provider="gemini",
            model="gemini-2.5-flash",
            markdown=LLM_REPORT,
            documents_used=1,
            segments_used=2,
            input_tokens_est=90,
            total_tokens=150,
            sections_complete=True,
        )

    monkeypatch.setattr(syn, "synthesize_report", fake_synthesize)

    tokens_before = orch.budget.tokens
    result = await orch.synthesize_llm()

    assert result is not None and result.ok
    report_path = orch.run_dir / "report_llm.md"
    assert report_path.exists()
    assert "## Ringkasan Eksekutif" in report_path.read_text(encoding="utf-8")
    assert orch.budget.tokens == tokens_before + 150

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_synthesis"]["provider"] == "gemini"
    assert manifest["llm_synthesis"]["model"] == "gemini-2.5-flash"
    assert manifest["llm_synthesis"]["total_tokens"] == 150
    assert manifest["llm_synthesis"]["sections_complete"] is True


@pytest.mark.asyncio
async def test_synthesize_llm_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Interactive ``[3] Selesai`` bila disusul main.py tidak menulis 2x."""
    import drx.brain.llm_synthesizer as syn

    settings = _settings(tmp_path)
    settings.llm.enabled = True
    orch = Orchestrator(settings, run_id="llm-once", context="", verbose=False)
    orch.entity = "Sri Lestari"
    orch.main_query = "Sri Lestari"
    orch.parquet.save(
        [
            Document(
                url="https://news.contoh.id/a",
                title="Sri Lestari perawat",
                text="Sri Lestari perawat Sleman. " * 10,
            )
        ]
    )
    calls = {"count": 0}

    async def fake_synthesize(
        parquet_path, target, context, *, config, env, language, evidence_facts=None
    ):
        calls["count"] += 1
        return LLMResult(
            provider="gemini",
            model="gemini-2.5-flash",
            markdown=LLM_REPORT,
            documents_used=1,
            segments_used=1,
            input_tokens_est=50,
            total_tokens=100,
            sections_complete=True,
        )

    monkeypatch.setattr(syn, "synthesize_report", fake_synthesize)

    first = await orch.synthesize_llm()
    second = await orch.synthesize_llm()

    assert calls["count"] == 1  # hanya satu panggilan LLM nyata
    assert second is first
    assert orch.budget.tokens == 100  # tidak double-count token


@pytest.mark.asyncio
async def test_synthesize_llm_disabled_returns_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drx.brain.llm_synthesizer as syn

    settings = _settings(tmp_path)
    settings.llm.enabled = False
    orch = Orchestrator(settings, run_id="llm-off", verbose=False)

    async def bomb(*args, **kwargs):
        raise AssertionError("synthesize_report must not be called when disabled")

    monkeypatch.setattr(syn, "synthesize_report", bomb)
    assert await orch.synthesize_llm() is None
    assert not (orch.run_dir / "report_llm.md").exists()


@pytest.mark.asyncio
async def test_synthesize_llm_skips_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    import logging

    settings = _settings(tmp_path)
    settings.llm.enabled = True
    orch = Orchestrator(settings, run_id="llm-nokey", verbose=False)
    orch.parquet.save(
        [Document(url="https://x.id/a", title="t", text="Isi dokumen contoh. " * 20)]
    )
    # Tanpa key sama sekali: sintesis harus dilewati, bukan crash / network call.
    monkeypatch.setattr(orch, "_llm_env", dict)

    with caplog.at_level(logging.INFO, logger="drx.orchestrator"):
        result = await orch.synthesize_llm()

    assert result is not None
    assert not result.ok
    assert "API key" in result.skipped_reason
    assert "LLM synthesis skipped" in caplog.text
    assert not (orch.run_dir / "report_llm.md").exists()

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_synthesis"]["skipped_reason"]


@pytest.mark.asyncio
async def test_synthesize_llm_error_is_graceful(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drx.brain.llm_synthesizer as syn

    settings = _settings(tmp_path)
    settings.llm.enabled = True
    orch = Orchestrator(settings, run_id="llm-err", verbose=False)
    orch.parquet.save(
        [Document(url="https://x.id/b", title="t", text="Isi dokumen contoh. " * 20)]
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(syn, "synthesize_report", boom)

    result = await orch.synthesize_llm()

    assert result is not None and not result.ok
    assert "provider down" in result.error
    assert not (orch.run_dir / "report_llm.md").exists()


# ---- Phase 2b: agentic critic loop ------------------------------------------

class _CriticScript:
    """Scriptable stand-in for ``brain.critic.analyze_fact_gaps``."""

    def __init__(self, verdicts: list[dict]) -> None:
        self.verdicts = list(verdicts)
        self.calls = 0

    async def __call__(
        self, parquet_path, target, context, *, config, env, language
    ):
        self.calls += 1
        if not self.verdicts:
            raise AssertionError("critic dipanggil lebih dari yang diharapkan")
        verdict = self.verdicts.pop(0)
        return CriticResult(**verdict)


def _critic_settings(tmp_path: Path) -> Settings:
    settings = _settings(tmp_path)
    settings.llm.enabled = True
    # Unit test mengikat jalur LEGACY critic (fake analyze_fact_gaps) supaya
    # tidak menyentuh jaringan: triple critic, planner, dan ekstraksi LLM
    # punya test-nya sendiri dengan client injeksi.
    settings.llm.critic_reads_triples = False
    settings.llm.planner_enabled = False
    settings.llm.extraction_enabled = False
    settings.search.lazy_min_docs = 0
    return settings


@pytest.mark.asyncio
async def test_critic_loop_executes_gap_queries_then_rechecks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Gap ditemukan -> sub-kueri dieksekusi -> critic dipanggil ulang."""
    import logging

    import drx.brain.critic as critic_mod

    settings = _critic_settings(tmp_path)
    orch = Orchestrator(
        settings, run_id="critic-loop", context="Surabaya, Informatika", verbose=False
    )
    orch.autosearch.backend = _UniqueUrlBackend()  # type: ignore[assignment]
    orch.survey.autosearch.backend = MockSearchBackend()
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    script = _CriticScript(
        [
            {
                "is_sufficient": False,
                "critique_summary": "Jabatan terbaru belum ada.",
                "suggested_sub_queries": [
                    "Entity A jabatan terbaru",
                    "Entity A profil lengkap",
                ],
            },
            {"is_sufficient": True, "critique_summary": "Sudah cukup."},
        ]
    )
    monkeypatch.setattr(critic_mod, "analyze_fact_gaps", script)

    with caplog.at_level(logging.INFO, logger="drx.orchestrator"):
        await orch.run("entity")

    assert script.calls == 2  # critic -> gap queries -> critic ulang
    assert (
        "Critic menemukan gap data: Jabatan terbaru belum ada. "
        "Mengeksekusi sub-kueri lanjutan..." in caplog.text
    )
    assert orch.critic_rounds == 1
    assert orch.critic_queries == [
        "Entity A jabatan terbaru",
        "Entity A profil lengkap",
    ]
    assert orch.critic_result is not None and orch.critic_result.is_sufficient

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["critic"]["is_sufficient"] is True
    assert manifest["critic"]["gap_rounds"] == 1
    assert manifest["critic"]["gap_queries"] == orch.critic_queries


@pytest.mark.asyncio
async def test_critic_loop_stops_when_sufficient(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import drx.brain.critic as critic_mod

    settings = _critic_settings(tmp_path)
    orch = Orchestrator(
        settings, run_id="critic-ok", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    script = _CriticScript(
        [{"is_sufficient": True, "critique_summary": "Cukup sejak awal."}]
    )
    monkeypatch.setattr(critic_mod, "analyze_fact_gaps", script)

    await orch.run("entity")

    assert script.calls == 1  # tidak ada iterasi tambahan
    assert orch.critic_rounds == 0
    assert orch.critic_queries == []


@pytest.mark.asyncio
async def test_critic_loop_respects_max_loops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    import logging

    import drx.brain.critic as critic_mod

    settings = _critic_settings(tmp_path)
    settings.llm.critic_max_loops = 1  # hanya boleh satu ronde gap-query
    orch = Orchestrator(
        settings, run_id="critic-max", context="Surabaya, Informatika", verbose=False
    )
    orch.autosearch.backend = _UniqueUrlBackend()  # type: ignore[assignment]
    orch.survey.autosearch.backend = MockSearchBackend()
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    gap = {
        "is_sufficient": False,
        "critique_summary": "Masih bolong.",
        "suggested_sub_queries": ["Entity A query satu", "Entity A query dua"],
    }
    script = _CriticScript([dict(gap), dict(gap)])
    monkeypatch.setattr(critic_mod, "analyze_fact_gaps", script)

    with caplog.at_level(logging.INFO, logger="drx.orchestrator"):
        await orch.run("entity")

    assert script.calls == 2  # 1 ronde gap + 1 verifikasi terakhir
    assert orch.critic_rounds == 1
    assert len(orch.critic_queries) == 2
    assert "loop habis" in caplog.text


@pytest.mark.asyncio
async def test_critic_loop_skipped_when_llm_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--no-llm (llm.enabled=False) juga mematikan jalur critic."""
    import drx.brain.critic as critic_mod

    settings = _settings(tmp_path)  # enabled=False bawaan helper
    orch = Orchestrator(
        settings, run_id="critic-off", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    async def bomb(*args, **kwargs):
        raise AssertionError("critic tidak boleh jalan saat llm.enabled=False")

    monkeypatch.setattr(critic_mod, "analyze_fact_gaps", bomb)

    await orch.run("entity")

    assert orch.critic_attempts == 0
    assert orch.critic_result is None


@pytest.mark.asyncio
async def test_critic_loop_skips_gracefully_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """Tanpa API key: critic dilewati, run tetap sukses (tanpa network call)."""
    import logging

    settings = _critic_settings(tmp_path)
    orch = Orchestrator(
        settings, run_id="critic-nokey", context="Surabaya, Informatika", verbose=False
    )
    mock = MockSearchBackend()
    orch.autosearch.backend = mock
    orch.survey.autosearch.backend = mock
    orch.scraper = _StubScraper()  # type: ignore[assignment]
    monkeypatch.setattr(orch, "_llm_env", dict)  # pastikan env benar-benar kosong

    with caplog.at_level(logging.INFO, logger="drx.orchestrator"):
        report = await orch.run("entity")

    assert report.markdown  # run tetap menghasilkan laporan
    assert orch.critic_attempts == 1
    assert orch.critic_skipped
    assert "critic skipped" in caplog.text

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["critic"]["skipped_reason"]
    assert manifest["critic"]["gap_rounds"] == 0


@pytest.mark.asyncio
async def test_synthesize_llm_writes_sidecar_json_and_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import networkx as nx

    import drx.brain.llm_synthesizer as syn

    settings = _settings(tmp_path)
    settings.llm.enabled = True
    orch = Orchestrator(settings, run_id="llm-sidecar", verbose=False)
    orch.entity = "Sri Lestari"
    orch.main_query = "Sri Lestari"
    orch.parquet.save(
        [
            Document(
                url="https://news.contoh.id/a",
                title="Sri Lestari perawat",
                text="Sri Lestari perawat Sleman. " * 10,
            )
        ]
    )
    # main.py sudah menulis sidecar ekstraktif Phase 1 sebelum sintesis LLM
    (orch.run_dir / "report_sidecar.json").write_text(
        json.dumps({"documents": 1, "entities": ["Sri Lestari"]}), encoding="utf-8"
    )

    sidecar = syn.ReportSidecar(
        target_entity="Sri Lestari",
        verified_facts=[
            syn.VerifiedFact(
                subject="Sri Lestari",
                predicate="bekerja sebagai",
                object="perawat",
                source_url="https://news.contoh.id/a",
            ),
            syn.VerifiedFact(
                subject="Sri Lestari",
                predicate="berlokasi di",
                object="Sleman",
                source_url="https://news.contoh.id/a",
            ),
        ],
        disambiguated_entities=["Sri Lestari (Jakarta)"],
        sources_used=["https://news.contoh.id/a"],
    )

    async def fake_synthesize(
        parquet_path, target, context, *, config, env, language, evidence_facts=None
    ):
        return LLMResult(
            provider="gemini",
            model="gemini-2.5-flash",
            markdown=LLM_REPORT,
            documents_used=1,
            segments_used=1,
            input_tokens_est=80,
            total_tokens=100,
            sections_complete=True,
            sidecar=sidecar,
            sidecar_complete=True,
        )

    monkeypatch.setattr(syn, "synthesize_report", fake_synthesize)

    result = await orch.synthesize_llm()

    assert result is not None and result.ok
    data = json.loads(
        (orch.run_dir / "report_sidecar.json").read_text(encoding="utf-8")
    )
    assert data["target_entity"] == "Sri Lestari"
    assert len(data["verified_facts"]) == 2
    assert data["disambiguated_entities"] == ["Sri Lestari (Jakarta)"]
    assert data["sources_used"] == ["https://news.contoh.id/a"]
    assert data["phase1"] == {"documents": 1, "entities": ["Sri Lestari"]}

    graph = nx.read_graphml(orch.run_dir / "graph.graphml")
    by_label = {
        data.get("label"): node
        for node, data in graph.nodes(data=True)
    }
    # Phase 3: node key adalah ID stabil (n1, n2, ...), label tetap manusiawi.
    assert by_label["perawat"].startswith("n")
    sri_id = by_label["Sri Lestari"]
    perawat_id = by_label["perawat"]
    edge_payload = graph.get_edge_data(sri_id, perawat_id)
    if isinstance(edge_payload, dict) and "predicate" in edge_payload:
        predicate = edge_payload["predicate"]
    else:
        predicate = next(iter(edge_payload.values()))["predicate"]
    assert predicate == "bekerja sebagai"

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["llm_synthesis"]["sidecar_complete"] is True
    assert manifest["llm_synthesis"]["sidecar_verified_facts"] == 2


@pytest.mark.asyncio
async def test_run_uses_query_planner_and_grounded_triples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 3 wiring: planner task masuk frontier, triple LLM masuk graph."""
    from drx.brain import extractors as ex
    from drx.brain import query_planner as qp

    settings = _settings(tmp_path)
    settings.llm.enabled = True
    settings.llm.critic_enabled = False  # test ini fokus ke planner + ekstraksi
    settings.search.lazy_min_docs = 0
    orch = Orchestrator(settings, run_id="phase3", context="", verbose=False)

    class _StubBackend:
        async def search(self, query, limit=10):
            clean = query.strip('" ')
            return [
                SearchResult(
                    url="https://contoh.id/profil",
                    title=f"{clean} profil dan karier",
                    snippet=f"Profil {clean} perawat Dinas Kesehatan Sleman.",
                    domain="contoh.id",
                )
            ]

    stub = _StubBackend()
    orch.autosearch.backend = stub
    orch.survey.autosearch.backend = stub
    orch.scraper = _StubScraper()  # type: ignore[assignment]

    planned = qp.QueryPlanResult(
        queries=["Sri Lestari profil lengkap"],
        rationale="variasi berbenih",
        provider="fake",
        model="fake-1",
        input_tokens_est=80,
        total_tokens=120,
    )

    async def fake_plan(target_entity, context="", **kwargs):
        assert target_entity == "Sri Lestari"
        return planned

    triple = ex.GroundedTriple(
        subject="Sri Lestari",
        predicate="bekerja_sebagai",
        object="perawat",
        source_id="S1",
        short_quote="Sri Lestari profil dan karier",
        source_url="https://contoh.id/profil",
        confidence=0.7,
    )

    async def fake_extract(documents, target_entity, context="", **kwargs):
        return ex.ExtractionResult(
            triples=list(triple and [triple]),
            batches=1,
            documents=len(documents),
            input_tokens_est=200,
            total_tokens=260,
        )

    monkeypatch.setattr(qp, "plan_queries", fake_plan)
    monkeypatch.setattr(ex, "extract_triples", fake_extract)

    report = await orch.run("Sri Lestari")

    assert report is not None
    assert orch.planner_result is planned
    assert any(
        t.rationale.startswith("P0 query planner") for t in orch.initial_tasks
    )
    assert orch.triples_extracted >= 1
    assert any(
        "bekerja_sebagai" in triple["predicate"]
        for triple in orch.graph_store.triples(limit=10_000)
    )
    assert orch.tokens_total >= 120  # planner + ekstraksi tercatat

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query_planner"]["queries"] == ["Sri Lestari profil lengkap"]
    assert manifest["triple_extraction"]["triples"] >= 1
    assert manifest["token_usage"]["total"] == orch.tokens_total
