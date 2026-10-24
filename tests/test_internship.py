"""Tests for the additive Internship Hunter Engine (A-G source contract).

These tests are fully offline: the engine receives ``MockFetcher`` or a tiny
fake fetcher.  No API keys, browsers or network connections are needed.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
from pathlib import Path

import main
from drx.config import Settings
from drx.internship.curator import InternshipCurator
from drx.internship.diagnostic_logger import DiagnosticLogger
from drx.internship.engine import run_internship
from drx.internship.models import RawJob
from drx.internship.planner import _pure_role, fallback_plan
from drx.internship.site_parsers import (
    ALL_SOURCES,
    FetchResponse,
    MockFetcher,
    _mock_page,  # test fixture helper
    detect_block_reason,
    fetch_source,
    glints_url,
    its_career_url,
    jobstreet_url,
    kalibrr_url,
    linkedin_dork,
    linkedin_dork_url,
    msib_url,
    nodeflair_url,
    parse_glints,
    parse_its_career,
    parse_jobstreet,
    parse_kalibrr,
    parse_linkedin_api_results,
    parse_msib,
    parse_nodeflair,
    parse_site,
    spec_url,
)

# ---- targeted URLs (A-G) ----------------------------------------------------


def test_exact_target_urls_match_source_spec():
    assert jobstreet_url("AI Engineer") == (
        "https://id.jobstreet.com/id/ai-engineer-intern-jobs/in-Indonesia"
        "?daterange=31"
    )
    assert glints_url("AI Engineer") == (
        "https://glints.com/id/en/opportunities/jobs/explore"
        "?keyword=AI+Engineer&country=ID&jobTypes=INTERNSHIP"
    )
    assert kalibrr_url("AI Engineer") == (
        "https://www.kalibrr.id/id-ID/home/w/100-internship-_-ojt/k/AI%20Engineer"
    )
    assert its_career_url("AI Engineer") == (
        "https://careers.its.ac.id/lowongan?keywords=AI+Engineer"
    )
    assert msib_url("AI Engineer") == (
        "https://simbelmawa.kemdiktisaintek.go.id/magang/lowongan"
        "?cari=AI+Engineer"
    )
    assert nodeflair_url("AI Engineer") == (
        "https://nodeflair.com/jobs?query=AI+Engineer"
        "&seniorities%5B%5D=intern&countries%5B%5D=Indonesia"
    )
    assert linkedin_dork("AI Engineer") == (
        'site:linkedin.com/jobs "internship" "AI Engineer" Indonesia'
    )
    assert linkedin_dork_url("AI Engineer").startswith("https://www.google.com/search?q=")


# ---- planner (pure roles contract) ------------------------------------------


def test_pure_role_strips_internship_and_location():
    assert _pure_role("AI Engineer") == "AI Engineer"
    assert _pure_role("AI Engineer Intern") == "AI Engineer"
    assert _pure_role("AI Engineer Intern Indonesia") == "AI Engineer"
    assert _pure_role("Data Analyst Intern") == "Data Analyst"
    assert _pure_role("Backend Engineer Magang Indonesia") == "Backend Engineer"


def test_role_planner_fallback_expands_pure_roles():
    plan = fallback_plan(
        "role", role="AI Engineer", location="Surabaya OR Remote", work_mode="Remote"
    )
    assert plan.used_fallback is True
    assert plan.roles[:3] == ["AI Engineer", "Machine Learning", "Data Science"]
    assert plan.queries == plan.roles
    for query in plan.queries:
        assert "Intern" not in query
        assert "Remote" not in query
        assert "Surabaya" not in query


def test_skill_planner_fallback_maps_three_pure_roles():
    plan = fallback_plan(
        "skill", skills=["Python", "Go", "SQL", "Docker"], location="Remote"
    )
    assert len(plan.roles) == 3
    assert "AI/ML Engineer" in plan.roles
    assert "Data Analyst" in plan.roles
    assert "Backend Engineer" in plan.roles
    assert plan.queries == plan.roles
    assert all("Intern" not in role for role in plan.roles)


def test_discovery_planner_keeps_location_as_metadata_only():
    plan = fallback_plan("discovery", location="Surabaya", work_mode="Hybrid")
    assert plan.queries == plan.roles
    assert plan.roles == ["Software Engineer", "Data Analyst", "AI Engineer"]
    for query in plan.queries:
        assert "Surabaya" not in query
        assert "Hybrid" not in query


# ---- diagnostics ------------------------------------------------------------


def test_diagnostic_logger_writes_required_fields_and_table(tmp_path: Path):
    logger = DiagnosticLogger("run-diagnostic")
    ok = logger.begin("Glints", "https://glints.example/1", query="AI")
    logger.complete(
        ok,
        status="SUCCESS",
        http_status_code=200,
        latency_ms=412,
        items_extracted_count=3,
        raw_chars_scraped=5210,
    )
    blocked = logger.begin("Kalibrr", "https://kalibrr.example/1")
    logger.fail(
        blocked,
        status="BLOCKED",
        http_status_code=403,
        failure_reason="Cloudflare CAPTCHA Blocked",
        latency_ms=900,
    )
    path = logger.write(tmp_path)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "internship_source_diagnostics.json"
    assert document["summary"]["sources"] == 2
    first = document["sources"][0]
    for key in (
        "source_name",
        "target_url",
        "access_status",
        "http_status_code",
        "failure_reason",
        "latency_ms",
        "raw_chars_scraped",
        "items_extracted_count",
    ):
        assert key in first
    assert first["raw_chars_scraped"] == 5210
    assert document["sources"][1]["access_status"] == "BLOCKED"
    assert "Cloudflare CAPTCHA" in document["sources"][1]["failure_reason"]
    assert "failed" not in document["summary"]
    table = logger.summary_table()
    assert "SUMBER" in table
    assert "RAW" in table
    assert "BLOCKED" in table
    assert "Cloudflare CAPTCHA Blocked" in table


def test_unknown_status_degrades_within_allowed_set():
    logger = DiagnosticLogger("run-diagnostic")
    record = logger.begin("JobStreet", "https://jobstreet.example/1")
    logger.fail(record, status="FAILED", failure_reason="legacy")
    assert record.access_status in {"SUCCESS", "EMPTY_RESULT", "BLOCKED", "TIMEOUT"}


def test_detect_block_reason_covers_antibot_pages():
    assert "CAPTCHA" in detect_block_reason(403, "<html>verify you are human</html>")
    assert "anti-bot" in detect_block_reason(999, "LinkedIn").lower()
    assert detect_block_reason(200, "<html><h1>Jobs</h1></html>") == ""


# ---- site card parsers ------------------------------------------------------


_JOBSTREET_URL = jobstreet_url("AI Engineer")
_GLINTS_URL = glints_url("AI Engineer")
_KALIBRR_URL = kalibrr_url("AI Engineer")
_ITS_URL = its_career_url("AI Engineer")
_MSIB_URL = msib_url("AI Engineer")
_NODEFLAIR_URL = nodeflair_url("AI Engineer")


def test_parse_jobstreet_normal_job_cards():
    html = _mock_page("jobstreet", "AI Engineer")
    jobs = parse_jobstreet(html, _JOBSTREET_URL, "JobStreet")
    assert len(jobs) == 2
    assert jobs[0].role_title == "AI Engineer"
    assert jobs[0].company_name == "Nusantara Tech"
    assert jobs[0].location == "Jakarta"
    assert jobs[0].apply_url == "https://id.jobstreet.com/id/job/74000001-ai-engineer"
    assert jobs[0].is_internship_hint is True
    assert jobs[1].role_title == "Data Analyst Intern"


def test_parse_glints_uses_job_links_and_ancestor_card_text():
    html = _mock_page("glints", "AI Engineer")
    jobs = parse_glints(html, _GLINTS_URL, "Glints")
    assert jobs
    for job in jobs:
        assert "/opportunities/jobs/" in job.apply_url
        assert "/explore" not in job.apply_url
    first = next(job for job in jobs if "Engineer" in job.role_title)
    assert first.role_title == "AI Engineer"
    assert first.company_name == "Nusantara Tech"


def test_parse_kalibrr_anchors_by_itemprop_name():
    html = _mock_page("kalibrr", "AI Engineer")
    jobs = parse_kalibrr(html, _KALIBRR_URL, "Kalibrr")
    assert len(jobs) == 2
    assert jobs[0].role_title == "AI Engineer"
    assert jobs[0].company_name == "Nusantara Tech"
    assert jobs[0].location == "Jakarta"
    assert "/c/nusantaratech/jobs/123-" in jobs[0].apply_url


def test_parse_its_inner_box_cards():
    html = _mock_page("its", "AI Engineer")
    jobs = parse_its_career(html, _ITS_URL, "ITS Career Center")
    assert len(jobs) == 2
    assert jobs[0].role_title == "AI Engineer"
    assert jobs[0].company_name == "Nusantara Tech"
    assert jobs[0].apply_url == "https://careers.its.ac.id/lowongan/detail/74000001"


def test_parse_msib_border_cards():
    html = _mock_page("msib", "AI Engineer")
    jobs = parse_msib(html, _MSIB_URL, "MSIB")
    assert len(jobs) == 2
    assert jobs[0].role_title == "AI Engineer"
    assert jobs[0].apply_url.endswith("/magang/lowongan/74000001")


def test_parse_nodeflair_job_links_only():
    html = _mock_page("nodeflair", "AI Engineer")
    jobs = parse_nodeflair(html, _NODEFLAIR_URL, "NodeFlair")
    assert len(jobs) == 2
    assert jobs[0].role_title == "AI Engineer"
    assert all("/jobs/" in job.apply_url for job in jobs)
    assert all(not job.apply_url.startswith("https://nodeflair.com/jobs?") for job in jobs)


def test_parse_site_body_inner_text_fallback():
    html = "<html><body class='css-hash'><p>Baris pertama magang</p><p>Baris kedua detail</p></body></html>"
    jobs, raw_text = parse_site("jobstreet", html, _JOBSTREET_URL, "JobStreet")
    assert jobs == []
    assert "Baris pertama magang" in raw_text
    assert "Baris kedua detail" in raw_text


def test_mock_fetcher_covers_every_source_a_g():
    async def scenario():
        fetcher = MockFetcher()
        out = {}
        for spec in ALL_SOURCES:
            content = await fetch_source(
                source_name=spec.name,
                source_key=spec.key,
                target_url=spec_url(spec, "AI Engineer"),
                query="AI Engineer",
                fetcher=fetcher,
            )
            out[spec.name] = content
        return out

    contents = asyncio.run(scenario())
    assert set(contents) == {
        "JobStreet",
        "Glints",
        "Kalibrr",
        "ITS Career Center",
        "MSIB",
        "NodeFlair",
        "LinkedIn Jobs",
    }
    for name, content in contents.items():
        assert content.access_status == "SUCCESS", (name, content.failure_reason)
        assert content.raw_chars_scraped > 0
    linkedin = contents["LinkedIn Jobs"]
    assert any(job.apply_url.startswith("https://www.linkedin.com/jobs/") for job in linkedin.raw_jobs)
    assert "Cuplikan tetap dikirim" in linkedin.raw_text


def test_classified_statuses_never_use_failed():
    blocked = _content_of(
        FetchResponse(
            url="https://x", status=403, text="<html>Just a moment Cloudflare captcha</html>"
        )
    )
    timeout = _content_of(
        FetchResponse(url="https://x", status=0, error="timeout 15s", error_kind="timeout")
    )
    empty = _content_of(
        FetchResponse(url="https://x", status=200, text="<html><body></body></html>")
    )
    not_found = _content_of(FetchResponse(url="https://x", status=404, text="<html>gone</html>"))
    assert blocked.access_status == "BLOCKED"
    assert timeout.access_status == "TIMEOUT"
    assert empty.access_status == "EMPTY_RESULT"
    assert not_found.access_status == "EMPTY_RESULT"


def _content_of(response: FetchResponse):
    from drx.internship.site_parsers import content_from_response

    return content_from_response(
        response,
        source_name="Mock",
        source_key="jobstreet",
        target_url=response.url,
        query="AI Engineer",
    )


def test_linkedin_api_parser_keeps_snippets_and_job_urls():
    payload = {
        "results": [
            {
                "title": "AI Engineer Intern | LinkedIn",
                "url": "https://www.linkedin.com/jobs/view/123",
                "content": "Magang AI Engineer hybrid, skill Python.",
            },
            {
                "title": "80 pekerjaan AI Engineer di Indonesia",
                "url": "https://id.linkedin.com/jobs/ai-engineer-jobs",
                "content": "halaman listing agregat",
            },
            {
                "title": "Feed update | LinkedIn",
                "url": "https://www.linkedin.com/feed/update/1",
                "content": "bukan lowongan",
            },
        ]
    }
    jobs = parse_linkedin_api_results(payload, "https://x", "LinkedIn Jobs")
    assert [job.role_title for job in jobs] == ["AI Engineer Intern"]
    assert jobs[0].apply_url == "https://www.linkedin.com/jobs/view/123"

    from drx.internship.site_parsers import _linkedin_snippets_text

    raw_text = _linkedin_snippets_text(payload)
    # Snippets of every result stay available for the LLM curator, while only
    # single vacancies (/jobs/view/) become structured raw jobs.
    assert "halaman listing agregat" in raw_text
    assert "bukan lowongan" in raw_text
    assert all("/jobs/view/" in job.apply_url for job in jobs)


# ---- curator (WFH/WFO/Hybrid contract) --------------------------------------


def test_work_mode_contract_is_wfh_wfo_hybrid():
    from drx.internship.curator import _normalise_work_mode, normalise_job

    assert _normalise_work_mode("Remote") == "WFH"
    assert _normalise_work_mode("work from home") == "WFH"
    assert _normalise_work_mode("On-site") == "WFO"
    assert _normalise_work_mode("WFO kantor") == "WFO"
    assert _normalise_work_mode("Hybrid") == "Hybrid"
    job = normalise_job(
        {
            "role_title": "AI Engineer Intern",
            "company_name": "Acme",
            "location": "Jakarta",
            "work_mode": "Remote",
            "apply_url": "https://example.test/apply",
            "is_valid_internship": True,
        }
    )
    assert job is not None
    assert job.work_mode == "WFH"


def test_curator_filters_fulltime_even_if_llm_returns_it():
    class FakeClient:
        name = "fake"
        model = "fake-model"

        async def generate(self, system, user, *, json_mode=False):
            return (
                json.dumps(
                    {
                        "jobs": [
                            {
                                "role_title": "AI Engineer Intern",
                                "company_name": "Acme",
                                "location": "Jakarta",
                                "work_mode": "WFH",
                                "requirements": ["Python"],
                                "deadline": "2026-12-31",
                                "apply_url": "https://example.test/intern",
                                "is_valid_internship": True,
                            },
                            {
                                "role_title": "Senior AI Engineer",
                                "company_name": "Acme",
                                "location": "Jakarta",
                                "apply_url": "https://example.test/senior",
                                "is_valid_internship": False,
                            },
                        ]
                    }
                ),
                {},
            )

    async def scenario():
        curator = InternshipCurator(Settings().llm, client=FakeClient())
        return await curator.curate(
            [RawJob(role_title="AI Engineer Intern", apply_url="https://example.test/a")],
            source_name="Fake",
            source_url="https://example.test",
        )

    result = asyncio.run(scenario())
    assert len(result.jobs) == 1
    assert result.jobs[0].role_title == "AI Engineer Intern"
    assert result.jobs[0].is_valid_internship is True


def test_normalizer_rejects_senior_and_fulltime():
    async def scenario():
        config = Settings().llm.model_copy(update={"enabled": False})
        curator = InternshipCurator(config, client=None)
        return await curator.curate(
            [
                RawJob(
                    role_title="Senior Backend Engineer",
                    company_name="Acme",
                    apply_url="https://example.test/senior",
                ),
                RawJob(
                    role_title="Backend Engineer Intern",
                    company_name="Acme",
                    work_mode="WFH",
                    apply_url="https://example.test/intern",
                ),
            ],
            source_name="Fake",
            source_url="https://example.test",
        )

    result = asyncio.run(scenario())
    assert [job.role_title for job in result.jobs] == ["Backend Engineer Intern"]
    assert result.jobs[0].work_mode == "WFH"


# ---- engine / report / graph ------------------------------------------------


def _settings(tmp_path: Path, *, no_llm: bool = True) -> Settings:
    settings = Settings()
    settings.paths.runs_dir = str(tmp_path / "runs")
    settings.llm.enabled = not no_llm
    return settings


def test_engine_offline_end_to_end(tmp_path: Path):
    async def scenario():
        return await run_internship(
            _settings(tmp_path),
            approach="role",
            role="AI Engineer",
            no_llm=True,
            mock=True,
            run_id="intern-test",
            verbose=False,
        )

    result = asyncio.run(scenario())
    assert result.jobs
    assert all(job.is_valid_internship for job in result.jobs)
    assert result.diagnostics_path is not None and result.diagnostics_path.exists()
    assert result.report_path is not None and result.report_path.exists()
    assert result.results_path is not None and result.results_path.exists()
    assert result.graph_path is not None and result.graph_path.exists()

    diagnostics = json.loads(result.diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["summary"]["sources"] >= 7
    assert diagnostics["summary"]["success"] >= 1
    allowed_statuses = {"SUCCESS", "EMPTY_RESULT", "BLOCKED", "TIMEOUT"}
    for source in diagnostics["sources"]:
        assert set(source) >= {
            "source_name",
            "target_url",
            "access_status",
            "http_status_code",
            "latency_ms",
            "raw_chars_scraped",
            "items_extracted_count",
            "failure_reason",
        }
        assert source["access_status"] in allowed_statuses

    report = (result.run_dir / "report_llm.md").read_text(encoding="utf-8")
    top_section = report.split("## Strategi Pencarian", 1)[0]
    assert "## Tabel Ringkasan Magang Siap Apply" in top_section
    assert "AI Engineer Intern" in top_section

    graph = (result.run_dir / "graph_interactive.html").read_text(encoding="utf-8")
    assert "hierarchical" in graph
    assert '"direction": "UD"' in graph or "direction: 'UD'" in graph
    assert "INTERNSHIP HUNTER" in graph
    assert "KUERI/SARAN" in graph
    assert "#082F49" in graph


def test_graph_hierarchy_root_company_position(tmp_path: Path):
    from drx.internship.graph import build_internship_store
    from drx.internship.models import InternshipJob, InternshipPlan

    plan = InternshipPlan(approach="role", roles=["AI Engineer"], queries=["AI Engineer"])
    jobs = [
        InternshipJob(
            role_title="AI Engineer Intern",
            company_name="Nusantara Tech",
            location="Jakarta",
            work_mode="WFH",
            requirements=[],
            deadline="",
            apply_url="https://example.test/1",
            is_valid_internship=True,
        ),
        InternshipJob(
            role_title="ML Intern",
            company_name="Nusantara Tech",
            location="Surabaya",
            work_mode="Hybrid",
            requirements=[],
            deadline="",
            apply_url="https://example.test/2",
            is_valid_internship=True,
        ),
    ]
    store = build_internship_store(tmp_path / "g", plan, jobs)
    levels = {
        store.label_of(node): attrs.get("level")
        for node, attrs in store.graph.nodes(data=True)
    }
    assert levels["Magang AI Engineer"] == 0      # center node: Kueri/Saran
    assert levels["Nusantara Tech"] == 1          # branch 1: Perusahaan
    assert levels["AI Engineer Intern"] == 2      # branch 2: Posisi
    assert levels.get("ML Intern") == 2


def test_graph_unknown_company_attaches_position_to_root(tmp_path: Path):
    from drx.internship.graph import build_internship_store
    from drx.internship.models import InternshipJob, InternshipPlan

    plan = InternshipPlan(approach="role", roles=["AI Engineer"], queries=["AI Engineer"])
    jobs = [
        InternshipJob(
            role_title="AI Engineer Intern",
            company_name="Tidak diketahui",
            location="",
            work_mode="Tidak disebutkan",
            requirements=[],
            deadline="",
            apply_url="https://example.test/9",
            is_valid_internship=True,
        ),
    ]
    store = build_internship_store(tmp_path / "g", plan, jobs)
    levels = {
        store.label_of(node): attrs.get("level")
        for node, attrs in store.graph.nodes(data=True)
    }
    # No placeholder company node at level 1; the position links to the root.
    assert "Tidak diketahui" not in levels
    assert levels.get("AI Engineer Intern") == 1
    predicates = {
        data.get("predicate") for _, _, data in store.graph.edges(data=True)
    }
    assert "saran_posisi" in predicates


def test_engine_diagnostic_statuses_for_blocked_timeout_and_empty(tmp_path: Path):
    class FakeFetcher:
        async def fetch(self, url: str) -> FetchResponse:
            if "glints.com" in url:
                return FetchResponse(
                    url=url,
                    status=403,
                    text="<html>Cloudflare CAPTCHA verify you are human</html>",
                )
            if "kalibrr" in url:
                return FetchResponse(url=url, status=0, error="timeout", error_kind="timeout")
            if "nodeflair.com" in url:
                return FetchResponse(url=url, status=200, text="<html><body></body></html>")
            return await MockFetcher().fetch(url)

    async def scenario():
        return await run_internship(
            _settings(tmp_path),
            approach="discovery",
            no_llm=True,
            fetcher=FakeFetcher(),
            run_id="intern-diagnostics",
            verbose=False,
        )

    result = asyncio.run(scenario())
    statuses: dict[str, str] = {}
    for record in result.diagnostics:
        statuses.setdefault(record.source_name, record.access_status)
    assert statuses["Glints"] == "BLOCKED"
    assert statuses["Kalibrr"] == "TIMEOUT"
    assert statuses["NodeFlair"] == "EMPTY_RESULT"
    assert all(
        record.access_status in {"SUCCESS", "EMPTY_RESULT", "BLOCKED", "TIMEOUT"}
        for record in result.diagnostics
    )
    glints = next(r for r in result.diagnostics if r.source_name == "Glints")
    assert glints.http_status_code == 403
    assert "CAPTCHA" in glints.failure_reason or "Forbidden" in glints.failure_reason
    kalibrr = next(r for r in result.diagnostics if r.source_name == "Kalibrr")
    assert kalibrr.http_status_code == 0
    assert "timeout" in kalibrr.failure_reason.lower()


def test_main_internship_mode_non_interactive_writes_outputs(
    monkeypatch, tmp_path: Path, capsys
):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(f"paths:\n  runs_dir: {tmp_path / 'runs'}\n", encoding="utf-8")
    rc = main.main(
        [
            "--mode",
            "internship",
            "--internship-approach",
            "role",
            "--internship-role",
            "AI Engineer",
            "--mock-search",
            "--no-llm",
            "--no-prompt",
            "--run-id",
            "main-intern",
            "--config",
            str(cfg),
        ]
    )
    assert rc == 0
    run_dir = tmp_path / "runs" / "main-intern"
    assert (run_dir / "internship_source_diagnostics.json").exists()
    assert (run_dir / "report_llm.md").exists()
    assert (run_dir / "graph_interactive.html").exists()
    out = capsys.readouterr().out
    assert "Tabel Ringkasan Diagnostik Sumber" in out
    assert "INTERNSHIP HUNTER — SELESAI" in out


def test_main_internship_no_prompt_role_requires_role(capsys):
    rc = main.main(
        [
            "--mode",
            "internship",
            "--internship-approach",
            "role",
            "--no-prompt",
        ]
    )
    assert rc == 2
    assert "--internship-role" in capsys.readouterr().err


class _TTY(io.StringIO):
    def isatty(self) -> bool:
        return True


def _feed(monkeypatch, answers):
    values = iter(answers)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(values)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)


def test_main_wizard_asks_mode_before_interface_and_runs_internship(
    monkeypatch, tmp_path: Path, capsys
):
    cfg = tmp_path / "settings.yaml"
    cfg.write_text(f"paths:\n  runs_dir: {tmp_path / 'runs'}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", _TTY())
    # [2] Internship Hunter (Sistem Baru) -> [1] CLI -> [A] Spesifik Role ->
    # role -> location -> mode -> (spare) answer.
    _feed(monkeypatch, ["2", "1", "a", "AI Engineer", "", "", ""])

    rc = main.main(["--mock-search", "--no-llm", "--config", str(cfg)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[1] General Investigation (Sistem Lama)" in out
    assert "[2] Internship Hunter (Sistem Baru)" in out
    assert out.index("PILIH MODUS UTAMA") < out.index("Pilih mode eksekusi")
    assert out.index("INTERNSHIP HUNTER ENGINE") < out.index("Tabel Ringkasan Diagnostik Sumber")
    assert "[A] Sub-Opsi A — Spesifik Role" in out
    assert list((tmp_path / "runs").glob("*/report_llm.md"))
