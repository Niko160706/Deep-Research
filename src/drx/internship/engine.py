"""Internship Hunter orchestration engine.

Flow: plan → direct fetch across the A–G sources (bounded concurrency,
Playwright-first with HTTP/BeautifulSoup fallback) → LLM/deterministic
curation → diagnostics JSON + terminal table → Markdown report → Vis.js
mindmap.

This module is intentionally isolated from ``drx.orchestrator``.  Mode 1's
search loop, hierarchy, REST endpoints and report synthesis are not imported
or modified.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from drx.brain.llm_synthesizer import LLMClient
from drx.config import Settings

from .curator import InternshipCurator
from .diagnostic_logger import DiagnosticLogger, SourceDiagnostic
from .graph import build_internship_store, render_internship_graph
from .models import (
    Approach,
    InternshipJob,
    InternshipPlan,
    RawJob,
    SourceContent,
)
from .planner import plan_internship
from .report import build_markdown_report, write_reports
from .site_parsers import (
    HTML_SOURCES,
    LINKEDIN,
    AiohttpFetcher,
    Fetcher,
    MockFetcher,
    PlaywrightFetcher,
    content_from_response,
    fetch_source,
    search_linkedin_dork,
    spec_url,
)

log = logging.getLogger("drx.internship.engine")


def make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-intern-{uuid.uuid4().hex[:6]}"


def _llm_env(settings: Settings) -> dict[str, str]:
    """Environment plus API keys loaded from settings/.env."""
    env = dict(os.environ)
    for var, value in (
        ("GEMINI_API_KEY", settings.gemini_api_key),
        ("OPENROUTER_API_KEY", settings.openrouter_api_key),
        ("GROQ_API_KEY", settings.groq_api_key),
    ):
        if value:
            env.setdefault(var, value)
    return env


@dataclass(frozen=True)
class _SourceTask:
    source_name: str
    source_key: str
    target_url: str
    query: str
    kind: str  # html | linkedin
    requires_playwright: bool = False
    requires_stealth: bool = False


@dataclass
class InternshipRunResult:
    run_id: str
    run_dir: Path
    plan: InternshipPlan
    jobs: list[InternshipJob] = field(default_factory=list)
    diagnostics: list[SourceDiagnostic] = field(default_factory=list)
    report_path: Path | None = None
    diagnostics_path: Path | None = None
    results_path: Path | None = None
    graph_path: Path | None = None
    manifest_path: Path | None = None
    terminal_summary: str = ""

    @property
    def diagnostics_logger(self) -> DiagnosticLogger:
        logger = DiagnosticLogger(self.run_id)
        logger.records.extend(self.diagnostics)
        return logger


class InternshipEngine:
    """Run one internship hunt without touching the general research engine."""

    def __init__(
        self,
        settings: Settings,
        *,
        approach: Approach = "discovery",
        role: str = "",
        skills: str | Sequence[str] | None = None,
        location: str = "Indonesia OR Remote",
        work_mode: str = "",
        max_jobs: int = 30,
        run_id: str | None = None,
        no_llm: bool = False,
        mock: bool = False,
        fetcher: Fetcher | None = None,
        llm_client: LLMClient | None = None,
        verbose: bool = True,
    ) -> None:
        self.settings = settings
        self.approach = approach
        self.role = role
        self.skills = skills or ""
        self.location = location or "Indonesia OR Remote"
        self.work_mode = work_mode
        self.max_jobs = max(1, int(max_jobs or 30))
        self.run_id = run_id or make_run_id()
        self.no_llm = bool(no_llm)
        self.mock = bool(mock)
        self.verbose = verbose
        self._fetcher_injected = fetcher is not None and not self.mock
        self.fetcher = fetcher or (MockFetcher() if mock else AiohttpFetcher(
            timeout_seconds=settings.search.timeout_seconds
        ))
        self._pw: PlaywrightFetcher | None = None
        self._pw_disabled = False
        self.llm_client = None if self.no_llm else llm_client
        self.llm_config = settings.llm.model_copy(
            update={"enabled": False} if self.no_llm else {}
        )
        self.diagnostics = DiagnosticLogger(self.run_id)
        self.run_dir = Path(settings.paths.runs_dir) / self.run_id
        self._semaphore = asyncio.Semaphore(5)
        self._pw_semaphore = asyncio.Semaphore(3)

    # ---- public API -----------------------------------------------------

    async def run(self) -> InternshipRunResult:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Engine instances are occasionally reused by scripts; never leak
        # diagnostics from an earlier run into this report.
        self.diagnostics = DiagnosticLogger(self.run_id)
        self._semaphore = asyncio.Semaphore(5)
        self._pw = None
        self._pw_disabled = False
        plan = await plan_internship(
            self.approach,
            role=self.role,
            skills=self.skills,
            location=self.location,
            work_mode=self.work_mode,
            config=self.llm_config,
            client=self.llm_client,
            env=_llm_env(self.settings),
        )
        if self.verbose:
            print(f"\n[Internship Hunter] Pendekatan: {plan.approach}")
            if plan.used_fallback and plan.error:
                print(f"[Internship Hunter] Planner fallback: {plan.error}")
            print(f"[Internship Hunter] Kueri: {', '.join(plan.queries)}")

        tasks = self._build_tasks(plan)
        try:
            outcomes = await asyncio.gather(
                *(self._fetch_task(task) for task in tasks),
                return_exceptions=True,
            )
        finally:
            await self._close_playwright()

        contents: list[tuple[SourceContent, SourceDiagnostic]] = []
        for task, outcome in zip(tasks, outcomes):
            if isinstance(outcome, BaseException):
                # A bug in one adapter must not kill the other sources.
                record = self.diagnostics.begin(
                    task.source_name,
                    task.target_url,
                    query=task.query,
                )
                self.diagnostics.fail(
                    record,
                    status="BLOCKED",
                    failure_reason=f"Adapter error: {outcome.__class__.__name__}: {outcome}",
                )
                contents.append(
                    (
                        SourceContent(
                            source_name=task.source_name,
                            target_url=task.target_url,
                            query=task.query,
                            access_status="BLOCKED",
                            failure_reason=str(outcome),
                        ),
                        record,
                    )
                )
            else:
                contents.append(outcome)

        curator = InternshipCurator(
            self.llm_config,
            client=self.llm_client,
            env=_llm_env(self.settings),
        )
        all_jobs: list[InternshipJob] = []
        for content, record in contents:
            if content.access_status == "SUCCESS":
                try:
                    curation = await curator.curate(
                        content.raw_jobs,
                        source_name=content.source_name,
                        source_url=content.target_url,
                        query=content.query,
                        raw_text=content.raw_text,
                    )
                except Exception as exc:
                    self.diagnostics.complete(
                        record,
                        status="BLOCKED",
                        http_status_code=content.http_status_code,
                        fetch_mode=content.fetch_mode,
                        failure_reason=(
                            f"Curator error: {exc.__class__.__name__}: {exc}"
                        ),
                        items_extracted_count=0,
                        raw_chars_scraped=content.raw_chars_scraped,
                    )
                    continue
                valid_jobs = _postcurate_filter(content, curation.jobs)
                if valid_jobs:
                    self.diagnostics.complete(
                        record,
                        status="SUCCESS",
                        http_status_code=content.http_status_code,
                        fetch_mode=content.fetch_mode,
                        failure_reason="",
                        items_extracted_count=len(valid_jobs),
                        raw_chars_scraped=content.raw_chars_scraped,
                    )
                    all_jobs.extend(valid_jobs)
                else:
                    if content.raw_jobs:
                        reason = (
                            "Tidak ada lowongan internship yang lolos validasi "
                            "dari kandidat halaman"
                        )
                    else:
                        reason = (
                            "Halaman tidak menghasilkan kandidat/teks yang dapat "
                            "dikurasi menjadi lowongan"
                        )
                    if self.no_llm:
                        reason += " (LLM nonaktif; normalisasi deterministik)"
                    elif curation.error:
                        reason += f" ({curation.error})"
                    if curation.jobs and not valid_jobs:
                        reason += "; kandidat dibuang filter URL post-kurasi (mis. listing agregat LinkedIn)"
                    self.diagnostics.complete(
                        record,
                        status="EMPTY_RESULT",
                        http_status_code=content.http_status_code,
                        fetch_mode=content.fetch_mode,
                        failure_reason=reason,
                        items_extracted_count=0,
                        raw_chars_scraped=content.raw_chars_scraped,
                    )
            else:
                # Fetch-level failure (BLOCKED/TIMEOUT/EMPTY_RESULT).
                self.diagnostics.complete(
                    record,
                    status=content.access_status,
                    http_status_code=content.http_status_code,
                    fetch_mode=content.fetch_mode,
                    failure_reason=content.failure_reason,
                    items_extracted_count=0,
                    raw_chars_scraped=content.raw_chars_scraped,
                )

        jobs = _rank_jobs(_dedupe_jobs(all_jobs), plan)[: self.max_jobs]
        diagnostics_path = self.diagnostics.write(self.run_dir)
        report_markdown = build_markdown_report(
            run_id=self.run_id,
            plan=plan,
            jobs=jobs,
            diagnostics=self.diagnostics.records,
        )
        report_path, _llm_path = write_reports(self.run_dir, report_markdown)
        results_path = self._write_results(
            plan=plan,
            jobs=jobs,
            diagnostics_path=diagnostics_path,
        )

        graph_path: Path | None = None
        try:
            store = build_internship_store(self.run_dir, plan, jobs)
            store.export()
            graph_path = render_internship_graph(
                store,
                self.run_dir / "graph_interactive.html",
                plan=plan,
                settings=self.settings,
                run_id=self.run_id,
                job_count=len(jobs),
            )
        except Exception as exc:  # rendering must never lose the report
            log.warning("internship graph gagal dirender: %s", exc)

        manifest_path = self._write_manifest(
            plan=plan,
            jobs=jobs,
            report_path=report_path,
            diagnostics_path=diagnostics_path,
            results_path=results_path,
            graph_path=graph_path,
        )
        summary = self._build_terminal_summary(
            plan=plan,
            jobs=jobs,
            report_path=report_path,
            diagnostics_path=diagnostics_path,
            results_path=results_path,
            graph_path=graph_path,
        )
        return InternshipRunResult(
            run_id=self.run_id,
            run_dir=self.run_dir,
            plan=plan,
            jobs=jobs,
            diagnostics=list(self.diagnostics.records),
            report_path=report_path,
            diagnostics_path=diagnostics_path,
            results_path=results_path,
            graph_path=graph_path,
            manifest_path=manifest_path,
            terminal_summary=summary,
        )

    # ---- source tasks ---------------------------------------------------

    def _build_tasks(self, plan: InternshipPlan) -> list[_SourceTask]:
        tasks: list[_SourceTask] = []
        queries = plan.queries or ["Software Engineer"]
        for spec in HTML_SOURCES:
            for query in queries:
                tasks.append(
                    _SourceTask(
                        source_name=spec.name,
                        source_key=spec.key,
                        target_url=spec_url(spec, query),
                        query=query,
                        kind="html",
                        requires_playwright=spec.requires_playwright,
                        requires_stealth=spec.requires_stealth,
                    )
                )
        # F: LinkedIn via Tavily/Exa API with the Google dork.
        for query in queries:
            tasks.append(
                _SourceTask(
                    source_name=LINKEDIN.name,
                    source_key=LINKEDIN.key,
                    target_url=spec_url(LINKEDIN, query),
                    query=query,
                    kind="linkedin",
                )
            )

        # Defensive de-duplication (e.g. two identical planner queries).
        seen: set[tuple[str, str]] = set()
        unique: list[_SourceTask] = []
        for task in tasks:
            key = (task.source_name, task.target_url)
            if key in seen:
                continue
            seen.add(key)
            unique.append(task)
        return unique

    def _fetch_mode_for(self, task: _SourceTask) -> str:
        if self.mock:
            return "mock"
        if task.kind == "linkedin":
            return "api"
        if self._pw_disabled or self._fetcher_injected:
            return "aiohttp"
        return "playwright"

    def _ensure_playwright(self) -> PlaywrightFetcher | None:
        """Lazily build the shared Playwright fetcher for a real-network run."""
        if self._pw is not None or self._pw_disabled:
            return self._pw
        try:
            import playwright  # noqa: F401  (availability probe)

            self._pw = PlaywrightFetcher(
                timeout_seconds=self.settings.search.timeout_seconds,
            )
        except Exception as exc:
            log.info("playwright tidak tersedia; fallback HTTP+BS4: %s", exc)
            self._pw_disabled = True
        return self._pw

    async def _close_playwright(self) -> None:
        if self._pw is not None:
            closer, self._pw = self._pw, None
            try:
                await closer.close()
            except Exception:
                pass

    async def _fetch_html_via_playwright(
        self,
        task: _SourceTask,
    ) -> SourceContent | None:
        pw = self._ensure_playwright()
        if pw is None:
            return None
        try:
            async with self._pw_semaphore:
                response = await pw.fetch(
                    task.target_url,
                    stealth=True,  # NodeFlair wajib stealth; uniform hardening
                    wait_ms=3000,
                )
        except Exception as exc:
            return SourceContent(
                source_name=task.source_name,
                target_url=task.target_url,
                query=task.query,
                access_status="BLOCKED",
                failure_reason=f"Playwright error: {exc.__class__.__name__}: {exc}",
            )
        if response.error_kind == "browser":
            # Launch/environment failure: disable Playwright for the rest of
            # the run and fall back to direct HTTP + BeautifulSoup.
            self._pw_disabled = True
            return None
        if response.error_kind in {"timeout", "network"} and not task.requires_playwright:
            # Transient page-load failure on a source that does not require
            # the browser: try direct HTTP + BeautifulSoup once.
            log.info(
                "playwright %s untuk %s tidak berhasil (%s); fallback HTTP",
                task.source_key,
                task.source_name,
                response.error_kind,
            )
            return None
        content = content_from_response(
            response,
            source_name=task.source_name,
            source_key=task.source_key,
            target_url=task.target_url,
            query=task.query,
        )
        if (
            content.access_status == "BLOCKED"
            and "anti-bot" in (content.failure_reason or "").casefold()
        ):
            # One bounded retry inside the SAME browser context: a Cloudflare
            # challenge that ran once usually clears on the second load
            # (shared cookies + warm fingerprint).
            retry = await pw.fetch(task.target_url, stealth=True, wait_ms=3000)
            if retry.error_kind == "" and retry.status and retry.status < 400:
                retry_content = content_from_response(
                    retry,
                    source_name=task.source_name,
                    source_key=task.source_key,
                    target_url=task.target_url,
                    query=task.query,
                )
                if retry_content.access_status != "BLOCKED":
                    content = retry_content
        content.fetch_mode = (
            "playwright-stealth" if task.requires_stealth else "playwright"
        )
        return content

    async def _fetch_linkedin(self, task: _SourceTask) -> SourceContent:
        injected = self._fetcher_injected or self.mock
        if injected:
            # Mock/injected fetchers serve canned API-shaped JSON for the dork
            # URL; the same parser/classifier then handles it.
            return await fetch_source(
                source_name=task.source_name,
                source_key=task.source_key,
                target_url=task.target_url,
                query=task.query,
                fetcher=self.fetcher,
            )
        tavily_key = self.settings.tavily_api_key or ""
        exa_key = self.settings.exa_api_key or ""
        if not tavily_key and not exa_key:
            return SourceContent(
                source_name=task.source_name,
                target_url=task.target_url,
                query=task.query,
                access_status="BLOCKED",
                failure_reason=(
                    "TAVILY_API_KEY atau EXA_API_KEY tidak tersedia "
                    "(dork LinkedIn dieksekusi lewat API)"
                ),
            )
        try:
            raw_jobs, raw_text, provider = await search_linkedin_dork(
                task.query,
                tavily_key=tavily_key,
                exa_key=exa_key,
                timeout_seconds=self.settings.search.timeout_seconds,
            )
        except Exception as exc:
            return SourceContent(
                source_name=task.source_name,
                target_url=task.target_url,
                query=task.query,
                access_status="BLOCKED",
                failure_reason=f"LinkedIn API gagal: {exc.__class__.__name__}: {exc}",
            )
        normalized_jobs: list[RawJob] = []
        for raw in raw_jobs:
            raw.source_name = task.source_name
            raw.source_url = task.target_url
            normalized_jobs.append(raw)
        content = SourceContent(
            source_name=task.source_name,
            target_url=task.target_url,
            query=task.query,
            http_status_code=200,
            fetch_mode=f"linkedin-api:{provider}",
            raw_jobs=normalized_jobs,
            raw_text=raw_text,
        )
        content.raw_chars_scraped = len(raw_text)
        if not content.raw_jobs and not content.raw_text.strip():
            content.access_status = "EMPTY_RESULT"
            content.failure_reason = "API tidak mengembalikan hasil"
        else:
            content.access_status = "SUCCESS"
        return content

    async def _fetch_task(
        self, task: _SourceTask
    ) -> tuple[SourceContent, SourceDiagnostic]:
        record = self.diagnostics.begin(
            task.source_name,
            task.target_url,
            query=task.query,
            fetch_mode=self._fetch_mode_for(task),
        )
        started = time.perf_counter()
        try:
            if task.kind == "linkedin":
                content = await self._fetch_linkedin(task)
            else:
                content = None
                if not self.mock and not self._fetcher_injected:
                    content = await self._fetch_html_via_playwright(task)
                if content is None:
                    content = await fetch_source(
                        source_name=task.source_name,
                        source_key=task.source_key,
                        target_url=task.target_url,
                        query=task.query,
                        fetcher=self.fetcher,
                    )
        except Exception as exc:
            content = SourceContent(
                source_name=task.source_name,
                target_url=task.target_url,
                query=task.query,
                access_status="BLOCKED",
                failure_reason=f"Fetch error: {exc.__class__.__name__}: {exc}",
            )
        latency = content.latency_ms or int((time.perf_counter() - started) * 1000)
        self.diagnostics.complete(
            record,
            status=content.access_status,
            http_status_code=content.http_status_code,
            failure_reason=content.failure_reason,
            latency_ms=latency,
            items_extracted_count=0,
            raw_chars_scraped=content.raw_chars_scraped,
            content_type=content.content_type,
            response_bytes=content.response_bytes,
            notes=content.notes,
            fetch_mode=content.fetch_mode,
        )
        return content, record

    # ---- persistence ----------------------------------------------------

    def _write_results(
        self,
        *,
        plan: InternshipPlan,
        jobs: Sequence[InternshipJob],
        diagnostics_path: Path,
    ) -> Path:
        path = self.run_dir / "internship_results.json"
        payload = {
            "run_id": self.run_id,
            "mode": "internship_hunter",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "approach": plan.approach,
            "plan": plan.to_dict(),
            "total_jobs": len(jobs),
            "jobs": [job.to_dict() for job in jobs],
            "diagnostics_file": diagnostics_path.name,
            "diagnostics_summary": self.diagnostics.summary(),
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _write_manifest(
        self,
        *,
        plan: InternshipPlan,
        jobs: Sequence[InternshipJob],
        report_path: Path | None,
        diagnostics_path: Path | None,
        results_path: Path | None,
        graph_path: Path | None,
    ) -> Path:
        path = self.run_dir / "manifest.json"
        payload = {
            "run_id": self.run_id,
            "mode": "internship_hunter",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "approach": plan.approach,
            "plan": plan.to_dict(),
            "job_count": len(jobs),
            "diagnostics": self.diagnostics.summary(),
            "files": {
                "report": report_path.name if report_path else "",
                "report_llm": "report_llm.md",
                "diagnostics": diagnostics_path.name if diagnostics_path else "",
                "results": results_path.name if results_path else "",
                "graph": graph_path.name if graph_path else "",
            },
        }
        path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def _build_terminal_summary(
        self,
        *,
        plan: InternshipPlan,
        jobs: Sequence[InternshipJob],
        report_path: Path,
        diagnostics_path: Path,
        results_path: Path,
        graph_path: Path | None,
    ) -> str:
        lines = [
            "",
            "=" * 76,
            "  INTERNSHIP HUNTER — SELESAI",
            "=" * 76,
            f"  Pendekatan        : {plan.approach}",
            f"  Kueri             : {', '.join(plan.queries)}",
            f"  Lowongan valid    : {len(jobs)}",
            f"  Laporan           : {report_path}",
            f"  Report LLM        : {self.run_dir / 'report_llm.md'}",
            f"  Hasil JSON        : {results_path}",
            f"  Diagnostik JSON   : {diagnostics_path}",
        ]
        if graph_path is not None:
            lines.append(f"  Mindmap           : {graph_path}")
        lines.append("")
        lines.append("Tabel Ringkasan Diagnostik Sumber")
        lines.append(self.diagnostics.summary_table())
        failures = [
            record
            for record in self.diagnostics.records
            if record.access_status != "SUCCESS"
        ]
        if failures:
            lines.append("")
            lines.append("URL sumber yang perlu diperiksa:")
            for record in failures:
                reason = record.failure_reason or "-"
                lines.append(
                    f"  - {record.source_name} [{record.access_status}]: "
                    f"{record.target_url} ({reason})"
                )
        lines.append("=" * 76)
        return "\n".join(lines)


def _postcurate_filter(
    content: SourceContent, jobs: Sequence[InternshipJob]
) -> list[InternshipJob]:
    """Deterministic sanity checks applied AFTER LLM/normalizer curation.

    Right now this guards the LinkedIn source: the dork snippets sometimes
    convince the curator to keep an aggregate listing page
    (``linkedin.com/jobs/<keyword>-jobs``) as if it were one vacancy — only
    single vacancies (``/jobs/view/") are apply-able internships.
    """
    guard = POSTCURATE_URL_GUARDS.get(content.source_name)
    if not guard:
        return list(jobs)
    return [job for job in jobs if guard(job)]


def _linkedin_job_url_ok(job: InternshipJob) -> bool:
    lowered = (job.apply_url or "").lower()
    return "/jobs/view/" in lowered


POSTCURATE_URL_GUARDS: dict[str, object] = {
    "LinkedIn Jobs": _linkedin_job_url_ok,
}


def _dedupe_jobs(jobs: Sequence[InternshipJob]) -> list[InternshipJob]:
    seen: set[str] = set()
    out: list[InternshipJob] = []
    for job in jobs:
        key = (
            f"{job.role_title.casefold()}|{job.company_name.casefold()}|"
            f"{job.apply_url.casefold()}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


_RANK_STOPWORDS = {
    "intern",
    "interns",
    "internship",
    "magang",
    "remote",
    "hybrid",
    "onsite",
    "on-site",
    "indonesia",
    "engineer",
    "engineering",
    "software",
    "teknologi",
    "tech",
    "dan",
    "or",
    "and",
    "the",
    "for",
    "with",
    "di",
    "ke",
}


def _rank_jobs(
    jobs: Sequence[InternshipJob], plan: InternshipPlan
) -> list[InternshipJob]:
    """Put roles matching the user's role/skill query at the top of the cap.

    Score is deterministic and only changes display ordering, never validity.
    """
    terms: list[str] = []
    for value in [*plan.roles, plan.user_input, *plan.queries]:
        for token in re.split(r"[^a-zA-Z0-9+#.]+", str(value or "").casefold()):
            token = token.strip(".")
            if (
                len(token) >= 2
                and token not in _RANK_STOPWORDS
                and token not in terms
            ):
                terms.append(token)
    if not terms:
        return list(jobs)

    patterns = [
        re.compile(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])")
        for term in terms
    ]

    def score(job: InternshipJob) -> tuple[int, int]:
        role_text = job.role_title.casefold()
        full_text = " ".join(
            [
                job.role_title,
                job.company_name,
                job.location,
                job.work_mode,
                *job.requirements,
            ]
        ).casefold()
        role_hits = sum(1 for pattern in patterns if pattern.search(role_text))
        full_hits = sum(1 for pattern in patterns if pattern.search(full_text))
        return (role_hits * 4 + full_hits, len(role_text))

    return sorted(jobs, key=score, reverse=True)


async def run_internship(
    settings: Settings,
    *,
    approach: Approach = "discovery",
    role: str = "",
    skills: str | Sequence[str] | None = None,
    location: str = "Indonesia OR Remote",
    work_mode: str = "",
    max_jobs: int = 30,
    run_id: str | None = None,
    no_llm: bool = False,
    mock: bool = False,
    fetcher: Fetcher | None = None,
    llm_client: LLMClient | None = None,
    verbose: bool = True,
) -> InternshipRunResult:
    """Convenience wrapper for scripts and tests."""
    engine = InternshipEngine(
        settings,
        approach=approach,
        role=role,
        skills=skills,
        location=location,
        work_mode=work_mode,
        max_jobs=max_jobs,
        run_id=run_id,
        no_llm=no_llm,
        mock=mock,
        fetcher=fetcher,
        llm_client=llm_client,
        verbose=verbose,
    )
    return await engine.run()
