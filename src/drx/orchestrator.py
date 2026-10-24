"""Orchestrator — the hybrid BFS/DFS research loop.

Phase 1 wires the full mechanical pipeline (muscle + memory + parquet) for
real, while the cognitive stages (router, super-agent, extractor, critics,
synthesis) return deterministic mocks so the loop is runnable end-to-end
without LLM keys.

Phase 2 adds :meth:`Orchestrator.synthesize_llm` — called by ``main.py`` after
:meth:`run` finishes — which hands the validated KG facts plus their supporting
evidence chunks to the LLM report synthesis engine
(``drx.brain.llm_synthesizer``) and writes ``runs/<run_id>/report_llm.md``.
The research loop itself stays 100% LLM-free.

Key invariants implemented here (from the design review):
  * ``task.depth`` is read from the popped task — never a global counter.
  * ``frontier.pop()`` drains the LIFO stack before the FIFO queue.
  * Empty search results / empty clean text are marked failed, not crashed.
  * Novelty (Python) is computed BEFORE any qualitative LLM check.
  * Consecutive low-novelty cycles trigger an early stop in the non-interactive
    loop; interactive mode only warns and waits for an explicit Stop.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from drx.brain import critic as llm_critic
from drx.brain import extractors as triple_extractors
from drx.brain import llm_synthesizer, query_planner
from drx.brain.batching import estimate_cost_usd
from drx.brain.llm_synthesizer import LLMResult, ReportSidecar
from drx.config import Settings
from drx.critic.entity_critic import EntityCritic
from drx.critic.evaluator import Evaluator
from drx.critic.heuristic_critic import HeuristicCritic
from drx.extract.entity_path import EntityExtractor
from drx.extract.heuristic_path import HeuristicExtractor
from drx.gate.decomposer import render_tree, split_context
from drx.gate.lexical_survey import LexicalSurvey
from drx.gate.router import Router
from drx.gate.super_agent import SuperAgent
from drx.memory.budget import Budget
from drx.memory.frontier import HybridFrontier
from drx.models import (
    Document,
    Fact,
    Layer,
    Mode,
    QueryTask,
    Report,
    SourceRef,
    TaskOrigin,
)
from drx.muscle.autosearch import AutoSearch
from drx.muscle.backends import build_backend, canonicalize_url
from drx.muscle.chunker import Chunker
from drx.muscle.cleaner import Cleaner
from drx.muscle.scraper import Scraper
from drx.store.graph_store import GraphStore
from drx.store.parquet_store import ParquetStore
from drx.store.vector_store import VectorStore
from drx.synthesis.packer import ContextPacker
from drx.synthesis.writer import SynthesisWriter
from drx.textutil import primary_entity, sanitize_query, split_aliases
from drx.visualizer import render_graph_html

log = logging.getLogger("drx.orchestrator")


class Orchestrator:
    """Coordinates the entire Dual-Stream pipeline for a single run."""

    def __init__(
        self,
        settings: Settings,
        run_id: str | None = None,
        context: str = "",
        verbose: bool = False,
    ) -> None:
        self.settings = settings
        self.context = context
        self.verbose = verbose
        self.run_id = run_id or self._make_run_id()
        self.run_dir = Path(settings.paths.runs_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.initial_tasks: list[QueryTask] = []
        self.main_query = ""
        self.entity = ""
        self.aliases: list[str] = []
        # Lazy decomposition: stop starting new sub-queries once this many
        # documents have survived cleaning (quota saver).
        self.lazy_min_docs = settings.search.lazy_min_docs
        self._valid_docs = 0
        # Global URL dedup cache: ONE set shared by the anchored (70%) and
        # exploratory (30%) search paths, so a URL fetched on one path is never
        # fetched again on the other.
        self.seen_urls: set[str] = set()

        bl = settings.budget_limits
        self.budget = Budget(bl.max_tokens, bl.max_runtime_seconds, bl.max_urls)

        # ---- muscle (real, 0 tokens) ----
        search_backend = build_backend(
            settings.search.backend,
            settings.search.timeout_seconds,
            settings.search.region,
        )
        self.autosearch = AutoSearch(
            backend=search_backend,
            results_per_query=settings.search.results_per_query,
            seen_urls=self.seen_urls,
        )
        self.scraper = Scraper(
            timeout=settings.search.timeout_seconds,
            max_playwright=bl.playwright_max_tabs,
            tor_proxy=settings.tor_proxy,
            min_content_chars=settings.scraping.min_text_chars,
            high_value_min_chars=settings.scraping.high_value_min_chars,
        )
        self.cleaner = Cleaner(
            min_text_length=settings.cleaning.min_text_length,
            bm25_keep_top_k=settings.cleaning.bm25_keep_top_k,
            target_lang=settings.cleaning.target_lang,
            require_context=settings.cleaning.require_context,
            asn_noise_threshold=settings.cleaning.asn_noise_threshold,
        )
        self.chunker = Chunker(
            settings.chunking.chunk_size, settings.chunking.chunk_overlap
        )
        self.parquet = ParquetStore(self.run_dir)

        # ---- memory ----
        self.frontier = HybridFrontier(max_depth=bl.max_depth)

        # ---- cognitive (mock in Phase 1) ----
        self.router = Router()
        self.super_agent = SuperAgent(context=context, backend=settings.search.backend)
        survey_search = AutoSearch(
            backend=build_backend(
                settings.search.backend,
                settings.search.timeout_seconds,
                settings.search.region,
            ),
            results_per_query=settings.search.results_per_query,
            # The survey is a pre-flight probe (nothing is scraped), so it uses
            # its own cache: its URLs must stay fetchable by the main loop.
        )
        self.survey = LexicalSurvey(survey_search)
        self.entity_extractor = EntityExtractor()
        self.heuristic_extractor = HeuristicExtractor(self.chunker)
        self.graph_store = GraphStore(self.run_dir)
        self.vector_store = VectorStore(self.run_dir)
        self.entity_critic = EntityCritic()
        self.heuristic_critic = HeuristicCritic()
        self.evaluator = Evaluator(
            settings.novelty.floor, settings.novelty.stagnation_limit
        )
        self.packer = ContextPacker()
        self.writer = SynthesisWriter(self.run_id)

        # ---- Phase 2: LLM report synthesis (dijalankan main.py setelah run) ----
        self.llm_result: LLMResult | None = None
        # ---- Phase 2b: agentic critic loop (fact verifier sebelum sintesis) ----
        self.critic_result: llm_critic.CriticResult | None = None
        self.critic_rounds = 0  # ronde gap-query yang benar-benar dieksekusi
        self.critic_queries: list[str] = []  # sub-kueri gap yang berhasil dijalankan
        self.critic_attempts = 0  # jumlah panggilan LLM critic
        self.critic_skipped = ""
        self.critic_error = ""
        self.critic_suggestions: list[dict] = []  # saran pivot node-ID untuk checkpoint

        # Mode Web App: render kanvas dengan REST client + command_source async.
        self.web_api = False

        # ---- Phase 3: production flow (planner, grounded triples, checkpoint) ----
        self.planner_result: query_planner.QueryPlanResult | None = None
        self.extraction_skipped = ""
        self.extraction_error = ""
        self.extraction_batches = 0
        self.triples_extracted = 0
        self.tokens_in = 0  # akumulasi token LLM (input) untuk checkpoint
        self.tokens_out = 0
        self.tokens_total = 0
        self.cycle = 0
        self.interactive = False
        self.final_report_generated = False
        self._synthesis_done = False
        self._expanded_urls: set[str] = set()
        self._pivot_history: set[str] = set()
        self.pivot_round = 0  # jumlah ronde pivot yang sukses dijadwalkan
        self._doc_levels: dict[str, int] = {}  # url kanonik -> level kanvas
        self._last_render_path: Path | None = None
        # Interactive budget override: a budget key is removed from the stop
        # check only after the user explicitly chooses to continue over it.
        self._budget_overrides: set[str] = set()
        # Expand is a real search operation (not just re-reading parquet):
        # track every query already executed and every expansion attempt so
        # repeated clicks only schedule genuinely new queries.
        self._executed_queries: set[str] = set()
        self._expand_history: set[str] = set()
        self._expand_round = 0

    @staticmethod
    def _make_run_id() -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:6]}"

    async def run(self, main_query: str) -> Report:
        """Pipeline non-interaktif (backward compatible dengan Phase 1/2)."""
        await self._setup(main_query)
        await self._research_loop()
        # ---- Phase 2b: agentic critic loop (LLM fact verifier) ----
        await self._agentic_critic_loop()
        return await self._finalize()

    async def _setup(self, main_query: str) -> None:
        # ---- query sanitization: "Bayu Skak (Bayu Eko Moektito)" ----
        # -> '"Bayu Skak" OR "Bayu Eko Moektito"' (brackets confuse backends).
        raw_query = (main_query or "").strip()
        query = sanitize_query(raw_query) or raw_query
        entity = primary_entity(raw_query) or query
        aliases = split_aliases(raw_query)
        self.main_query = query
        self.entity = entity
        self.aliases = aliases
        # Canvas anchor: pin the root query node to level 0 (left-most column
        # of the hierarchical mindmap) before any fact is stored.
        self.graph_store.ensure_root(entity)
        log.info("run %s started for: %s", self.run_id, query)
        if len(aliases) > 1:
            log.info(
                "query sanitized: %r -> %r (primary entity=%r, aliases=%s)",
                raw_query,
                query,
                entity,
                aliases,
            )

        # ---- gate: plan + survey + decompose ----
        mode = await self.router.classify(query)
        plan = await self.super_agent.def1(query, mode)
        survey = await self.survey.survey(plan)
        tasks = await self.super_agent.def2(plan, survey, mode)
        self.initial_tasks = tasks
        if self.verbose:
            print(render_tree(query, tasks, aliases=aliases))
        for task in tasks:
            if not self.frontier.push(task, layer=Layer.QUEUE):
                task.status = "skipped"

        # ---- strict --context policy (global, announced once) ----
        # The rule is enforced inside Cleaner.clean(), the single chokepoint
        # between scraping and parquet/graph/vector storage. Every sub-query path
        # goes through it, so anchored 70% and exploratory 30% get the very same
        # requirement; this log line makes that guarantee auditable per run.
        context_terms = split_context(self.context)
        if self.cleaner.require_context and context_terms:
            log.info(
                "strict --context enforcement: ON for ALL paths (anchored 70%% + "
                "exploratory 30%%) -> setiap dokumen wajib memuat salah satu dari %s",
                context_terms,
            )
        elif self.cleaner.require_context:
            log.info(
                "strict --context enforcement: ON tanpa konteks -> tanpa syarat "
                "kata kunci, tapi profil sosmed (facebook/instagram/tiktok) "
                "slug==nama dibuang"
            )
        else:
            log.info(
                "strict --context enforcement: OFF (--loose-context) -> konteks "
                "hanya bonus skor, nama pasaran bisa lolos"
            )

        # ---- P0: seeded query planner (satu panggilan LLM kecil, ~400 token) ----
        await self._run_query_planner(tasks, mode)

    async def _run_query_planner(self, tasks: list[QueryTask], mode: Mode) -> None:
        """P0 — variasi sub-kueri berbenih dari satu panggilan LLM kecil.

        Best-effort: tanpa API key / error -> tidak ada task tambahan, keputusan
        deterministik Def1/Def2 tetap berjalan. Task ditandai anchored bila
        memuat salah satu kata kunci ``--context`` (aturan 70/30 yang sama).
        """
        config = self.settings.llm
        if not config.enabled or not config.planner_enabled:
            return
        try:
            plan = await query_planner.plan_queries(
                self.entity or self.main_query,
                self.context,
                config=config,
                env=self._llm_env(),
                language=self.settings.synthesis.language,
            )
        except Exception as exc:  # planner tidak boleh menjatuhkan run
            log.warning("query planner crashed: %s", exc)
            return
        self.planner_result = plan
        self._record_usage(
            plan.input_tokens_est,
            max(0, plan.total_tokens - plan.input_tokens_est),
            plan.total_tokens,
        )
        if not plan.queries or plan.skipped_reason:
            log.info(
                "query planner: tidak ada variasi baru (%s)",
                plan.skipped_reason or "kosong",
            )
            return
        context_terms = split_context(self.context)
        existing = {t.query.lower() for t in tasks}
        added = 0
        for query in plan.queries:
            if query.lower() in existing:
                continue
            anchored = bool(context_terms) and any(
                term.lower() in query.lower() for term in context_terms
            )
            task = QueryTask(
                id=str(uuid.uuid4()),
                query=query,
                mode=mode,
                depth=1,
                origin=TaskOrigin.INITIAL,
                rationale="P0 query planner (variasi berbenih)",
                backend=self.settings.search.backend,
                is_anchored=anchored,
                context=context_terms if anchored else [],
            )
            if not self.frontier.push(task, layer=Layer.QUEUE):
                task.status = "skipped"
                continue
            self.initial_tasks.append(task)
            existing.add(query.lower())
            added += 1
        log.info(
            "query planner: +%d sub-kueri berbenih (fallback=%s, %d token)",
            added,
            plan.used_fallback,
            plan.total_tokens,
        )

    async def _research_loop(self) -> None:
        # 70/30 guard: no exploratory task in the plan -> stop is unrestricted.
        exploratory_probe_done = not any(
            not t.is_anchored for t in self.initial_tasks
        )

        # ---- main research loop ----
        while True:
            reason = self._stop_reason()
            if reason:
                log.info("%s -> stop", reason)
                break
            kind, task = self._pop_task()
            if kind == "lazy_stop":
                break
            if task is None:
                continue

            # 70/30 strategy guard: the 30% exploratory probe must get a turn
            # before an early "self-answer sufficient" stop can end the run.
            if task.origin is TaskOrigin.INITIAL and not task.is_anchored:
                exploratory_probe_done = True

            log.info(
                "[depth=%d | mode=%s | %s] %s",
                task.depth,
                task.mode.value,
                "anchored" if task.is_anchored else "exploratory",
                task.query,
            )

            status = await self._execute_cycle(task)
            if status == "check_sufficient":
                if not exploratory_probe_done:
                    log.info(
                        "self-answer sufficient but deferred: "
                        "30% exploratory probe has not run yet"
                    )
                elif await self.evaluator.self_answer_sufficient(self.graph_store):
                    log.info("self-answer sufficient -> stop")
                    break

    def _stop_reason(self) -> str | None:
        """Quantitative stop checks, cheapest first (shared by both loops)."""
        if not self.frontier.has_work():
            return "frontier exhausted"
        if self.budget.tokens_exhausted():
            return f"token budget exhausted ({self.budget.tokens})"
        if self.budget.clock_exhausted():
            return f"wall-clock exhausted ({self.budget.elapsed:.1f}s)"
        if self.budget.urls_exhausted():
            return f"URL budget exhausted ({self.budget.urls})"
        if self.evaluator.stagnated():
            return f"novelty stagnated {self.evaluator.zero_streak} cycles"
        return None

    def _pop_task(self, *, interactive: bool = False) -> tuple[str, QueryTask | None]:
        """Pop the next runnable task (depth guard + lazy decomposition).

        Returns ``("ok", task)``, ``("lazy_stop", None)`` (non-interactive stop),
        or ``("empty", None)``. In interactive mode lazy decomposition skips the
        remaining initial sub-queries but keeps pivot/expand cycles usable.
        """
        while True:
            if not self.frontier.has_work():
                return "empty", None
            task = self.frontier.pop()
            if task is None:
                continue
            if task.depth > self.settings.budget_limits.max_depth:
                self.frontier.mark_exhausted(task)
                task.status = "skipped"
                continue

            # ---- lazy sub-query decomposition (quota guard) ----
            # Once enough valid documents exist, the remaining initial
            # sub-queries are never executed (no more search/scrape quota spent).
            if (
                self.lazy_min_docs > 0
                and self._valid_docs >= self.lazy_min_docs
                and task.origin is TaskOrigin.INITIAL
            ):
                task.status = "skipped"
                skipped = 1 + self._skip_pending_initial_tasks()
                log.info(
                    "lazy decomposition: %d valid docs >= %d -> stop searching "
                    "(%d sub-query dilewati)",
                    self._valid_docs,
                    self.lazy_min_docs,
                    skipped,
                )
                if interactive:
                    continue
                return "lazy_stop", None
            return "ok", task

    async def _execute_cycle(self, task: QueryTask) -> str:
        """One search -> scrape -> clean -> store -> extract cycle (steps 2-6).

        Returns ``"continue"`` or ``"check_sufficient"`` (the caller decides
        whether the LLM self-answer check may stop the run).
        """
        # 2. muscle (0 tokens)
        self._executed_queries.add(self._norm_query(task.query))
        hits = await self.autosearch.search(
            task.query, limit=task.search_limit or None
        )
        if not hits:
            log.info("no search results -> mark_failed, continue")
            task.status = "empty"
            self.frontier.mark_failed(task, "empty_results")
            return "continue"
        task.score = round(max(r.score for r in hits), 3)
        task.result_count = len(hits)
        task.status = "searched"
        log.info("search: %d unique URLs", len(hits))

        docs = await self.scraper.fetch(hits)
        self.budget.add_url(len(hits))
        docs = self._dedupe_docs(docs)
        raw_chars = sum(len(d.text) for d in docs)
        log.info("scraped: %d/%d docs, %d raw chars", len(docs), len(hits), raw_chars)

        # Cleaning is the single chokepoint: only docs that survive it reach
        # parquet, graph, vector store and the final report — and strict
        # --context is applied inside it for every path (anchored +
        # exploratory), so nothing can bypass the requirement downstream.
        clean = self.cleaner.clean(
            docs,
            query=task.query,
            context=self.context,
            entity_query=self._entity_query(task),
        )
        clean_chars = sum(len(d.text) for d in clean)
        if not clean:
            log.info("nothing usable after cleaning -> mark_failed, continue")
            task.status = "failed"
            self.frontier.mark_failed(task, "empty_clean")
            return "continue"

        self._valid_docs += len(clean)
        retention = (100.0 * clean_chars / raw_chars) if raw_chars else 0.0
        log.info(
            "cleaned: %d docs, %d chars (retention %.1f%%) | valid docs so far: %d",
            len(clean),
            clean_chars,
            retention,
            self._valid_docs,
        )

        self.parquet.save(clean)
        task_level = max(1, int(getattr(task, "level", 1) or 1))
        for doc in clean:
            self._doc_levels[canonicalize_url(doc.url)] = task_level
            if task.origin is TaskOrigin.EXPAND:
                self._expanded_urls.add(canonicalize_url(doc.url))

        # 3. bifurcation by task mode
        new_nodes = 0
        if task.mode in (Mode.ENTITY, Mode.HYBRID):
            facts = await self.entity_extractor.extract(clean)
            new_nodes += self.graph_store.resolve_and_add(facts, level=task_level)
        if task.mode in (Mode.HEURISTIC, Mode.HYBRID):
            cards = self.heuristic_extractor.make_cards(clean)
            await self.heuristic_extractor.embed(cards)
            self.vector_store.add(cards)
            new_nodes += len(cards)
        # 3b. Phase 3: grounded LLM triples (batch <= 3.2k token, anti-halusinasi)
        new_nodes += await self._extract_grounded_triples(clean, level=task_level)
        # Incremental persistent graph: aman dipanggil setiap siklus.
        self.graph_store.export()

        processed = max(1, len(clean))

        # 4. critics (only when there is new material)
        gaps = []
        if new_nodes > 0:
            gaps.extend(await self.entity_critic.check(self.graph_store))
            gaps.extend(await self.heuristic_critic.check(self.vector_store))
        for gap in gaps:
            child = QueryTask(
                id=str(uuid.uuid4()),
                parent_id=task.id,
                query=gap.suggested_query,
                mode=gap.mode,
                depth=task.depth + 1,
                origin=TaskOrigin.CRITIC,
            )
            self.frontier.push(child, layer=Layer.STACK)

        # 5. quantitative evaluation (0 tokens)
        novelty = self.evaluator.novelty_rate(new_nodes, processed)
        if novelty <= self.settings.novelty.floor:
            self.evaluator.record_zero()
            log.info("low novelty (%.3f) -> demote, continue", novelty)
            return "continue"
        self.evaluator.reset_zero()

        # 6. qualitative evaluation (LLM, guarded) — the caller may defer/stop
        if self.evaluator.should_self_answer(new_nodes):
            return "check_sufficient"
        return "continue"

    async def _extract_grounded_triples(
        self, docs: list[Document], *, level: int = 1
    ) -> int:
        """LLM triple extraction dengan grounding wajib (source_id + quote).

        Best-effort: tanpa key / error provider -> pipa mekanis tetap jalan;
        kegagalan pertama dicatat dan tidak diulang-ulang (hemat token).
        """
        config = self.settings.llm
        if not config.enabled or not config.extraction_enabled:
            return 0
        if self.extraction_error:
            return 0
        try:
            result = await triple_extractors.extract_triples(
                docs,
                self.entity or self.main_query,
                self.context,
                config=config,
                env=self._llm_env(),
                language=self.settings.synthesis.language,
            )
        except Exception as exc:  # never break the loop
            self.extraction_error = f"{type(exc).__name__}: {exc}"
            log.warning("grounded extraction crashed: %s", exc)
            return 0
        self.extraction_batches += result.batches
        self._record_usage(
            result.input_tokens_est,
            max(0, result.total_tokens - result.input_tokens_est),
            result.total_tokens,
        )
        if result.skipped_reason:
            if not self.extraction_skipped:
                self.extraction_skipped = result.skipped_reason
                log.info("ekstraksi triple dilewati: %s", result.skipped_reason)
            return 0
        if result.error:
            self.extraction_error = result.error
            log.warning("ekstraksi triple error: %s", result.error)
            return 0
        if not result.triples:
            return 0
        facts = triple_extractors.triples_to_facts(result.triples)
        new_nodes = self.graph_store.resolve_and_add(facts, level=level)
        self.triples_extracted += len(result.triples)
        log.info(
            "ekstraksi triple: +%d grounded (ditolak %d, %d batch) -> +%d node",
            len(result.triples),
            result.rejected,
            result.batches,
            new_nodes,
        )
        return new_nodes

    async def _finalize(self) -> Report:
        """Prune + export graph, laporan ekstraktif Phase 1, lalu manifest."""
        query = self.main_query
        self.graph_store.prune_louvain(self.entity)
        self.graph_store.export()
        if self.verbose:
            print(render_tree(query, self.initial_tasks, aliases=self.aliases))
        context = self.packer.build(
            query, self.graph_store, self.vector_store, self.parquet
        )
        report = await self.writer.synthesize(query, context)
        self._render_graph(self.critic_suggestions)
        self._write_manifest()
        log.info("run %s complete -> %s", self.run_id, self.run_dir)
        return report

    # ---- Phase 3: token accounting + interactive checkpoint ----------------

    def _record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        """Akumulasi token LLM untuk checkpoint + budget (tanpa double count)."""
        tokens_in = max(0, int(input_tokens or 0))
        tokens_out = max(0, int(output_tokens or 0))
        total = int(total_tokens or 0) or (tokens_in + tokens_out)
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.tokens_total += total
        if total:
            self.budget.add_tokens(total)

    def estimated_cost_usd(self) -> float:
        """Estimasi biaya dari harga per 1M token di settings (0 = free tier)."""
        config = self.settings.llm
        return estimate_cost_usd(
            self.tokens_in,
            self.tokens_out,
            price_input_per_1m=config.cost_input_per_1m,
            price_output_per_1m=config.cost_output_per_1m,
        )

    def _render_graph(self, suggestions: list[dict] | None = None) -> Path | None:
        """Render ``graph_interactive.html`` (PRD_design.md) — best-effort."""
        config = self.settings.visualizer
        if not config.enabled:
            return None
        self._last_render_path = render_graph_html(
            self.graph_store,
            self.run_dir / "graph_interactive.html",
            target_entity=self.entity or self.main_query,
            context=self.context,
            critic_suggestions=suggestions or self.critic_suggestions,
            cycle=self.cycle,
            tokens=self.tokens_total,
            cost_usd=self.estimated_cost_usd(),
            max_nodes=config.max_nodes,
            cdn_resources=config.cdn_resources,
            run_id=self.run_id,
            web_api=self.web_api,
            budget=self._budget_status(),
            pivoted_nodes=sorted(self._pivot_history),
        )
        return self._last_render_path

    async def run_interactive(
        self,
        main_query: str,
        *,
        input_fn: object | None = None,
        output_fn: object | None = None,
        command_source: object | None = None,
        web_api: bool = False,
    ) -> Report:
        """Checkpoint setelah setiap siklus pencarian (terminal atau Web App).

        Setiap siklus: cari -> scrape -> bersihkan -> triple grounding -> graph
        -> render ``graph_interactive.html`` -> tampilkan token/biaya + saran ID
        node dari critic -> user memilih ``[1]`` pivot, ``[2]`` expand, ``[3]``
        selesai, atau ``[4]`` melihat semua node (khusus CLI). Laporan naratif
        LLM ditulis **sekali** saat selesai.

        Mode interaktif tidak berhenti otomatis saat frontier kosong atau
        novelty stagnan: checkpoint tetap terbuka sampai user memilih Stop.
        Batas token/URL/waktu tetap dicek; bila terlampaui, user diminta
        konfirmasi (CLI) atau mode web lanjut dengan peringatan.

        ``command_source`` (Mode Web App) adalah callable async yang menunggu
        perintah ``{"action": "pivot"|"expand"|"stop", "node_ids": [...]}`` dari
        server lokal; bila None, checkpoint memakai ``input_fn`` (terminal).
        ``web_api=True`` merender tombol kanvas sebagai REST client.
        """
        if input_fn is None:
            input_fn = input
        if output_fn is None:
            output_fn = print
        self.interactive = True
        self.web_api = bool(web_api)
        await self._setup(main_query)

        while True:
            reason = self._interactive_stop_reason()
            if reason:
                key, message = reason
                output_fn(f"\n[checkpoint] {message}.")
                if not self._confirm_budget_continue(
                    key, input_fn, output_fn, command_source
                ):
                    break
                self._budget_overrides.add(key)
                output_fn(
                    f"  -> lanjut melebihi batas {key}; pilih [3] Stop kapan saja."
                )
                continue

            kind, task = self._pop_task(interactive=True)
            if kind == "empty":
                # Frontier kosong bukan akhir di mode interaktif: user masih
                # boleh pivot node lama atau Expand untuk mencari gelombang
                # dokumen level-1 baru.
                suggestions = await self._checkpoint_suggestions()
                self._render_graph(suggestions)
                command = await self._checkpoint_command(
                    input_fn, output_fn, suggestions, command_source
                )
                if await self._apply_interactive_command(
                    command, input_fn, output_fn
                ):
                    break
                continue
            if task is None:
                continue
            log.info(
                "[depth=%d | mode=%s | %s] %s",
                task.depth,
                task.mode.value,
                "anchored" if task.is_anchored else "exploratory",
                task.query,
            )
            await self._execute_cycle(task)
            self.cycle += 1

            suggestions = await self._checkpoint_suggestions()
            self._render_graph(suggestions)
            command = await self._checkpoint_command(
                input_fn, output_fn, suggestions, command_source
            )
            if await self._apply_interactive_command(command, input_fn, output_fn):
                break

        report = await self._finalize()
        if self.settings.llm.enabled:
            # Satu-satunya tulisan laporan naratif LLM (idempoten bila main.py
            # memanggil synthesize_llm lagi setelah ini).
            await self.synthesize_llm()
        self.final_report_generated = True
        return report

    def _interactive_stop_reason(self) -> tuple[str, str] | None:
        """Batas keras yang masih berlaku di mode interaktif.

        Frontier kosong dan novelty stagnan TIDAK menghentikan mode interaktif;
        user yang memutuskan lewat ``[3] Stop``. Budget yang sudah di-override
        user juga tidak dicek lagi.
        """
        if "token" not in self._budget_overrides and self.budget.tokens_exhausted():
            return (
                "token",
                (
                    f"token budget exhausted ({self.budget.tokens} / "
                    f"{self.budget.max_tokens})"
                ),
            )
        if "clock" not in self._budget_overrides and self.budget.clock_exhausted():
            return (
                "clock",
                (
                    f"wall-clock exhausted ({self.budget.elapsed:.1f}s / "
                    f"{self.budget.max_runtime_seconds}s)"
                ),
            )
        if "urls" not in self._budget_overrides and self.budget.urls_exhausted():
            return (
                "urls",
                (
                    f"URL budget exhausted ({self.budget.urls} / "
                    f"{self.budget.max_urls})"
                ),
            )
        return None

    def _confirm_budget_continue(
        self,
        key: str,
        input_fn,
        output_fn,
        command_source: object | None,
    ) -> bool:
        """Minta izin user sebelum melewati batas budget yang keras."""
        if command_source is not None:
            log.warning(
                "budget %s terlampaui -> mode web lanjut otomatis "
                "(user dapat menekan Stop)",
                key,
            )
            output_fn(
                "  [web] budget terlampaui; run lanjut otomatis. "
                "Tekan Stop di browser untuk mengakhiri."
            )
            return True
        answer = self._ask_yes_no(
            input_fn,
            output_fn,
            f"Lanjut melebihi batas {key}? [y/n]: ",
        )
        return bool(answer)

    async def _apply_interactive_command(
        self,
        command: dict,
        input_fn,
        output_fn,
    ) -> bool:
        """Jalankan satu perintah checkpoint; True berarti finalisasi."""
        action = command.get("action")
        if action == "stop":
            return True
        if action == "pivot":
            node_ids = [
                str(node_id).strip()
                for node_id in (command.get("node_ids") or [])
                if str(node_id).strip()
            ]
            if not node_ids and not self.web_api:
                node_ids = self._ask_node_ids(input_fn, output_fn) or []
            if not node_ids:
                output_fn("  -> tidak ada node pivot yang dikirim.")
                return False
            created = self._make_pivot_tasks(node_ids)
            if created:
                output_fn(
                    f"  -> {created} kueri pivot dijadwalkan "
                    "(level anak = level parent + 1)."
                )
            else:
                output_fn(
                    "  -> 0 kueri pivot dijadwalkan "
                    "(node sudah pernah dipivot / tidak valid)."
                )
            return False
        if action == "expand":
            queries_run, new_nodes = await self._expand_documents()
            if queries_run and new_nodes:
                output_fn(
                    f"  -> Expand: {queries_run} kueri level-1 dijalankan "
                    f"(+{new_nodes} node baru; dokumen baru masuk parquet)."
                )
            elif queries_run:
                output_fn(
                    f"  -> Expand: {queries_run} kueri level-1 dijalankan, "
                    "tetapi tidak ada node baru (hasil sudah pernah terlihat). "
                    "Coba pivot node lain atau Stop."
                )
            else:
                output_fn(
                    "  -> Expand: tidak ada kueri baru yang bisa dijadwalkan; "
                    "coba pivot node lain atau Stop."
                )
        return False

    async def _checkpoint_suggestions(self) -> list[dict]:
        """Saran node-ID untuk pivot: critic JSON-triple, fallback deterministik."""
        config = self.settings.llm
        triples = self.graph_store.triples(limit=config.critic_max_triples)
        if (
            config.enabled
            and config.critic_enabled
            and config.critic_reads_triples
            and triples
        ):
            verdict = await self._run_triple_critic(triples)
            if verdict is not None and verdict.suggested_nodes:
                filtered = [
                    node.model_dump()
                    for node in verdict.suggested_nodes
                    if str(node.node_id) not in self._pivot_history
                    and self.graph_store.graph is not None
                    and node.node_id in self.graph_store.graph
                ]
                if filtered:
                    self.critic_suggestions = filtered
                    return self.critic_suggestions
        self.critic_suggestions = self.graph_store.suggest_nodes(
            self.entity or self.main_query,
            self.context,
            limit=5,
            exclude=self._pivot_history,
        )
        return self.critic_suggestions

    async def _run_triple_critic(
        self, triples: list[dict]
    ) -> llm_critic.CriticResult | None:
        """Critic HANYA membaca JSON triple (bukan teks mentah parquet)."""
        self.critic_attempts += 1
        try:
            verdict = await llm_critic.analyze_graph_gaps(
                triples,
                self.entity or self.main_query,
                self.context,
                config=self.settings.llm,
                env=self._llm_env(),
                language=self.settings.synthesis.language,
            )
        except Exception as exc:  # graceful: critic tidak boleh menjatuhkan run
            self.critic_error = f"{type(exc).__name__}: {exc}"
            log.warning("critic(triple) crashed: %s", exc)
            return None

        self.critic_result = verdict
        if verdict.skipped_reason:
            self.critic_skipped = verdict.skipped_reason
            log.info("critic(triple) skipped: %s", verdict.skipped_reason)
            return None
        if verdict.error:
            self.critic_error = verdict.error
            log.warning("critic(triple) error: %s", verdict.error)
            return None
        self._record_usage(
            verdict.input_tokens_est,
            max(0, verdict.total_tokens - verdict.input_tokens_est),
            verdict.total_tokens,
        )
        log.info(
            "critic(triple): sufficient=%s (%d batch, %d node, %d token)",
            verdict.is_sufficient,
            verdict.batches,
            verdict.nodes_used,
            verdict.total_tokens,
        )
        return verdict

    def _budget_status(self) -> dict:
        """Budget snapshot + interactive overrides for terminal/Web App UI."""
        data = self.budget.status()
        data["overrides"] = sorted(self._budget_overrides)
        data["interactive"] = bool(self.interactive)
        return data

    @staticmethod
    def _fmt_int(value: float) -> str:
        return f"{int(value):,}".replace(",", ".")

    def _print_checkpoint(
        self,
        output_fn,
        suggestions: list[dict],
        *,
        show_all: bool = True,
    ) -> None:
        verified = sum(
            1 for edge in self.graph_store.edge_records() if edge.get("verified")
        )
        cost = self.estimated_cost_usd()
        free_tier = (
            self.settings.llm.cost_input_per_1m == 0
            and self.settings.llm.cost_output_per_1m == 0
        )
        budget = self._budget_status()
        tokens = int(budget.get("tokens") or 0)
        max_tokens = int(budget.get("max_tokens") or 0)
        tokens_remaining = int(budget.get("tokens_remaining") or 0)
        urls = int(budget.get("urls") or 0)
        max_urls = int(budget.get("max_urls") or 0)
        urls_remaining = int(budget.get("urls_remaining") or 0)
        elapsed = float(budget.get("elapsed_seconds") or 0.0)
        max_seconds = float(budget.get("max_runtime_seconds") or 0.0)
        seconds_remaining = float(budget.get("seconds_remaining") or 0.0)
        token_over = max_tokens > 0 and tokens >= max_tokens
        url_over = max_urls > 0 and urls >= max_urls
        time_over = max_seconds > 0 and elapsed >= max_seconds
        lines = [
            "",
            "=" * 64,
            f"CHECKPOINT SIKLUS {self.cycle} - {self.entity or self.main_query}",
            "=" * 64,
            f"Parquet dokumen : {self.parquet.count()}",
            (
                f"Graph           : {self.graph_store.node_count()} node / "
                f"{self.graph_store.edge_count()} edge "
                f"({verified} terverifikasi)"
            ),
            (
                f"Token kumulatif : {self._fmt_int(self.tokens_in)} input + "
                f"{self._fmt_int(self.tokens_out)} output = "
                f"{self._fmt_int(self.tokens_total)} token"
            ),
            (
                f"  Sisa token    : {self._fmt_int(tokens_remaining)} / "
                f"{self._fmt_int(max_tokens)}"
                + ("  (OVER BUDGET)" if token_over else "")
            ),
            (
                f"URL             : {self._fmt_int(urls)} / "
                f"{self._fmt_int(max_urls)} "
                f"(sisa {self._fmt_int(urls_remaining)})"
                + ("  (OVER BUDGET)" if url_over else "")
            ),
            (
                f"Waktu           : {elapsed:.1f}s / {max_seconds:.0f}s "
                f"(sisa {seconds_remaining:.1f}s)"
                + ("  (OVER BUDGET)" if time_over else "")
            ),
            f"Estimasi biaya  : ${cost:.6f}"
            + (" (free tier / harga belum diatur)" if free_tier else ""),
        ]
        if self._budget_overrides:
            lines.append(
                "Budget override : " + ", ".join(sorted(self._budget_overrides))
            )
        if self.evaluator.stagnated():
            lines.append(
                f"Novelty         : stagnan {self.evaluator.zero_streak} siklus "
                "(mode interaktif tetap lanjut)"
            )
        lines.append("Saran node (critic):")
        if suggestions:
            for item in suggestions:
                node_id = str(item.get("node_id") or "")
                level = ""
                graph = self.graph_store.graph
                if graph is not None and node_id in graph:
                    level = f" (level {int(graph.nodes[node_id].get('level') or 0)})"
                lines.append(
                    f"  [{node_id}] {item.get('label')}{level} "
                    f"- {item.get('reason')}"
                )
        else:
            lines.append("  (belum ada saran)")
        lines.extend(
            [
                "",
                "Opsi:",
                "  [1] Pivot pada ID node (contoh: n12,n7; bisa banyak, pisah koma)",
                (
                    "  [2] Expand: cari gelombang dokumen level-1 baru "
                    "(deep search + sudut baru)"
                ),
                "  [3] Selesai & susun laporan final (sekali)",
            ]
        )
        if show_all:
            lines.append(
                "  [4] Lihat semua node per level (tidak mengubah apa pun)"
            )
        output_fn("\n".join(lines))

    def _print_all_nodes(self, output_fn) -> None:
        """Cetak SEMUA node (bukan top-5) dikelompokkan per level."""
        records = self.graph_store.node_records()
        if not records:
            output_fn("  (graf masih kosong)")
            return
        records.sort(
            key=lambda item: (
                int(item.get("level") or 0),
                -int(item.get("degree") or 0),
                str(item.get("label") or "").casefold(),
            )
        )
        lines = [
            "",
            f"SEMUA NODE ({len(records)}) - urut level, lalu derajat:",
        ]
        current_level: int | None = None
        for item in records:
            level = int(item.get("level") or 0)
            if level != current_level:
                lines.append(f"Level {level}:")
                current_level = level
            marks = []
            if item.get("root"):
                marks.append("root")
            if str(item.get("id")) in self._pivot_history:
                marks.append("sudah dipivot")
            suffix = f"  [{', '.join(marks)}]" if marks else ""
            lines.append(
                f"  [{item.get('id')}] {item.get('label')} "
                f"(derajat {item.get('degree')}){suffix}"
            )
        output_fn("\n".join(lines))

    async def _checkpoint_command(
        self,
        input_fn,
        output_fn,
        suggestions: list[dict],
        command_source: object | None = None,
    ) -> dict:
        """Satu keputusan checkpoint: ``{action: pivot|expand|stop, node_ids}``.

        Mode Web App memakai ``command_source`` (callable async dari server);
        Mode CLI memakai menu ``1/2/3/4`` + konfirmasi sebelum aksi dieksekusi.
        """
        if command_source is not None:
            self._print_checkpoint(output_fn, suggestions, show_all=False)
            output_fn(
                "[web] menunggu aksi dari browser: Pivot / Expand / Stop "
                "(server lokal)."
            )
            try:
                command = await command_source()
            except Exception as exc:
                log.warning("command source gagal: %s -> selesai otomatis", exc)
                return {"action": "stop"}
            if not isinstance(command, dict):
                log.warning("perintah web tidak valid: %r -> selesai", command)
                return {"action": "stop"}
            action = str(command.get("action") or "").strip().lower()
            if action not in {"pivot", "expand", "stop"}:
                log.warning("aksi web tidak dikenal: %r -> selesai", action)
                return {"action": "stop"}
            return {"action": action, "node_ids": command.get("node_ids") or []}

        while True:
            choice = self._checkpoint_choice(
                input_fn, output_fn, suggestions, show_all=True
            )
            if choice == "4":
                self._print_all_nodes(output_fn)
                continue
            if choice == "1":
                while True:
                    node_ids = self._ask_node_ids(input_fn, output_fn)
                    if node_ids is None:
                        return {"action": "stop"}
                    if not node_ids:
                        break  # batal ke menu utama
                    confirmed = self._confirm_pivot_ids(
                        node_ids, input_fn, output_fn
                    )
                    if confirmed is None:
                        return {"action": "stop"}
                    if confirmed:
                        return {"action": "pivot", "node_ids": node_ids}
                    # `n`: langsung kembali ke pemilihan node, bukan menu.
                continue
            if choice == "2":
                confirmed = self._ask_yes_no(
                    input_fn,
                    output_fn,
                    "Expand akan mencari gelombang dokumen level-1 baru. "
                    "Lanjut? [y/n]: ",
                )
                if confirmed is None:
                    return {"action": "stop"}
                if confirmed:
                    return {"action": "expand"}
                continue
            confirmed = self._ask_yes_no(
                input_fn,
                output_fn,
                "Selesai dan susun laporan final? [y/n]: ",
            )
            if confirmed is None or confirmed:
                return {"action": "stop"}

    def _checkpoint_choice(
        self,
        input_fn,
        output_fn,
        suggestions: list[dict],
        *,
        show_all: bool = True,
    ) -> str:
        self._print_checkpoint(output_fn, suggestions, show_all=show_all)
        valid = {"1", "2", "3", "4"} if show_all else {"1", "2", "3"}
        prompt = "Pilihan [1/2/3/4]: " if show_all else "Pilihan [1/2/3]: "
        while True:
            try:
                raw = input_fn(prompt)
            except (EOFError, KeyboardInterrupt):
                output_fn("\n[checkpoint] input tidak tersedia -> selesai otomatis.")
                return "3"
            choice = (raw or "").strip()
            if choice in valid:
                return choice
            output_fn(
                "  Pilihan tidak dikenal; masukkan "
                + ("1, 2, 3, atau 4." if show_all else "1, 2, atau 3.")
            )

    def _ask_yes_no(self, input_fn, output_fn, prompt: str) -> bool | None:
        """Konfirmasi y/n; ``None`` bila input tidak tersedia (EOF/CTRL-C)."""
        while True:
            try:
                raw = input_fn(prompt)
            except (EOFError, KeyboardInterrupt):
                output_fn("\n[checkpoint] input tidak tersedia -> selesai otomatis.")
                return None
            answer = (raw or "").strip().lower()
            if answer in {"y", "ya", "yes"}:
                return True
            if answer in {"n", "tidak", "no"}:
                return False
            output_fn("  Jawab y (ya) atau n (tidak).")

    def _confirm_pivot_ids(
        self,
        node_ids: list[str],
        input_fn,
        output_fn,
    ) -> bool | None:
        """Tampilkan daftar pivot tervalidasi sebelum satu batch dikirim."""
        graph = self.graph_store.graph
        lines = ["Pivot yang dipilih:"]
        already = 0
        for node_id in node_ids:
            label = self.graph_store.label_of(node_id)
            level = 0
            if graph is not None and node_id in graph:
                level = int(graph.nodes[node_id].get("level") or 0)
            mark = ""
            if node_id in self._pivot_history:
                mark = "  [sudah pernah dipivot]"
                already += 1
            lines.append(f"  [{node_id}] {label} (level {level}){mark}")
        lines.append(f"Total: {len(node_ids)} node")
        if already == len(node_ids):
            lines.append(
                "Catatan: semua node ini sudah pernah dipivot; "
                "tidak ada kueri baru yang akan dibuat."
            )
        output_fn("\n".join(lines))
        return self._ask_yes_no(input_fn, output_fn, "Kirim pivot ini? [y/n]: ")

    def _ask_node_ids(self, input_fn, output_fn) -> list[str] | None:
        output_fn("Masukkan ID node untuk pivot (pisah koma, contoh: n12,n7):")
        while True:
            try:
                raw = input_fn("ID node: ")
            except (EOFError, KeyboardInterrupt):
                return None
            parts: list[str] = []
            for part in (raw or "").replace(";", ",").split(","):
                node_id = part.strip()
                if node_id and node_id not in parts:
                    parts.append(node_id)
            if not parts:
                return []
            graph = self.graph_store.graph
            invalid = [
                node_id
                for node_id in parts
                if graph is None or node_id not in graph
            ]
            if invalid:
                output_fn(
                    "  ID tidak ditemukan: "
                    + ", ".join(invalid)
                    + " -> perbaiki dulu (tidak ada yang dikirim)."
                )
                continue
            return parts

    def _make_pivot_tasks(self, node_ids: list[str]) -> int:
        """Pivot by node ID: query = label + konteks + label tetangga.

        Level anak mengikuti level parent (``parent.level + 1``), bukan nomor
        ronde global. Jadi pivot dari level 2 menghasilkan level 3, pivot dari
        level 1 menghasilkan level 2, dst. — tanpa batas level.

        Neighbor context is appended so a common name (e.g. "Sri Lestari")
        cannot be contaminated by a different person with the same name.
        """
        created = 0
        graph = self.graph_store.graph
        max_depth = self.settings.budget_limits.max_depth
        depth = min(2, max_depth) if max_depth >= 1 else 0
        for node_id in node_ids:
            if graph is None or node_id not in graph:
                continue
            if node_id in self._pivot_history:
                log.info("pivot %s sudah pernah dijalankan -> skip", node_id)
                continue
            label = self.graph_store.label_of(node_id)
            neighbors = self.graph_store.neighbors(node_id, limit=4)
            parts = [f'"{label}"']
            if self.context.strip():
                parts.append(self.context.strip())
            for item in neighbors[:2]:
                parts.append(item["label"])
            query = " ".join(part for part in parts if part)
            parent_level = int(graph.nodes[node_id].get("level") or 0)
            child_level = max(1, parent_level + 1)
            task = QueryTask(
                id=str(uuid.uuid4()),
                parent_id=node_id,
                query=query,
                mode=Mode.ENTITY,
                depth=depth,
                origin=TaskOrigin.CRITIC,
                rationale=f"pivot node {node_id} ({label}) + konteks tetangga",
                backend=self.settings.search.backend,
                is_anchored=True,
                context=split_context(self.context),
                level=child_level,
            )
            if not self.frontier.push(task, layer=Layer.STACK):
                log.info(
                    "pivot %s -> query duplikat/ditolak guard; tidak dijadwalkan",
                    node_id,
                )
                self._pivot_history.add(node_id)
                continue
            self._pivot_history.add(node_id)
            created += 1
            log.info(
                "pivot %s (level %d) -> level %d: %r",
                node_id,
                parent_level,
                child_level,
                query,
            )
        if created:
            self.pivot_round += 1
        return created

    @staticmethod
    def _norm_query(query: str) -> str:
        return " ".join(
            (query or "").strip().strip('"').casefold().split()
        )

    def _expand_angle_queries(self) -> list[str]:
        """Sudut pencarian level-1 yang belum pernah dijalankan."""
        base = (self.entity or self.main_query or "").strip().strip('"')
        if not base:
            return []
        context = " ".join(
            part.strip() for part in split_context(self.context)[:1]
            if part.strip()
        )
        angles = [
            "profil",
            "karier",
            "berita",
            "wawancara",
            "organisasi",
            "pendidikan",
            "penghargaan",
            "kegiatan",
        ]
        offset = self._expand_round % len(angles)
        rotated = angles[offset:] + angles[:offset]
        out: list[str] = []
        for angle in rotated:
            query = f'"{base}" {angle}'
            if context:
                query += f" {context}"
            if self._norm_query(query) in self._executed_queries:
                continue
            out.append(query)
        return out

    def _related_level_one_queries(self, limit: int = 3) -> list[str]:
        """Fallback expand seeds dari label node level-1 yang paling terhubung."""
        base = (self.entity or self.main_query or "").strip().strip('"')
        if not base:
            return []
        records = [
            item
            for item in self.graph_store.node_records()
            if int(item.get("level") or 0) == 1 and not item.get("root")
        ]
        records.sort(key=lambda item: -int(item.get("degree") or 0))
        out: list[str] = []
        for item in records[:limit]:
            label = str(item.get("label") or "").strip()
            if not label or len(label) > 100:
                continue
            query = f'"{base}" "{label}"'
            if self._norm_query(query) in self._executed_queries:
                continue
            out.append(query)
        return out

    async def _expand_documents(self, batch_size: int = 3) -> tuple[int, int]:
        """Cari gelombang dokumen level-1 baru, lalu ekstrak ke graph.

        Expand bukan lagi sekadar membaca ulang parquet: ia menjalankan
        deep-search kueri awal (limit diperbesar untuk mengambil hasil di luar
        top-N pertama) dan sudut kueri baru. Semua fakta baru disimpan pada
        level 1. Mengembalikan ``(jumlah_kueri, jumlah_node_baru)``.
        """
        self._expand_round += 1
        max_queries = max(1, int(batch_size))
        candidates: list[tuple[str, int, str]] = []
        seen_candidates: set[str] = set()

        def add_candidate(query: str, limit: int, reason: str) -> bool:
            key = self._norm_query(query)
            if not key or key in seen_candidates:
                return False
            if len(candidates) >= max_queries:
                return False
            seen_candidates.add(key)
            candidates.append((query, max(1, int(limit)), reason))
            return True

        # 1) Deep-search kueri awal: limit naik tiap ronde expand, sementara
        # global URL cache membuang halaman yang sudah pernah di-scrape.
        base_queries: list[str] = [self.main_query]
        base_queries.extend(task.query for task in self.initial_tasks)
        if base_queries:
            offset = (self._expand_round - 1) % len(base_queries)
            base_queries = base_queries[offset:] + base_queries[:offset]
        deep_limit = min(
            60,
            max(10, int(self.settings.search.results_per_query))
            * (self._expand_round + 1),
        )
        deep_slots = max(1, max_queries - 1)
        for query in base_queries:
            if len(candidates) >= deep_slots:
                break
            if not query.strip():
                continue
            key = f"deep:{self._norm_query(query)}:{deep_limit}"
            if key in self._expand_history:
                continue
            if add_candidate(query, deep_limit, f"deep search limit {deep_limit}"):
                self._expand_history.add(key)

        # 2) Sudut baru (profil/karier/berita/...) untuk menambah cabang level 1.
        for query in self._expand_angle_queries():
            if len(candidates) >= max_queries:
                break
            key = f"angle:{self._norm_query(query)}"
            if key in self._expand_history:
                continue
            if add_candidate(
                query,
                self.settings.search.results_per_query,
                "sudut kueri baru",
            ):
                self._expand_history.add(key)

        # 3) Fallback: telusuri nama node level-1 yang sudah ada.
        for query in self._related_level_one_queries():
            if len(candidates) >= max_queries:
                break
            key = f"related:{self._norm_query(query)}"
            if key in self._expand_history:
                continue
            if add_candidate(
                query,
                self.settings.search.results_per_query,
                "relasi node level-1",
            ):
                self._expand_history.add(key)

        queries_run = 0
        node_delta = 0
        for query, limit, reason in candidates:
            if (
                self.budget.tokens_exhausted()
                or self.budget.urls_exhausted()
                or self.budget.clock_exhausted()
            ):
                log.info("expand: budget habis -> hentikan gelombang ini")
                break
            before_nodes = self.graph_store.node_count()
            task = QueryTask(
                id=str(uuid.uuid4()),
                query=query,
                mode=Mode.ENTITY,
                depth=1,
                origin=TaskOrigin.EXPAND,
                rationale=f"expand level 1 ({reason})",
                backend=self.settings.search.backend,
                is_anchored=bool(split_context(self.context)),
                context=split_context(self.context),
                level=1,
                search_limit=limit,
            )
            log.info("expand level 1: %r (limit %d)", query, limit)
            await self._execute_cycle(task)
            self.cycle += 1
            queries_run += 1
            node_delta += max(0, self.graph_store.node_count() - before_nodes)

        self.graph_store.export()
        log.info(
            "expand: %d kueri dijalankan -> +%d node baru (parquet=%d, level-1)",
            queries_run,
            node_delta,
            self.parquet.count(),
        )
        return queries_run, node_delta

    # ---- Phase 2b: agentic critic loop --------------------------------------

    async def _agentic_critic_loop(self) -> None:
        """Fact-verifier loop sebelum sintesis (PRD: autonomous research loop).

        Alur: critic menilai kecukupan bukti -> bila belum cukup dan kuota loop
        belum habis, sub-kueri gap dijalankan dan critic dipanggil ulang.
        Berhenti saat bukti cukup, loop habis, budget habis, atau critic tidak
        tersedia (tanpa API key / error) — tidak pernah menjatuhkan run.
        """
        config = self.settings.llm
        if not config.enabled or not config.critic_enabled:
            log.info(
                "agentic critic: nonaktif (llm.enabled=%s, critic_enabled=%s)",
                config.enabled,
                config.critic_enabled,
            )
            return

        for iteration in range(config.critic_max_loops + 1):
            verdict = await self._run_critic()
            if verdict is None:
                return
            if verdict.is_sufficient:
                log.info(
                    "critic: bukti dinilai CUKUP -> lanjut sintesis (%s)",
                    verdict.critique_summary or "tanpa catatan",
                )
                return
            if iteration >= config.critic_max_loops:
                log.info(
                    "critic: loop habis (max_loops=%d) -> lanjut sintesis",
                    config.critic_max_loops,
                )
                return
            log.info(
                "Critic menemukan gap data: %s. Mengeksekusi sub-kueri lanjutan...",
                (verdict.critique_summary or "(tanpa ringkasan)").rstrip(". "),
            )
            executed = await self._execute_gap_queries(verdict.suggested_sub_queries)
            if executed == 0:
                log.info(
                    "critic: tidak ada sub-kueri gap yang berhasil -> lanjut sintesis"
                )
                return
            self.critic_rounds += 1

    async def _run_critic(self) -> llm_critic.CriticResult | None:
        """Fact verifier: triple JSON (default) atau parquet legacy (opt-out)."""
        config = self.settings.llm
        if config.critic_reads_triples:
            triples = self.graph_store.triples(limit=config.critic_max_triples)
            if triples:
                return await self._run_triple_critic(triples)
        return await self._run_legacy_critic()

    async def _run_legacy_critic(self) -> llm_critic.CriticResult | None:
        """Legacy parquet-segment critic (backward compat ``critic_reads_triples=false``)."""
        self.critic_attempts += 1
        try:
            verdict = await llm_critic.analyze_fact_gaps(
                self.parquet.path,
                self.entity or self.main_query,
                self.context,
                config=self.settings.llm,
                env=self._llm_env(),
                language=self.settings.synthesis.language,
            )
        except Exception as exc:  # graceful: critic tidak boleh menjatuhkan run
            self.critic_error = f"{type(exc).__name__}: {exc}"
            log.warning("critic crashed: %s", exc)
            return None

        self.critic_result = verdict
        if verdict.skipped_reason:
            self.critic_skipped = verdict.skipped_reason
            log.info("critic skipped: %s", verdict.skipped_reason)
            return None
        if verdict.error:
            self.critic_error = verdict.error
            log.warning("critic error: %s", verdict.error)
            return None
        self._record_usage(
            verdict.input_tokens_est,
            max(0, verdict.total_tokens - verdict.input_tokens_est),
            verdict.total_tokens,
        )
        log.info(
            "critic: sufficient=%s (segmen=%d dari %d dokumen, %d token)",
            verdict.is_sufficient,
            verdict.segments_used,
            verdict.documents_used,
            verdict.total_tokens,
        )
        return verdict

    async def _execute_gap_queries(self, queries: list[str]) -> int:
        """Jalankan sub-kueri gap: search -> scrape -> clean -> parquet.

        Semua dokumen tetap melewati ``Cleaner.clean()`` (chokepoint tunggal),
        jadi strict --context dan gate identitas berlaku sama seperti Phase 1.
        Returns jumlah sub-kueri yang benar-benar menghasilkan dokumen baru.
        """
        limit = max(0, self.settings.llm.critic_max_sub_queries)
        executed = 0
        for raw in queries[:limit]:
            query = " ".join((raw or "").split())
            if not query:
                continue
            if (
                self.budget.tokens_exhausted()
                or self.budget.clock_exhausted()
                or self.budget.urls_exhausted()
            ):
                log.info("critic gap: budget habis -> stop sub-kueri")
                break
            log.info("[critic gap | query] %s", query)
            hits = await self.autosearch.search(query)
            if not hits:
                log.info("critic gap: no search results -> skip")
                continue
            docs = await self.scraper.fetch(hits)
            self.budget.add_url(len(hits))
            docs = self._dedupe_docs(docs)
            clean = self.cleaner.clean(
                docs,
                query=query,
                context=self.context,
                entity_query=self.main_query or self.entity,
            )
            if not clean:
                log.info("critic gap: nothing usable after cleaning -> skip")
                continue
            self._valid_docs += len(clean)
            self.parquet.save(clean)
            executed += 1
            self.critic_queries.append(query)
            log.info(
                "critic gap: +%d dokumen lolos gate (total parquet=%d)",
                len(clean),
                self.parquet.count(),
            )
        return executed

    def _add_verified_facts_to_graph(self, sidecar: ReportSidecar) -> int:
        """Masukkan ``verified_facts`` sidecar ke NetworkX + export GraphML.

        Fakta ini sudah diverifikasi LLM DAN di-clamp ke URL bukti di
        ``brain.llm_synthesizer.build_sidecar``, jadi aman masuk graph final
        (``runs/<run_id>/graph.graphml``).
        """
        if not sidecar.verified_facts:
            return 0
        facts = [
            Fact(
                subject=item.subject,
                predicate=item.predicate,
                object=item.object,
                confidence_score=0.9,
                provenance=[SourceRef(url=item.source_url)],
            )
            for item in sidecar.verified_facts
        ]
        added = self.graph_store.resolve_and_add(facts)
        self.graph_store.export()
        log.info(
            "verified facts: %d triple -> graph (+%d node baru) -> graph.graphml",
            len(facts),
            added,
        )
        return added

    async def synthesize_llm(self) -> LLMResult | None:
        """Phase 2 — grounded LLM report synthesis from validated KG facts.

        Dipanggil ``main.py`` setelah :meth:`run` selesai. Mengambil **seluruh**
        fakta KG yang sudah tervalidasi (``triples(limit=0)``) sebagai
        satu-satunya basis fakta; parquet hanya dipakai untuk mencari chunk /
        kutipan yang mendukung fakta tersebut. Segmen dibatasi
        ``settings.llm.max_segments`` / ~``max_input_tokens``, lalu laporan
        investigasi final diminta ke provider LLM (Groq gpt-oss-120b → Groq
        qwen3.8-27b → Gemini) dan disimpan ke ``runs/<run_id>/report_llm.md``.

        Tidak pernah melempar exception: kegagalan apa pun kembali sebagai
        :class:`LLMResult` dengan ``skipped_reason``/``error``; ``None``
        berarti sintesis memang dinonaktifkan (``settings.llm.enabled=false``).
        """
        config = self.settings.llm
        if not config.enabled:
            log.info("LLM synthesis: disabled (settings.llm.enabled=false)")
            return None
        if self._synthesis_done and self.llm_result is not None:
            log.info("LLM synthesis: sudah dijalankan -> memakai hasil yang ada")
            return self.llm_result

        try:
            evidence_facts = self.graph_store.triples(limit=0)
            result = await llm_synthesizer.synthesize_report(
                self.parquet.path,
                self.entity or self.main_query,
                self.context,
                config=config,
                env=self._llm_env(),
                language=self.settings.synthesis.language,
                evidence_facts=evidence_facts,
            )
        except Exception as exc:
            log.warning("LLM synthesis crashed: %s", exc)
            result = LLMResult(error=f"{type(exc).__name__}: {exc}")

        self.llm_result = result
        if result.skipped_reason:
            log.info("LLM synthesis skipped: %s", result.skipped_reason)
        elif result.error:
            log.warning("LLM synthesis error: %s", result.error)
        elif result.ok:
            path = llm_synthesizer.write_llm_report(self.run_dir, result)
            sidecar_path = None
            graph_added = 0
            facts_used = 0
            if result.sidecar is not None:
                facts_used = len(result.sidecar.verified_facts)
                sidecar_path = llm_synthesizer.write_report_sidecar(
                    self.run_dir, result.sidecar
                )
                graph_added = self._add_verified_facts_to_graph(result.sidecar)
            self._record_usage(
                result.input_tokens_est,
                max(0, result.total_tokens - result.input_tokens_est),
                result.total_tokens,
            )
            self._synthesis_done = True
            self._render_graph(self.critic_suggestions)
            log.info(
                "LLM synthesis: %s/%s -> %s + %s (%d segmen dari %d dokumen, "
                "±%d token input, %d token total, %d fakta -> graph +%d node)",
                result.provider,
                result.model,
                path,
                sidecar_path or "(tanpa sidecar)",
                result.segments_used,
                result.documents_used,
                result.input_tokens_est,
                result.total_tokens,
                facts_used,
                graph_added,
            )
        self._write_manifest()
        return result

    def _llm_env(self) -> dict[str, str]:
        """``os.environ`` + API keys loaded into Settings from ``.env``."""
        env = dict(os.environ)
        for var, value in (
            ("GEMINI_API_KEY", self.settings.gemini_api_key),
            ("OPENROUTER_API_KEY", self.settings.openrouter_api_key),
            ("GROQ_API_KEY", self.settings.groq_api_key),
        ):
            if value:
                env.setdefault(var, value)
        return env

    def _entity_query(self, task: QueryTask) -> str:
        """Entity used by the Fellegi-Sunter gate (with every alias).

        The target identity never changes per sub-query, so the sanitized main
        query (``"A" OR "B"``) is used instead of the task hypothesis.
        """
        _ = task
        return self.main_query or self.entity

    def _skip_pending_initial_tasks(self) -> int:
        """Mark every not-yet-run initial sub-query as skipped (lazy stop)."""
        skipped = 0
        for task in self.initial_tasks:
            if task.status == "pending" and task.origin is TaskOrigin.INITIAL:
                task.status = "skipped"
                skipped += 1
        return skipped

    @staticmethod
    def _dedupe_docs(docs: list[Document]) -> list[Document]:
        """Global URL dedup cache: drop repeated pages (mobile/AMP variants)."""
        seen: set[str] = set()
        out: list[Document] = []
        for doc in docs:
            key = canonicalize_url(doc.url)
            if key in seen:
                continue
            seen.add(key)
            out.append(doc)
        if len(out) != len(docs):
            log.info(
                "global dedup cache: %d duplicate document(s) dropped",
                len(docs) - len(out),
            )
        return out

    def _write_manifest(self) -> None:
        manifest = {
            "run_id": self.run_id,
            "budget": self.budget.snapshot(),
            "frontier_size": len(self.frontier),
            "parquet_documents": self.parquet.count(),
            "graph_nodes": self.graph_store.node_count(),
            "graph_edges": self.graph_store.edge_count(),
            "vector_cards": self.vector_store.card_count(),
        }
        manifest["token_usage"] = {
            "input": self.tokens_in,
            "output": self.tokens_out,
            "total": self.tokens_total,
            "budget_tokens": self.budget.tokens,
        }
        manifest["estimated_cost_usd"] = round(self.estimated_cost_usd(), 6)
        manifest["interactive"] = bool(self.interactive)
        if self.planner_result is not None:
            manifest["query_planner"] = {
                "queries": self.planner_result.queries,
                "used_fallback": self.planner_result.used_fallback,
                "provider": self.planner_result.provider,
                "model": self.planner_result.model,
                "input_tokens_est": self.planner_result.input_tokens_est,
                "total_tokens": self.planner_result.total_tokens,
                "skipped_reason": self.planner_result.skipped_reason or None,
                "error": self.planner_result.error or None,
            }
        if (
            self.extraction_batches
            or self.triples_extracted
            or self.extraction_skipped
            or self.extraction_error
        ):
            manifest["triple_extraction"] = {
                "batches": self.extraction_batches,
                "triples": self.triples_extracted,
                "skipped_reason": self.extraction_skipped or None,
                "error": self.extraction_error or None,
            }
        if self._last_render_path is not None:
            manifest["visualizer"] = {"file": str(self._last_render_path)}
        if self.llm_result is not None:
            manifest["llm_synthesis"] = {
                "provider": self.llm_result.provider,
                "model": self.llm_result.model,
                "segments_used": self.llm_result.segments_used,
                "documents_used": self.llm_result.documents_used,
                "input_tokens_est": self.llm_result.input_tokens_est,
                "total_tokens": self.llm_result.total_tokens,
                "sections_complete": self.llm_result.sections_complete,
                "sidecar_complete": self.llm_result.sidecar_complete,
                "sidecar_verified_facts": (
                    len(self.llm_result.sidecar.verified_facts)
                    if self.llm_result.sidecar is not None
                    else 0
                ),
                "skipped_reason": self.llm_result.skipped_reason or None,
                "error": self.llm_result.error or None,
            }
        if self.critic_attempts or self.critic_skipped or self.critic_error:
            manifest["critic"] = {
                "is_sufficient": (
                    self.critic_result.is_sufficient
                    if self.critic_result is not None and self.critic_result.ok
                    else None
                ),
                "attempts": self.critic_attempts,
                "gap_rounds": self.critic_rounds,
                "gap_queries": self.critic_queries,
                "suggested_nodes": self.critic_suggestions,
                "critique_summary": (
                    self.critic_result.critique_summary
                    if self.critic_result is not None
                    else ""
                ),
                "skipped_reason": self.critic_skipped or None,
                "error": self.critic_error or None,
            }
        path = self.run_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
