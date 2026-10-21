"""Tests for the Phase 3 interactive checkpoint (pivot / expand / finish).

Fully offline: mock search backend + stub scraper, LLM disabled, injected
``input_fn``/``output_fn`` capture the terminal checkpoint instead of a TTY.
"""
from __future__ import annotations

import json

import pytest

from drx.config import Settings
from drx.models import Document, Fact, SearchResult, SourceRef
from drx.orchestrator import Orchestrator


class _StubBackend:
    """Deterministic backend whose results always mention the query terms."""

    name = "stub"

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        clean = query.strip('" ')
        slug = "-".join(clean.lower().split())[:40]
        rows = [
            SearchResult(
                url=f"https://contoh.id/{slug}-{index}",
                title=f"{clean} profil dan karier {index}",
                snippet=(
                    f"Profil {clean}: perawat di Dinas Kesehatan Sleman, "
                    "koordinator imunisasi."
                ),
                domain="contoh.id",
            )
            for index in range(3)
        ]
        return rows[:limit]


class _StubScraper:
    async def fetch(self, results):
        body = (
            "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan Sleman. "
            "Ia mengoordinasikan program imunisasi wilayah Sleman. "
        )
        return [
            Document(url=result.url, title=result.title, text=body * 8)
            for result in results
        ]


def _make_orchestrator(tmp_path) -> Orchestrator:
    settings = Settings.load(tmp_path / "missing.yaml")
    settings.paths.runs_dir = str(tmp_path)
    settings.cleaning.min_text_length = 50
    settings.llm.enabled = False  # tidak ada panggilan jaringan
    settings.search.lazy_min_docs = 0
    orch = Orchestrator(settings, run_id="interactive", context="", verbose=False)
    stub = _StubBackend()
    orch.autosearch.backend = stub
    orch.survey.autosearch.backend = stub
    orch.scraper = _StubScraper()  # type: ignore[assignment]
    return orch


def _answers(*values: str):
    iterator = iter(values)

    def feed(_prompt: str = "") -> str:
        try:
            return next(iterator)
        except StopIteration:
            # No more scripted answers: behave like a closed stdin so the
            # checkpoint stops instead of looping forever on an invalid choice.
            raise EOFError

    return feed


@pytest.mark.asyncio
async def test_interactive_checkpoint_finish_writes_canvas_and_manifest(
    tmp_path,
) -> None:
    orch = _make_orchestrator(tmp_path)
    output: list[str] = []

    report = await orch.run_interactive(
        "Sri Lestari", input_fn=_answers("3", "y"), output_fn=output.append
    )

    assert report is not None
    text = "\n".join(output)
    assert "CHECKPOINT SIKLUS 1" in text
    assert "Token kumulatif" in text
    assert "Sisa token" in text
    assert "Estimasi biaya" in text
    assert "Saran node (critic)" in text
    assert "[1] Pivot pada ID node" in text
    assert "[2] Expand" in text
    assert "[3] Selesai" in text
    assert "[4] Lihat semua node" in text

    assert (orch.run_dir / "graph_interactive.html").exists()
    assert (orch.run_dir / "graph.graphml").exists()
    assert report.markdown  # laporan ekstraktif dikembalikan ke main.py
    assert orch.final_report_generated is True
    assert orch.cycle == 1

    manifest = json.loads((orch.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["interactive"] is True
    assert manifest["token_usage"]["total"] == orch.tokens_total
    assert "visualizer" in manifest


@pytest.mark.asyncio
async def test_interactive_pivot_by_node_id_enqueues_neighbor_context_query(
    tmp_path,
) -> None:
    orch = _make_orchestrator(tmp_path)
    state = {"stage": "choose", "node_id": ""}

    def feed(_prompt: str = "") -> str:
        if state["stage"] == "choose":
            state["stage"] = "node"
            state["node_id"] = orch.graph_store.node_records()[0]["id"]
            return "1"
        if state["stage"] == "node":
            state["stage"] = "confirm"
            return state["node_id"]
        if state["stage"] == "confirm":
            state["stage"] = "choose-stop"
            return "y"
        if state["stage"] == "choose-stop":
            state["stage"] = "stop-confirm"
            return "3"
        if state["stage"] == "stop-confirm":
            state["stage"] = "done"
            return "y"
        return "3"

    output: list[str] = []
    await orch.run_interactive("Sri Lestari", input_fn=feed, output_fn=output.append)

    assert state["node_id"] in orch._pivot_history
    assert any("kueri pivot dijadwalkan" in line for line in output)
    assert any("Pivot yang dipilih" in line for line in output)
    assert orch.cycle == 2  # pivot cycle benar-benar dieksekusi


@pytest.mark.asyncio
async def test_interactive_expand_runs_new_level_one_searches(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    output: list[str] = []

    await orch.run_interactive(
        "Sri Lestari",
        input_fn=_answers("2", "y", "3", "y"),
        output_fn=output.append,
    )

    assert any("kueri level-1 dijalankan" in line for line in output)
    assert orch._expand_round == 1
    assert orch._expanded_urls  # hasil pencarian expand benar-benar diproses
    assert any(
        node["level"] == 1
        for node in orch.graph_store.node_records()
        if not node["root"]
    )


@pytest.mark.asyncio
async def test_interactive_eof_finishes_instead_of_hanging(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    output: list[str] = []

    def no_input(_prompt: str = "") -> str:
        raise EOFError

    report = await orch.run_interactive(
        "Sri Lestari", input_fn=no_input, output_fn=output.append
    )

    assert report is not None
    assert any("selesai otomatis" in line for line in output)
    assert orch.final_report_generated is True


@pytest.mark.asyncio
async def test_interactive_rejects_unknown_choice_then_accepts(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    output: list[str] = []

    await orch.run_interactive(
        "Sri Lestari",
        input_fn=_answers("x", "9", "3", "y"),
        output_fn=output.append,
    )

    assert sum("Pilihan tidak dikenal" in line for line in output) == 2


@pytest.mark.asyncio
async def test_make_pivot_tasks_uses_label_and_neighbors(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    orch.graph_store.resolve_and_add(
        [
            Fact(
                subject="Sri Lestari",
                predicate="bekerja_di",
                object="Dinas Kesehatan Sleman",
                provenance=[
                    SourceRef(
                        url="https://a.id", quote="Sri Lestari bekerja di dinas"
                    )
                ],
            )
        ]
    )

    created = orch._make_pivot_tasks(["n1"])

    assert created == 1
    task = orch.frontier.pop()
    assert task is not None
    assert "Sri Lestari" in task.query
    assert "Dinas Kesehatan Sleman" in task.query  # konteks tetangga
    assert task.rationale.startswith("pivot node n1")
    assert task.origin.value == "critic"
    # First pivot round is canvas level 2 (initial search = level 1).
    assert task.level == 2
    assert orch.pivot_round == 1
    # pivot node yang sama tidak diulang
    assert orch._make_pivot_tasks(["n1"]) == 0
    assert orch._make_pivot_tasks(["n99"]) == 0
    assert orch.pivot_round == 1  # tidak ada ronde baru bila 0 task dibuat


@pytest.mark.asyncio
async def test_interactive_pivot_confirmation_can_cancel(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    # run_interactive builds its own graph in _setup; use a stateful feeder that
    # picks the first node only after the first checkpoint exists.
    state = {"choice": 0, "node_id": ""}
    prompts: list[str] = []
    output: list[str] = []

    def feed(prompt: str = "") -> str:
        prompts.append(prompt)
        if state["choice"] == 0:
            state["choice"] = 1
            state["node_id"] = orch.graph_store.node_records()[0]["id"]
            return "1"
        if state["choice"] == 1:
            state["choice"] = 2
            return state["node_id"]
        if state["choice"] == 2:
            state["choice"] = 3
            return "n"  # batalkan -> langsung kembali ke pemilihan node
        if state["choice"] == 3:
            state["choice"] = 4
            return ""  # kosongkan ID -> kembali ke menu
        if state["choice"] == 4:
            state["choice"] = 5
            return "3"
        return "y"

    await orch.run_interactive("Sri Lestari", input_fn=feed, output_fn=output.append)

    assert any("Pivot yang dipilih" in line for line in output)
    assert sum("ID node:" in prompt for prompt in prompts) == 2
    assert state["node_id"] not in orch._pivot_history
    assert not any("kueri pivot dijadwalkan" in line for line in output)


@pytest.mark.asyncio
async def test_interactive_menu_4_lists_all_nodes(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    output: list[str] = []

    await orch.run_interactive(
        "Sri Lestari",
        input_fn=_answers("4", "3", "y"),
        output_fn=output.append,
    )

    text = "\n".join(output)
    assert "SEMUA NODE" in text
    assert "Level 1:" in text
    assert "sudah dipivot" not in text


def test_make_pivot_tasks_uses_parent_level_per_branch(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    orch.graph_store.ensure_root("Root")
    orch.graph_store.resolve_and_add(
        [
            Fact(
                subject="Root",
                predicate="terkait",
                object="Anak Level 1",
                provenance=[SourceRef(url="https://a.id", quote="Root terkait Anak")],
            )
        ],
        level=1,
    )
    orch.graph_store.resolve_and_add(
        [
            Fact(
                subject="Anak Level 1",
                predicate="punya",
                object="Cucu Level 2",
                provenance=[SourceRef(url="https://b.id", quote="Anak punya Cucu")],
            )
        ],
        level=2,
    )
    by_label = {
        node["label"]: node["id"] for node in orch.graph_store.node_records()
    }

    created = orch._make_pivot_tasks(
        [by_label["Anak Level 1"], by_label["Cucu Level 2"]]
    )

    assert created == 2
    tasks = [orch.frontier.pop(), orch.frontier.pop()]
    levels = {task.parent_id: task.level for task in tasks if task is not None}
    assert levels[by_label["Anak Level 1"]] == 2
    assert levels[by_label["Cucu Level 2"]] == 3


def test_interactive_budget_override_unlocks_stop_reason(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    orch.budget.add_tokens(orch.budget.max_tokens)

    assert orch._interactive_stop_reason() is not None
    orch._budget_overrides.add("token")
    assert orch._interactive_stop_reason() is None


@pytest.mark.asyncio
async def test_interactive_empty_frontier_still_opens_checkpoint(
    tmp_path, monkeypatch
) -> None:
    orch = _make_orchestrator(tmp_path)
    monkeypatch.setattr(orch, "_pop_task", lambda **_kwargs: ("empty", None))
    output: list[str] = []

    await orch.run_interactive(
        "Sri Lestari",
        input_fn=_answers("3", "y"),
        output_fn=output.append,
    )

    text = "\n".join(output)
    assert "CHECKPOINT SIKLUS 0" in text
    assert "[3] Selesai" in text


def test_interactive_stop_reason_ignores_frontier_and_novelty(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    orch.evaluator.zero_streak = 99

    assert orch._stop_reason() is not None  # non-interactive auto-stop
    assert orch._interactive_stop_reason() is None  # interaktif: user decides


@pytest.mark.asyncio
async def test_expand_deep_search_uses_larger_limit(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    calls: list[tuple[str, int]] = []

    class _RecordingBackend(_StubBackend):
        async def search(self, query: str, limit: int = 10):
            calls.append((query, limit))
            return await super().search(query, limit=limit)

    orch.autosearch.backend = _RecordingBackend()
    await orch._setup("Sri Lestari")
    queries_run, new_nodes = await orch._expand_documents()

    assert queries_run == 3
    assert new_nodes > 0
    assert max(limit for _query, limit in calls) > 10
    assert any(key.startswith("deep:") for key in orch._expand_history)


@pytest.mark.asyncio
async def test_interactive_budget_limit_asks_before_continuing(tmp_path) -> None:
    orch = _make_orchestrator(tmp_path)
    orch.budget.max_tokens = 0
    prompts: list[str] = []
    output: list[str] = []

    def feed(prompt: str = "") -> str:
        prompts.append(prompt)
        return "n"

    report = await orch.run_interactive(
        "Sri Lestari", input_fn=feed, output_fn=output.append
    )

    text = "\n".join(output)
    assert report is not None
    assert "token budget exhausted" in text
    assert any("Lanjut melebihi batas token" in prompt for prompt in prompts)
    assert orch.cycle == 0
    assert "token" not in orch._budget_overrides
