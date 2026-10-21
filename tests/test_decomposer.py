"""Unit tests for the 70/30 decomposer and transparency tree."""
from __future__ import annotations

import pytest

from drx.gate.decomposer import Decomposer, build_anchored, render_tree, split_context
from drx.models import Mode, ResearchPlan, SurveyResult


def test_split_context() -> None:
    assert split_context("Surabaya, Informatika") == ["Surabaya", "Informatika"]
    assert split_context("Surabaya; Informatika|Jakarta") == ["Surabaya", "Informatika", "Jakarta"]
    assert split_context("") == []


def test_build_anchored() -> None:
    assert (
        build_anchored("Niko", ["Surabaya", "Informatika"])
        == '"Niko" AND ("Surabaya" OR "Informatika")'
    )
    assert build_anchored("Niko", []) == "Niko"


def _plan(query: str) -> ResearchPlan:
    return ResearchPlan(hypothesis=query, seed_queries=[query])


@pytest.mark.asyncio
async def test_decompose_with_context_majority_anchored() -> None:
    d = Decomposer(context="Surabaya, Informatika")
    tasks = d.decompose(_plan("Niko Rajani Syahputra Pane"), SurveyResult(query="", hits=[]), Mode.ENTITY)
    anchored = [t for t in tasks if t.is_anchored]
    exploratory = [t for t in tasks if not t.is_anchored]
    assert len(anchored) == 3  # combined + one per context term
    assert len(exploratory) >= 1
    assert len(anchored) > len(exploratory)
    assert all(t.mode == Mode.ENTITY for t in tasks)


@pytest.mark.asyncio
async def test_decompose_ratio_stays_near_70_30() -> None:
    d = Decomposer(context="Surabaya, Informatika")
    tasks = d.decompose(
        _plan("profil lengkap Budi Santoso mahasiswa"),
        SurveyResult(query="", hits=[]),
        Mode.ENTITY,
    )
    anchored = [t for t in tasks if t.is_anchored]
    ratio = len(anchored) / len(tasks)
    assert 0.6 <= ratio <= 0.8
    assert ratio > 0.5
    # keyword-anchored variant is generated when stopwords can be stripped
    assert any("varian kata kunci" in t.rationale for t in anchored)


@pytest.mark.asyncio
async def test_decompose_anchored_queries_are_context_locked() -> None:
    d = Decomposer(context="Surabaya, Informatika")
    tasks = d.decompose(_plan("Niko"), SurveyResult(query="", hits=[]), Mode.ENTITY)
    queries = [t.query for t in tasks]
    assert '"Niko" AND ("Surabaya" OR "Informatika")' in queries
    assert '"Niko" AND ("Surabaya")' in queries
    assert '"Niko" AND ("Informatika")' in queries
    for t in tasks:
        if t.is_anchored:
            assert ' AND ("' in t.query
            assert t.context == ["Surabaya", "Informatika"]
        else:
            assert "AND" not in t.query
            assert t.context == []


@pytest.mark.asyncio
async def test_decompose_without_context_is_100pct_exploratory() -> None:
    d = Decomposer(context="")
    tasks = d.decompose(_plan("Alan Turing"), SurveyResult(query="", hits=[]), Mode.ENTITY)
    assert tasks
    assert all(not t.is_anchored for t in tasks)


def test_decompose_exploratory_task_sampled_early() -> None:
    """The 30% branch must not be starved when the run stops early."""
    d = Decomposer(context="Surabaya, Informatika")
    tasks = d.decompose(_plan("Niko"), SurveyResult(query="", hits=[]), Mode.ENTITY)
    assert tasks[0].is_anchored  # anchored leads
    assert any(not t.is_anchored for t in tasks[:2])  # exploratory within first two


def test_render_tree_contains_split_and_rationale() -> None:
    tasks = Decomposer(context="Surabaya").decompose(
        _plan("Niko"), SurveyResult(query="", hits=[]), Mode.ENTITY
    )
    tasks[0].score = 0.42
    tasks[0].result_count = 7
    tasks[0].status = "searched"
    tree = render_tree("Niko", tasks)
    assert "Query Decomposition Tree" in tree
    assert "anchored" in tree
    assert "exploratory" in tree
    assert "rationale" in tree
    assert "score=0.42" in tree
    assert "results=7" in tree
    assert "status=searched" in tree


def test_build_anchored_groups_alias_queries() -> None:
    """An alias query is grouped, not re-quoted (no nested quotes)."""
    assert (
        build_anchored('"Bayu Skak" OR "Bayu Eko Moektito"', ["Sleman"])
        == '("Bayu Skak" OR "Bayu Eko Moektito") AND ("Sleman")'
    )


def test_render_tree_shows_aliases_and_status() -> None:
    from drx.models import QueryTask, TaskOrigin

    tasks = [
        QueryTask(
            id="t1",
            query='"Bayu Skak" AND ("Sleman")',
            mode=Mode.ENTITY,
            depth=1,
            origin=TaskOrigin.INITIAL,
            hypothesis="Bayu Skak",
            rationale="terikat context ['Sleman']",
            backend="hybrid",
            is_anchored=True,
            status="searched",
            result_count=7,
            score=0.5,
        )
    ]
    tree = render_tree(
        '"Bayu Skak" OR "Bayu Eko Moektito"',
        tasks,
        aliases=["Bayu Skak", "Bayu Eko Moektito"],
    )
    assert "Entity aliases: Bayu Skak, Bayu Eko Moektito" in tree
    assert "status=searched" in tree
    assert "results=7" in tree
