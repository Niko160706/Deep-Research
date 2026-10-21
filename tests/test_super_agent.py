"""Unit tests for the Super Agent (mode propagation + drift prevention)."""
from __future__ import annotations

import pytest

from drx.gate.super_agent import SuperAgent
from drx.models import Mode, ResearchPlan, SearchResult, SurveyResult


@pytest.mark.asyncio
async def test_def2_propagates_router_mode() -> None:
    agent = SuperAgent()
    plan = ResearchPlan(
        hypothesis="why did the Roman Empire fall",
        seed_queries=["why did the Roman Empire fall"],
    )
    tasks = await agent.def2(plan, SurveyResult(query="", hits=[]), Mode.HEURISTIC)
    assert tasks
    assert all(t.mode == Mode.HEURISTIC for t in tasks)


@pytest.mark.asyncio
async def test_def2_does_not_research_raw_survey_titles() -> None:
    agent = SuperAgent()
    plan = ResearchPlan(
        hypothesis="why did the Roman Empire fall",
        seed_queries=["why did the Roman Empire fall"],
    )
    survey = SurveyResult(
        query="",
        hits=[SearchResult(url="http://x", title="WHY | Cambridge Dictionary")],
    )
    tasks = await agent.def2(plan, survey, Mode.HEURISTIC)
    queries = [t.query for t in tasks]
    assert "WHY | Cambridge Dictionary" not in queries
    assert "why did the Roman Empire fall" in queries


def test_keyword_query_strips_stopwords() -> None:
    agent = SuperAgent()
    assert agent._keyword_query("why did the Roman Empire fall") == "Roman Empire fall"
