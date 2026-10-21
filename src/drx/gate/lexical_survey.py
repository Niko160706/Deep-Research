"""Empirical Lexical Survey — Python-only, 0 LLM tokens.

Runs the mechanical search once and returns raw titles/snippets so Def2 can
revise its plan against real-world signals. This is the "eyes" of the planner
and costs nothing in tokens.
"""
from __future__ import annotations

import logging

from drx.models import ResearchPlan, SurveyResult
from drx.muscle.autosearch import AutoSearch

log = logging.getLogger("drx.gate.lexical_survey")


class LexicalSurvey:
    """Fetches a lightweight, token-free snapshot of the search landscape."""

    def __init__(self, autosearch: AutoSearch) -> None:
        self.autosearch = autosearch

    async def survey(self, plan: ResearchPlan) -> SurveyResult:
        queries = plan.seed_queries or []
        if not queries:
            return SurveyResult(query="")
        primary = queries[0]
        hits = await self.autosearch.search(primary)
        log.info("lexical survey for %r returned %d hits", primary, len(hits))
        return SurveyResult(query=primary, hits=hits)
