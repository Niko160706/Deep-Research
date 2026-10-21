"""Super Agent — Def1 (hypothesis) and Def2 (plan decomposition).

Phase 1 uses deterministic (non-LLM) outputs so the pipeline is runnable
without DeepSeek keys. Phase 2 wires:
    * Def1 -> ``deepseek-chat`` (roles.def1_planner)
    * Def2 -> ``deepseek-chat`` (roles.def2_decomposer)
Def2 delegates to :class:`drx.gate.decomposer.Decomposer` for the 70/30
context-diversification strategy and transparency-tree metadata.
"""
from __future__ import annotations

import logging

from drx.gate.decomposer import Decomposer
from drx.models import Mode, QueryTask, ResearchPlan, SurveyResult
from drx.textutil import keyword_query

log = logging.getLogger("drx.gate.super_agent")


class SuperAgent:
    """Two-stage planner."""

    def __init__(self, context: str = "", backend: str = "hybrid") -> None:
        self.context = context
        self.backend = backend

    async def def1(self, query: str, mode: Mode) -> ResearchPlan:
        """Produce an initial hypothesis + strategy + seed queries (mock)."""
        return ResearchPlan(
            hypothesis=query,
            strategy=f"{mode.value}-track mapping (mock Def1)",
            seed_queries=[query],
        )

    async def def2(
        self,
        plan: ResearchPlan,
        survey: SurveyResult,
        mode: Mode,
    ) -> list[QueryTask]:
        """Decompose the plan into sub-queries (delegates to the Decomposer)."""
        return Decomposer(self.context, backend=self.backend).decompose(plan, survey, mode)

    @staticmethod
    def _keyword_query(query: str) -> str:
        return keyword_query(query)
