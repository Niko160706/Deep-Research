"""Cycle evaluator — quantitative novelty (Python) + guarded self-answer (LLM).

Novelty is computed in pure Python (0 tokens) and runs BEFORE any qualitative
LLM check. The qualitative check is further guarded so it only fires when
there is genuinely new material, and its result is cached against a graph
snapshot in Phase 2.
"""
from __future__ import annotations

import logging

from drx.store.graph_store import GraphStore

log = logging.getLogger("drx.critic.evaluator")


class Evaluator:
    """Decides whether the research loop should continue."""

    def __init__(self, floor: float = 0.10, stagnation_limit: int = 3) -> None:
        self.floor = floor
        self.stagnation_limit = stagnation_limit
        self.zero_streak = 0
        self._self_answer_cache: dict[int, bool] = {}

    # ---- quantitative (0 tokens) ----
    def novelty_rate(self, new_nodes: int, processed: int) -> float:
        if processed <= 0:
            return 0.0
        return new_nodes / processed

    def record_zero(self) -> None:
        self.zero_streak += 1

    def reset_zero(self) -> None:
        self.zero_streak = 0

    def stagnated(self) -> bool:
        return self.zero_streak >= self.stagnation_limit

    # ---- qualitative (LLM, guarded) ----
    def should_self_answer(self, new_nodes: int) -> bool:
        return new_nodes > 0

    async def self_answer_sufficient(self, graph_store: GraphStore) -> bool:
        # Mock — Phase 2 wires deepseek-reasoner and caches on graph snapshot.
        return graph_store.edge_count() >= 20
