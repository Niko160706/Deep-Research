"""Heuristic Critic — evaluates argument logic for fallacies and gaps.

Phase 1 returns no gaps (mock). Phase 2 wires ``deepseek-reasoner`` to detect
weak causality and missing evidence across the vector store.
"""
from __future__ import annotations

import logging

from drx.models import Gap
from drx.store.vector_store import VectorStore

log = logging.getLogger("drx.critic.heuristic")


class HeuristicCritic:
    """Spars against the argument cards and surfaces holes as Gap objects."""

    async def check(self, vector_store: VectorStore) -> list[Gap]:
        return []
