"""Entity Critic — verifies the knowledge graph for contradictions.

Phase 1 returns no gaps (mock). Phase 2 wires ``deepseek-reasoner``
(roles.critic) to look for timeline contradictions, conflicting affiliations,
and missing links, then emits follow-up sub-queries.
"""
from __future__ import annotations

import logging

from drx.models import Gap
from drx.store.graph_store import GraphStore

log = logging.getLogger("drx.critic.entity")


class EntityCritic:
    """Spars against the graph and surfaces holes as Gap objects."""

    async def check(self, graph_store: GraphStore) -> list[Gap]:
        return []
