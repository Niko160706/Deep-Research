"""LLM Router — classifies a query into Entity / Heuristic / Hybrid.

Phase 1 returns a deterministic heuristic so the loop runs without keys.
Phase 2 wires ``deepseek-chat`` (config/models.yaml ``roles.router``) with a
low temperature to emit a single mode token.
"""
from __future__ import annotations

import logging

from drx.models import Mode

log = logging.getLogger("drx.gate.router")

_WHY_HOW_WORDS = (
    "why", "how", "mengapa", "bagaimana", "cara", "alasan", "sebab",
    "compare", "versus", "vs", "impact", "effect", "solution", "solusi",
)


class Router:
    """Classify the primary query into a research mode."""

    async def classify(self, query: str) -> Mode:
        # Mock cognitive decision — deterministic heuristic stand-in.
        words = {w.strip(".,!?;:") for w in query.lower().split()}
        if words & set(_WHY_HOW_WORDS):
            return Mode.HEURISTIC
        return Mode.ENTITY
