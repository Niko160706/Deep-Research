"""Hybrid scheduler: FIFO queue (breadth-first) + LIFO stack (depth-first).

The single most important invariant: ``pop()`` drains the **stack** before the
**queue**. Critic-generated sub-queries (deep-dives into anomalies) are pushed
onto the stack and handled immediately; Def2's initial sub-queries (broad
mapping) sit in the queue and are handled in order.
"""
from __future__ import annotations

import logging
from collections import deque

from drx.models import Layer, QueryTask

log = logging.getLogger("drx.memory.frontier")


def normalize_query(query: str) -> str:
    """Order-insensitive, case-folded key for fuzzy-ish dedup."""
    return " ".join(sorted(t for t in query.lower().split() if t))


class HybridFrontier:
    """Queue + stack with depth guards, dedup and a failure graveyard."""

    def __init__(self, max_depth: int = 4) -> None:
        self.max_depth = max_depth
        self._stack: deque[QueryTask] = deque()  # LIFO (DFS)
        self._queue: deque[QueryTask] = deque()  # FIFO (BFS)
        self._visited_queries: set[str] = set()
        self._graveyard: dict[str, str] = {}      # task_id -> failure reason
        self._failed_queries: list[str] = []      # normalized failed queries
        self._exhausted: set[str] = set()         # task ids dropped by depth

    # ---- push / pop ----
    def push(self, task: QueryTask, layer: Layer = Layer.QUEUE) -> bool:
        """Add a task. Returns False (and drops it) when it violates a guard."""
        if task.depth > self.max_depth:
            log.debug("drop %s: depth %d > max %d", task.id, task.depth, self.max_depth)
            self._exhausted.add(task.id)
            return False
        if self.is_duplicate(task):
            return False
        if self.is_failed_similar(task):
            log.debug("drop %s: similar to a previously failed query", task.id)
            return False

        self._visited_queries.add(normalize_query(task.query))
        if layer is Layer.STACK:
            self._stack.append(task)
        else:
            self._queue.append(task)
        return True

    def pop(self) -> QueryTask | None:
        """Return the next task: stack (LIFO) first, then queue (FIFO)."""
        if self._stack:
            return self._stack.pop()
        if self._queue:
            return self._queue.popleft()
        return None

    # ---- status ----
    def has_work(self) -> bool:
        return bool(self._stack or self._queue)

    def __len__(self) -> int:
        return len(self._stack) + len(self._queue)

    # ---- dedup ----
    def is_duplicate(self, task: QueryTask) -> bool:
        key = normalize_query(task.query)
        if key in self._visited_queries:
            log.debug("duplicate query skipped: %s", task.query)
            return True
        return False

    # ---- failure handling (anti-cycle) ----
    def mark_failed(self, task: QueryTask, reason: str) -> None:
        """Record a dead-end so critics cannot respawn a near-identical query."""
        self._graveyard[task.id] = reason
        self._failed_queries.append(normalize_query(task.query))
        log.info("task failed (%s): %s", reason, task.query)

    def is_failed_similar(self, task: QueryTask, threshold: int = 90) -> bool:
        """Block respawning a task whose query is ~identical to a failure."""
        norm = normalize_query(task.query)
        if not norm:
            return False
        for failed in self._failed_queries:
            score = self._similarity(norm, failed)
            if score >= threshold:
                return True
        return False

    @staticmethod
    def _similarity(a: str, b: str) -> int:
        if a == b:
            return 100
        try:  # rapidfuzz is a base dep; degrade to exact-match otherwise
            from rapidfuzz import fuzz

            return int(fuzz.ratio(a, b))
        except ImportError:  # pragma: no cover
            return 0

    def mark_exhausted(self, task: QueryTask) -> None:
        self._exhausted.add(task.id)
