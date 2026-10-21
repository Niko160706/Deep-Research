"""Unit tests for the hybrid FIFO/LIFO frontier and its guards."""
from __future__ import annotations

import uuid

from drx.memory.frontier import HybridFrontier
from drx.models import Layer, Mode, QueryTask, TaskOrigin


def make_task(query: str, depth: int = 1, origin: TaskOrigin = TaskOrigin.INITIAL) -> QueryTask:
    return QueryTask(
        id=str(uuid.uuid4()),
        query=query,
        mode=Mode.ENTITY,
        depth=depth,
        origin=origin,
    )


def test_queue_is_fifo() -> None:
    f = HybridFrontier(max_depth=4)
    f.push(make_task("alpha"), Layer.QUEUE)
    f.push(make_task("beta"), Layer.QUEUE)
    assert f.pop().query == "alpha"
    assert f.pop().query == "beta"
    assert f.pop() is None


def test_stack_has_priority_over_queue() -> None:
    f = HybridFrontier(max_depth=4)
    f.push(make_task("alpha"), Layer.QUEUE)
    f.push(make_task("beta"), Layer.QUEUE)
    f.push(make_task("gamma"), Layer.STACK)
    assert f.pop().query == "gamma"   # LIFO first
    assert f.pop().query == "alpha"   # then FIFO
    assert f.pop().query == "beta"


def test_depth_guard_on_push() -> None:
    f = HybridFrontier(max_depth=4)
    assert f.push(make_task("deep", depth=5)) is False
    assert not f.has_work()


def test_dedup_order_insensitive() -> None:
    f = HybridFrontier(max_depth=4)
    assert f.push(make_task("Hello World")) is True
    assert f.push(make_task("world hello")) is False


def test_graveyard_blocks_similar_query() -> None:
    f = HybridFrontier(max_depth=4)
    task = make_task("what is the capital of france")
    f.push(task)
    f.mark_failed(task, "empty_results")
    assert f.is_failed_similar(make_task("what is the capital of france")) is True
