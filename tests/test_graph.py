"""Unit tests for GraphStore Louvain community pruning."""
from __future__ import annotations

from pathlib import Path

from drx.models import Fact
from drx.store.graph_store import GraphStore


def _store(tmp_path: Path) -> GraphStore:
    return GraphStore(tmp_path)


def test_prune_removes_isolated_component(tmp_path: Path) -> None:
    gs = _store(tmp_path)
    # main component
    gs.graph.add_edge("niko", "surabaya")
    gs.graph.add_edge("surabaya", "informatika")
    # disconnected island
    gs.graph.add_edge("island1", "island2")
    removed = gs.prune_louvain()
    assert removed == 2
    assert gs.node_count() == 3
    assert "island1" not in gs.graph


def test_prune_keeps_bridged_cluster(tmp_path: Path) -> None:
    gs = _store(tmp_path)
    gs.graph.add_edge("niko", "surabaya")
    gs.graph.add_edge("surabaya", "informatika")
    # a cluster bridged into the main one through a single edge
    gs.graph.add_edge("informatika", "riset")
    gs.graph.add_edge("riset", "publikasi")
    removed = gs.prune_louvain()
    assert gs.node_count() == 5
    assert removed == 0


def test_prune_prefers_target_cluster(tmp_path: Path) -> None:
    gs = _store(tmp_path)
    gs.graph.add_edge("target", "topic_a")
    gs.graph.add_edge("unrelated", "topic_b")
    gs.graph.add_edge("unrelated", "topic_c")
    gs.graph.add_edge("unrelated", "topic_d")
    removed = gs.prune_louvain(target="target")
    # target cluster kept, larger-but-unrelated island removed
    assert "target" in gs.graph
    assert "topic_a" in gs.graph
    assert removed == 4


def test_prune_noop_on_tiny_graph(tmp_path: Path) -> None:
    gs = _store(tmp_path)
    gs.graph.add_node("solo")
    assert gs.prune_louvain() == 0


def test_prune_after_resolve_and_add_updates_known_nodes(tmp_path: Path) -> None:
    gs = _store(tmp_path)
    facts = [
        Fact(subject="niko", predicate="lives_in", object="surabaya", confidence_score=0.9),
        Fact(subject="island", predicate="part_of", object="faraway", confidence_score=0.4),
    ]
    gs.resolve_and_add(facts)
    removed = gs.prune_louvain()
    assert removed == 2
    assert gs.node_count() == 2


def test_prune_pins_isolated_root_anchor_and_keeps_real_cluster(tmp_path: Path) -> None:
    """Level-0 root anchor must not hijack Louvain main-cluster selection."""
    gs = _store(tmp_path)
    gs.ensure_root("Sri Lestari")  # isolated UI anchor, exact target label
    gs.graph.add_node("profil", label="Sri Lestari profil dan karier 0")
    gs.graph.add_edge("sleman", "profil")
    gs.graph.add_edge("sri", "profil")
    gs.graph.add_edge("lestari", "profil")

    removed = gs.prune_louvain("Sri Lestari")

    assert removed == 0
    assert gs.node_count() == 5
    assert "Sri Lestari" in {gs.label_of(node) for node in gs.graph.nodes}  # pinned
    assert {"sleman", "sri", "lestari", "profil"} <= set(gs.graph.nodes)
