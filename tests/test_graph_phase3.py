"""Tests for the Phase 3 graph store: stable IDs, rapidfuzz resolution, export.

The store is the bridge between grounded triples and the interactive canvas, so
these tests pin the invariants the UI and critic rely on.
"""
from __future__ import annotations

from pathlib import Path

import networkx as nx

from drx.models import Fact, SourceRef
from drx.store.graph_store import GraphStore


def _fact(
    subject: str,
    predicate: str,
    obj: str,
    url: str,
    quote: str = "kutipan bukti",
    confidence: float = 0.6,
) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        confidence_score=confidence,
        provenance=[SourceRef(url=url, source_id="S1", quote=quote)],
    )


def test_pt_variants_merge_with_stable_ids(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    added = store.resolve_and_add(
        [
            _fact(
                "PT Jaya Abadi",
                "bekerja_sama",
                "Dinas Kesehatan Sleman",
                "https://a.id",
                quote="PT Jaya Abadi bermitra dengan dinas",
            ),
            _fact(
                "PT. Jaya Abadi",
                "bekerja_sama",
                "Dinas Kesehatan Sleman",
                "https://b.id",
                quote="PT Jaya Abadi mengerjakan proyek",
            ),
        ]
    )

    assert added == 2
    assert store.node_count() == 2
    records = {record["label"]: record for record in store.node_records()}
    # ID stabil dan dapat dipakai user untuk pivot
    assert records["PT Jaya Abadi"]["id"] == "n1"
    assert records["Dinas Kesehatan Sleman"]["id"] == "n2"

    # dua sumber independen -> satu edge verified
    assert store.edge_count() == 1
    triple = store.triples()[0]
    assert triple["source_count"] == 2
    assert triple["verified"] is True
    assert triple["subject_id"] == "n1"
    assert triple["object_id"] == "n2"


def test_fuzzy_typo_merges_but_extra_name_tokens_do_not(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    store.resolve_and_add(
        [
            _fact("Universitas Gadjah Mada", "berlokasi_di", "Yogyakarta", "https://a.id"),
            _fact("Universitas Gajah Mada", "berlokasi_di", "Yogyakarta", "https://b.id"),
            _fact("Sri Lestari", "bekerja_sebagai", "perawat", "https://c.id"),
            _fact(
                "Sri Lestari Wulandari",
                "bekerja_sebagai",
                "bidan",
                "https://d.id",
            ),
        ]
    )

    labels = {record["label"] for record in store.node_records()}
    assert "Universitas Gadjah Mada" in labels
    assert "Universitas Gajah Mada" not in labels  # typo -> merged
    # nama lebih panjang bukan alias otomatis (anti name contamination)
    assert "Sri Lestari" in labels
    assert "Sri Lestari Wulandari" in labels


def test_functional_predicate_conflict_flags_contradiction(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    store.resolve_and_add(
        [
            _fact("Sri Lestari", "bekerja_sebagai", "perawat", "https://a.id"),
            _fact("Sri Lestari", "bekerja_sebagai", "bidan", "https://b.id"),
        ]
    )

    edges = store.edge_records()
    assert len(edges) == 2
    assert all(edge["contradiction"] is True for edge in edges)


def test_export_graphml_with_labels_and_scalarized_lists(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    store.resolve_and_add(
        [
            _fact("Sri Lestari", "bekerja_di", "Dinas Kesehatan Sleman", "https://a.id"),
            _fact("Sri Lestari", "bekerja_di", "Dinas Kesehatan Sleman", "https://b.id"),
        ]
    )

    path = store.export()

    assert path is not None and path.exists()
    graph = nx.read_graphml(path)
    by_label = {data.get("label"): node for node, data in graph.nodes(data=True)}
    assert by_label["Sri Lestari"].startswith("n")
    sri = by_label["Sri Lestari"]
    dinkes = by_label["Dinas Kesehatan Sleman"]
    payload = graph.get_edge_data(sri, dinkes)
    edge = payload if "predicate" in payload else next(iter(payload.values()))
    assert edge["predicate"] == "bekerja_di"
    assert edge["verified"] is True


def test_suggest_nodes_returns_grounded_ids_and_neighbors(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    store.resolve_and_add(
        [
            _fact("Sri Lestari", "bekerja_di", "Dinas Kesehatan Sleman", "https://a.id"),
            _fact("Sri Lestari", "bekerja_di", "Dinas Kesehatan Sleman", "https://b.id"),
            _fact("PT Jaya Abadi", "bekerja_sama", "Dinas Kesehatan Sleman", "https://c.id"),
        ]
    )

    suggestions = store.suggest_nodes("Sri Lestari", "Sleman", limit=3)

    assert suggestions
    assert all(item["node_id"] in store.graph for item in suggestions)
    assert all(item["label"] and item["reason"] for item in suggestions)
    assert all(item["node_id"] != "n1" for item in suggestions)  # target excluded

    neighbors = store.neighbors("n2", limit=5)
    assert {item["label"] for item in neighbors} >= {"Sri Lestari", "PT Jaya Abadi"}


def test_node_levels_pin_root_and_keep_first_seen_level(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    root_id = store.ensure_root("Sri Lestari")
    assert root_id is not None

    store.resolve_and_add(
        [_fact("Sri Lestari", "bekerja_di", "Dinas Kesehatan Sleman", "https://a.id")]
    )
    # Pivot round 1: a brand-new node is level 2, while the already-known
    # Dinas node keeps the level it was first discovered at (1).
    store.resolve_and_add(
        [
            _fact(
                "Dinas Kesehatan Sleman",
                "membawahi",
                "Puskesmas Depok",
                "https://b.id",
            )
        ],
        level=2,
    )

    records = {record["label"]: record for record in store.node_records()}
    assert records["Sri Lestari"]["id"] == root_id
    assert records["Sri Lestari"]["level"] == 0
    assert records["Sri Lestari"]["root"] is True
    assert records["Dinas Kesehatan Sleman"]["level"] == 1
    assert records["Dinas Kesehatan Sleman"]["root"] is False
    assert records["Puskesmas Depok"]["level"] == 2

    # Level is a scalar GraphML attribute -> survives export for auditing.
    path = store.export()
    assert path is not None
    graph = nx.read_graphml(path)
    by_label = {data.get("label"): data for _, data in graph.nodes(data=True)}
    assert int(by_label["Sri Lestari"]["level"]) == 0
    assert int(by_label["Puskesmas Depok"]["level"]) == 2


def test_triples_limit_zero_returns_all_edges(tmp_path: Path) -> None:
    store = GraphStore(tmp_path)
    facts = [
        _fact(
            f"Node{index:03d}x{index * 7919}",
            "terkait",
            f"Org{index:03d}y{index * 104729}",
            f"https://e.id/{index}",
        )
        for index in range(60)
    ]
    store.resolve_and_add(facts)

    assert len(store.triples(limit=5)) == 5
    assert len(store.triples(limit=0)) == 60
