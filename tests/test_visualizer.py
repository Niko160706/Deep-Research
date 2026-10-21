"""Tests for the PRD-compliant interactive canvas renderer.

The HTML must carry the exact design tokens from ``PRD_design_new.md`` (clean
minimalist light theme: off-white dot-grid canvas, white node cards, dark slate
text, accent badges) plus the horizontal hierarchical mindmap layout
(Left-to-Right ``cubicBezier``) and the interaction surface from
``PRD_design.md`` (action bar, inspector, suggestions, multi-select).
"""
from __future__ import annotations

from pathlib import Path

from drx.brain.critic import SuggestedNode
from drx.models import Fact, SourceRef
from drx.store.graph_store import GraphStore
from drx.visualizer import graph_view
from drx.visualizer.graph_view import (
    BG,
    BORDER,
    EDGE_COLORS,
    GRID,
    NODE_BADGES,
    NODE_COLORS,
    PANEL,
    TEXT,
    build_payload,
    edge_style,
    render_graph_html,
    vis_elements,
)


def _fact(subject: str, predicate: str, obj: str, urls: list[str]) -> Fact:
    return Fact(
        subject=subject,
        predicate=predicate,
        object=obj,
        provenance=[
            SourceRef(
                url=url,
                source_id="S1",
                quote=f"{subject} {predicate.replace('_', ' ')} {obj}",
            )
            for url in urls
        ],
    )


def _store(tmp_path: Path) -> GraphStore:
    store = GraphStore(tmp_path)
    store.ensure_root("Sri Lestari")  # level 0 anchor
    store.resolve_and_add(
        [
            _fact(
                "Sri Lestari",
                "bekerja_sebagai",
                "perawat",
                ["https://a.id", "https://b.id"],
            ),
            _fact(
                "Sri Lestari",
                "bekerja_di",
                "Dinas Kesehatan Sleman",
                ["https://a.id"],
            ),
            _fact(
                "PT Jaya Abadi",
                "bekerja_sama",
                "Dinas Kesehatan Sleman",
                ["https://c.id"],
            ),
            _fact("SK Bupati Sleman 2024", "menetapkan", "Sri Lestari", ["https://d.id"]),
            _fact("Pelantikan Bupati Sleman", "melibatkan", "Sri Lestari", ["https://e.id"]),
        ]
    )
    # Pivot round 1 -> new node at level 2, plus a backward/cross edge to level 1.
    store.resolve_and_add(
        [
            _fact(
                "Puskesmas Depok",
                "dikunjungi_oleh",
                "Sri Lestari",
                ["https://f.id"],
            ),
        ],
        level=2,
    )
    return store


def test_palette_matches_prd_new() -> None:
    # Clean Minimalist Light Theme — PRD_design_new.md §1.1.
    assert (BG, PANEL, BORDER, GRID, TEXT) == (
        "#F8FAFC",
        "#FFFFFF",
        "#E2E8F0",
        "#CBD5E1",
        "#0F172A",
    )
    assert NODE_BADGES["root"] == {
        "bg": "#EEF2FF",
        "border": "#6366F1",
        "text": "#4338CA",
    }
    assert NODE_BADGES["query"] == {
        "bg": "#ECFEFF",
        "border": "#06B6D4",
        "text": "#0E7490",
    }
    assert NODE_BADGES["individu"] == {
        "bg": "#F3E8FF",
        "border": "#A855F7",
        "text": "#6B21A8",
    }
    assert NODE_BADGES["instansi"] == {
        "bg": "#FEF3C7",
        "border": "#F59E0B",
        "text": "#B45309",
    }
    assert NODE_BADGES["perusahaan"] == NODE_BADGES["instansi"]
    # Backward-compatible border map derives from the badges.
    assert NODE_COLORS["individu"] == "#A855F7"
    assert NODE_COLORS["instansi"] == "#F59E0B"
    assert NODE_COLORS["perusahaan"] == "#F59E0B"
    assert NODE_COLORS["dokumen"] == "#10B981"
    assert NODE_COLORS["peristiwa"] == "#F43F5E"
    assert EDGE_COLORS == {
        "verified": "#22C55E",
        "single": "#94A3B8",
        "contradiction": "#EF4444",
    }


def test_edge_style_verification_rules() -> None:
    assert edge_style({"verified": True, "source_count": 2}) == {
        "color": "#22C55E",
        "dashes": False,
    }
    assert edge_style({"verified": False, "source_count": 1}) == {
        "color": "#94A3B8",
        "dashes": True,
    }
    assert edge_style(
        {"verified": True, "source_count": 3, "contradiction": True}
    ) == {"color": "#EF4444", "dashes": False}


def test_build_payload_trims_and_filters(tmp_path: Path) -> None:
    store = _store(tmp_path)
    suggestions = [
        SuggestedNode(node_id="n1", label="Sri Lestari", reason="target").model_dump(),
        SuggestedNode(node_id="n99", label="Hilang", reason="halusinasi").model_dump(),
    ]

    payload = build_payload(
        store,
        target_entity="Sri Lestari",
        context="Sleman",
        critic_suggestions=suggestions,
        cycle=2,
        tokens=1234,
        cost_usd=0.5,
        max_nodes=3,
        run_id="run-x",
    )

    assert len(payload["nodes"]) == 3
    kept = {node["id"] for node in payload["nodes"]}
    assert all(edge["source"] in kept and edge["target"] in kept for edge in payload["edges"])
    assert [item["node_id"] for item in payload["suggestions"]] == ["n1"]
    assert payload["meta"]["tokens"] == 1234
    assert payload["meta"]["cycle"] == 2
    # Root input/prompt badge: the node matching the target entity is flagged.
    root_nodes = [node for node in payload["nodes"] if node["root"]]
    assert len(root_nodes) == 1
    assert root_nodes[0]["label"] == "Sri Lestari"
    assert root_nodes[0]["level"] == 0
    assert all(node["level"] >= 1 for node in payload["nodes"] if not node["root"])


def test_vis_elements_builds_white_cards_and_styled_edges(tmp_path: Path) -> None:
    payload = build_payload(_store(tmp_path), target_entity="Sri Lestari")

    elements = vis_elements(payload)
    by_label = {node["label"]: node for node in elements["nodes"]}
    sri = by_label["Sri Lestari"]
    assert sri["color"]["background"] == "#FFFFFF"  # white node card
    assert sri["color"]["border"] == "#6366F1"  # root input/prompt accent
    assert sri["font"]["color"] == "#0F172A"  # dark slate typography
    assert sri["font"]["bold"] is True  # root anchor stays visually dominant
    assert sri["shape"] == "box"
    assert sri["level"] == 0  # explicit level-0 root, not edge-derived
    assert sri["borderWidth"] == 1  # thin accent border
    assert "\U0001F464" not in sri["label"]  # cards are text-only (no icon)

    # Non-root nodes use their entity accent badge.
    jaya = by_label["PT Jaya Abadi"]
    assert jaya["color"]["background"] == "#FFFFFF"
    assert jaya["color"]["border"] == "#F59E0B"  # perusahaan amber accent
    assert jaya["level"] == 1  # initial-search fact
    dinas = by_label["Dinas Kesehatan Sleman"]
    assert dinas["color"]["border"] == "#F59E0B"  # instansi amber accent
    assert by_label["Puskesmas Depok"]["level"] == 2  # pivot round 1

    assert payload["meta"]["cross_edges"] >= 1
    assert elements["edges"], "expected at least one edge"
    first = elements["edges"][0]
    assert first["id"] == "e0"
    assert first["from"] and first["to"]
    assert first["color"]["color"] in {*EDGE_COLORS.values(), "#CBD5E1"}
    assert isinstance(first["dashes"], bool)

    cross = [edge for edge in elements["edges"] if edge["cross"]]
    assert cross, "expected at least one backward cross-edge"
    assert all(edge["hidden"] for edge in cross)  # hidden by default
    assert all(edge["color"]["color"] == "#CBD5E1" for edge in cross)


def test_render_graph_html_is_prd_compliant(tmp_path: Path) -> None:
    store = _store(tmp_path)
    out = tmp_path / "runs" / "r1" / "graph_interactive.html"

    path = render_graph_html(
        store,
        out,
        target_entity="Sri Lestari",
        context="Sleman",
        critic_suggestions=[
            SuggestedNode(
                node_id="n2", label="Dinas Kesehatan Sleman", reason="pusat relasi"
            ).model_dump()
        ],
        cycle=1,
        tokens=987,
        cost_usd=0.0004,
        run_id="r1",
    )

    assert path is not None and path.exists()
    html = path.read_text(encoding="utf-8")
    assert path.stat().st_size < 1_500_000  # PRD performance budget

    # Clean Minimalist Light Theme tokens.
    for token in ("#F8FAFC", "#FFFFFF", "#E2E8F0", "#CBD5E1", "#0F172A", "20px 20px"):
        assert token in html
    assert "rgba(0,0,0,0.08)" in html  # clean box shadow
    assert "Inter" in html and "Fira Code" in html
    assert "#1E293B" not in html  # dark panel tokens removed
    for badge in NODE_BADGES.values():
        for value in badge.values():
            assert value in html

    # Horizontal hierarchical mindmap (vis.js) + cubicBezier smooth curves.
    assert "hierarchical" in html
    assert '"direction": "LR"' in html or "direction: 'LR'" in html or "'LR'" in html
    assert "nodeSpacing" in html and "180" in html
    assert "levelSeparation" in html and "350" in html
    assert '"level"' in html  # explicit node levels (0 root, 1 initial, 2+ pivot)
    assert "cubicBezier" in html
    assert '"physics"' in html
    assert "multiselect" in html

    # Selection feedback + live pivot counter + cross-edge toggle.
    assert "#3B82F6" in html  # selected background/border
    assert "#60A5FA" in html  # glow ring
    assert "drx-pivot-count" in html
    assert "node terpilih" in html
    assert "drx-btn-cross" in html
    assert "applySelectionStyles" in html
    assert "drx-btn-ready" in html

    # Click-to-toggle multi-select + selection tray + pivot confirmation modal.
    assert "toggleSelected" in html
    assert "drx-tray-count" in html
    assert "drx-clear-selection" in html
    assert "drx-modal-confirm" in html
    assert "Konfirmasi Pivot" in html
    assert "multiselect\": false" in html or '"multiselect": false' in html
    # Remaining token/URL/time budget is rendered in the toolbar stats.
    assert "tokens_remaining" in html
    assert "urls_remaining" in html
    assert "seconds_remaining" in html

    # interaction surface + action bar
    for label in (
        "Pivot Selected Nodes",
        "Expand Document Pool",
        "Stop &amp; Generate Final Report",
        "Inspeksi Node",
        "Saran Node (Critic)",
    ):
        assert label in html
    assert "[n2]" in html or "n2" in html

    # provenance inspector content
    assert "Sri Lestari" in html
    assert "Dinas Kesehatan Sleman" in html
    assert "https://a.id" in html
    assert "Verified" in html
    assert "drx-sidebar" in html
    assert "vis-network" in html
    # CLI mode: buttons point the user back to the terminal.
    assert "const DRX_WEB = false;" in html
    assert "/api/pivot" in html  # endpoint code present but gated by DRX_WEB


def test_render_graph_html_web_app_mode_wires_rest_endpoints(tmp_path: Path) -> None:
    store = _store(tmp_path)
    out = tmp_path / "graph_interactive.html"

    path = render_graph_html(
        store,
        out,
        target_entity="Sri Lestari",
        critic_suggestions=[
            SuggestedNode(node_id="n2", label="Dinas Kesehatan", reason="r").model_dump()
        ],
        web_api=True,
    )

    assert path is not None
    html = path.read_text(encoding="utf-8")
    assert "const DRX_WEB = true;" in html
    assert "/api/pivot" in html
    assert "/api/expand" in html
    assert "/api/stop" in html
    assert "/api/graph" in html
    assert "setInterval" in html
    assert '"api": true' in html


def test_render_falls_back_to_static_html(
    tmp_path: Path, monkeypatch
) -> None:
    store = _store(tmp_path)

    def boom():
        raise RuntimeError("pyvis tidak tersedia")

    monkeypatch.setattr(graph_view, "_load_pyvis", boom)
    out = tmp_path / "graph_interactive.html"

    path = render_graph_html(store, out, target_entity="Sri Lestari")

    assert path is not None and path.exists()
    html = path.read_text(encoding="utf-8")
    assert "Kanvas statis" in html
    assert "Pivot Selected Nodes" in html
    assert "Sri Lestari" in html
    assert NODE_COLORS["individu"] in html
    assert "#F8FAFC" in html  # light theme survives the fallback


def test_build_payload_carries_budget_and_pivot_history(tmp_path: Path) -> None:
    payload = build_payload(
        _store(tmp_path),
        target_entity="Sri Lestari",
        budget={
            "tokens": 1200,
            "max_tokens": 150000,
            "tokens_remaining": 148800,
            "urls": 12,
            "max_urls": 200,
            "urls_remaining": 188,
            "elapsed_seconds": 42.5,
            "max_runtime_seconds": 600,
            "seconds_remaining": 557.5,
            "overrides": [],
        },
        pivoted_nodes=["n1"],
    )

    assert payload["meta"]["budget"]["tokens_remaining"] == 148800
    assert payload["meta"]["pivoted_nodes"] == ["n1"]
    flagged = [node for node in payload["nodes"] if node["pivoted"]]
    assert flagged and flagged[0]["id"] == "n1"


def test_vis_elements_marks_pivoted_nodes_in_tooltip(tmp_path: Path) -> None:
    payload = build_payload(
        _store(tmp_path),
        target_entity="Sri Lestari",
        pivoted_nodes=["n1"],
    )
    elements = vis_elements(payload)
    node = next(item for item in elements["nodes"] if item["id"] == "n1")
    assert "sudah pernah dipivot" in node["title"]
