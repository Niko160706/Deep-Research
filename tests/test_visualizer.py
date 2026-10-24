"""Tests for the PRD-compliant interactive canvas renderer.

The HTML must carry the exact design tokens from ``PRD_design.md`` (Dark Neon:
``#0B0F17`` canvas, ``#151C2C`` cards, ``#334155`` borders), the fixed 180 px
card / ``short_label`` contract, the vertical Top-to-Bottom hierarchy, and the
interaction surface (hover tooltip, 360 px inspector drawer, multi-select,
budget status row separate from the action buttons).
"""
from __future__ import annotations

from pathlib import Path

from drx.brain.critic import SuggestedNode
from drx.models import Fact, SourceRef
from drx.store.graph_store import GraphStore
from drx.visualizer import graph_view
from drx.visualizer.graph_view import (
    ACCENT,
    BG,
    BODY_TEXT,
    BORDER,
    EDGE_BASE,
    EDGE_COLORS,
    GRID,
    NODE_BADGES,
    NODE_COLORS,
    PANEL,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    TOOLTIP_BG,
    TOOLTIP_BORDER,
    build_payload,
    edge_style,
    node_badge,
    node_color,
    render_graph_html,
    vis_elements,
)


def _fact(
    subject: str,
    predicate: str,
    obj: str,
    urls: list[str],
    *,
    subject_short_label: str = "",
    object_short_label: str = "",
) -> Fact:
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
        subject_short_label=subject_short_label,
        object_short_label=object_short_label,
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
                object_short_label="Dinkes Sleman",
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


def test_palette_matches_dark_neon_prd() -> None:
    assert (BG, PANEL, BORDER, GRID, SURFACE, TEXT) == (
        "#0B0F17",
        "#151C2C",
        "#334155",
        "#1E293B",
        "#0F172A",
        "#F8FAFC",
    )
    assert (TEXT_MUTED, BODY_TEXT, TOOLTIP_BG, TOOLTIP_BORDER, ACCENT) == (
        "#94A3B8",
        "#CBD5E1",
        "#1E293B",
        "#475569",
        "#38BDF8",
    )
    assert NODE_BADGES["individu"] == {
        "bg": "#2E1065",
        "border": "#581C87",
        "text": "#C084FC",
        "label": "INDIVIDU",
    }
    assert NODE_BADGES["instansi"] == {
        "bg": "#451A03",
        "border": "#78350F",
        "text": "#FCD34D",
        "label": "INSTANSI",
    }
    assert NODE_BADGES["aktor"] == {
        "bg": "#450A0A",
        "border": "#7F1D1D",
        "text": "#FCA5A5",
        "label": "AKTOR",
    }
    assert NODE_BADGES["verified"] == {
        "bg": "#022C22",
        "border": "#064E3B",
        "text": "#6EE7B7",
        "label": "VERIFIED",
    }
    assert NODE_BADGES["root"]["text"] == "#38BDF8"
    assert NODE_BADGES["perusahaan"] == NODE_BADGES["instansi"]
    assert NODE_BADGES["peristiwa"] == NODE_BADGES["aktor"]
    assert NODE_BADGES["dokumen"] == NODE_BADGES["verified"]
    # Backward-compatible accent map derives from the badge text color.
    assert NODE_COLORS["individu"] == "#C084FC"
    assert NODE_COLORS["instansi"] == "#FCD34D"
    assert NODE_COLORS["perusahaan"] == "#FCD34D"
    assert NODE_COLORS["dokumen"] == "#6EE7B7"
    assert NODE_COLORS["peristiwa"] == "#FCA5A5"
    assert node_color("individu") == NODE_COLORS["individu"]
    assert node_badge("perusahaan") == NODE_BADGES["instansi"]
    assert EDGE_COLORS == {
        "verified": "#22C55E",
        "single": "#94A3B8",
        "contradiction": "#EF4444",
        "base": "#334155",
    }
    assert EDGE_BASE == BORDER


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
    assert all(node["short_label"] for node in payload["nodes"])


def test_short_labels_are_display_only_and_preserve_full_labels(tmp_path: Path) -> None:
    store = _store(tmp_path)
    payload = build_payload(store, target_entity="Sri Lestari")
    records = {record["label"]: record for record in payload["nodes"]}

    assert records["Dinas Kesehatan Sleman"]["short_label"] == "Dinkes Sleman"
    assert records["Dinas Kesehatan Sleman"]["label"] == "Dinas Kesehatan Sleman"
    assert records["Puskesmas Depok"]["short_label"] == "Puskesmas Depok"
    assert "grounding" in records["Dinas Kesehatan Sleman"]
    assert "critic_reason" in records["Dinas Kesehatan Sleman"]
    for record in records.values():
        assert 1 <= len(record["short_label"].split()) <= 3
        assert len(record["short_label"]) <= 20

    elements = vis_elements(payload)
    by_full = {node["full_label"]: node for node in elements["nodes"]}
    dinas = by_full["Dinas Kesehatan Sleman"]
    # Canvas label is short; the full label remains in the payload.
    assert dinas["label"] == "Dinkes Sleman"
    assert dinas["full_label"] == "Dinas Kesehatan Sleman"
    assert dinas["short_label"] == "Dinkes Sleman"


def test_vis_elements_builds_dark_neon_cards_and_styled_edges(tmp_path: Path) -> None:
    payload = build_payload(_store(tmp_path), target_entity="Sri Lestari")

    elements = vis_elements(payload)
    by_full = {node["full_label"]: node for node in elements["nodes"]}
    sri = by_full["Sri Lestari"]
    assert sri["label"] == "Sri Lestari"
    assert sri["color"]["background"] == PANEL  # dark card fill #151C2C
    assert sri["color"]["border"] == BORDER  # default 1px #334155 border
    assert sri["font"]["color"] == TEXT
    assert sri["font"]["size"] == 12
    assert sri["font"]["bold"] is False
    assert sri["shape"] == "box"
    assert sri["level"] == 0  # explicit level-0 root, not edge-derived
    assert sri["borderWidth"] == 1
    assert sri["borderWidthSelected"] == 2
    assert sri["widthConstraint"] == {"minimum": 180, "maximum": 180}
    assert "heightConstraint" not in sri  # vis 9.1.x rejects maximum; custom renderer owns 54px
    assert sri["shapeProperties"] == {"borderRadius": 8}
    assert "\U0001F464" not in sri["label"]  # cards are text-only (no icon)

    # Every card uses the same neutral border; type color lives in the badge.
    jaya = by_full["PT Jaya Abadi"]
    assert jaya["color"]["background"] == PANEL
    assert jaya["color"]["border"] == BORDER
    assert jaya["level"] == 1  # initial-search fact
    dinas = by_full["Dinas Kesehatan Sleman"]
    assert dinas["color"]["border"] == BORDER
    assert by_full["Puskesmas Depok"]["level"] == 2  # pivot round 1

    assert payload["meta"]["cross_edges"] >= 1
    assert elements["edges"], "expected at least one edge"
    first = elements["edges"][0]
    assert first["id"] == "e0"
    assert first["from"] and first["to"]
    assert first["color"]["color"] in {*EDGE_COLORS.values()}
    assert first["width"] == 1.5
    assert isinstance(first["dashes"], bool)

    cross = [edge for edge in elements["edges"] if edge["cross"]]
    assert cross, "expected at least one backward cross-edge"
    assert all(edge["hidden"] for edge in cross)  # hidden by default
    assert all(edge["color"]["color"] == EDGE_BASE for edge in cross)


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

    # Dark Neon tokens, dot-grid, typography.
    for token in (
        BG,
        PANEL,
        BORDER,
        GRID,
        SURFACE,
        TEXT,
        TEXT_MUTED,
        BODY_TEXT,
        TOOLTIP_BG,
        TOOLTIP_BORDER,
        ACCENT,
    ):
        assert token in html
    assert "radial-gradient" in html and "20px 20px" in html
    assert "Inter" in html and "Fira Code" in html
    for color in ("#C084FC", "#FCD34D", "#FCA5A5", "#6EE7B7", "#38BDF8"):
        assert color in html
    for badge in NODE_BADGES.values():
        for value in badge.values():
            assert value in html

    # Vertical top-to-bottom hierarchy with the exact PRD spacing.
    assert "hierarchical" in html
    assert '"direction": "UD"' in html
    assert '"nodeSpacing": 180' in html
    assert '"levelSeparation": 130' in html
    assert '"treeSpacing": 200' in html
    assert '"physics": {"enabled": false}' in html
    assert '"forceDirection": "vertical"' in html
    assert "cubicBezier" in html
    assert '"direction": "LR"' not in html
    assert '"level"' in html  # explicit node levels (0 root, 1 initial, 2+ pivot)
    assert "multiselect" in html
    assert "anchorRootTop" in html  # level-0 root pinned to top center

    # Fixed-width cards + short-label cards + custom canvas renderer.
    assert '"label": "Dinkes Sleman"' in html
    assert '"full_label": "Dinas Kesehatan Sleman"' in html
    assert '"short_label": "Dinkes Sleman"' in html
    assert '"widthConstraint": {"minimum": 180, "maximum": 180}' in html
    assert '"shapeProperties": {"borderRadius": 8}' in html
    assert "ctxRenderer" in html
    assert 'shape: "custom"' in html

    # Hover tooltip (200ms) and click/selectNode inspector drawer.
    assert "hoverNode" in html
    assert "blurNode" in html
    assert "selectNode" in html
    assert "drx-tooltip" in html
    assert "drx-tooltip-title" in html
    assert "drx-tooltip-summary" in html
    assert "drx-tooltip-meta" in html
    assert "}, 200);" in html or "200)" in html
    assert "Bukti Grounding" in html
    assert "URL Sumber" in html
    assert "Saran Critic" in html
    assert "width: 360px" in html

    # Header fix: action buttons first, then token/URL/time stats, then status.
    actions = html.index("drx-toolbar-actions")
    stats = html.index('class="drx-stats"')
    status = html.index('class="drx-status"')
    assert actions < stats < status
    assert html.count('id="drx-status"') == 1
    assert "top: 64px" not in html

    # Selection feedback + live pivot counter + cross-edge toggle.
    assert ACCENT in html
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
    assert "Dinkes Sleman" in html
    assert NODE_COLORS["instansi"] in html
    assert BG in html
    assert "radial-gradient" in html


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


def test_vis_elements_marks_pivoted_nodes(tmp_path: Path) -> None:
    payload = build_payload(
        _store(tmp_path),
        target_entity="Sri Lestari",
        pivoted_nodes=["n1"],
    )
    elements = vis_elements(payload)
    node = next(item for item in elements["nodes"] if item["id"] == "n1")
    assert node["pivoted"] is True
    assert node["label"]
