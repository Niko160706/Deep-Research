"""Interactive canvas renderer (PRD_design.md — Dark Neon).

Public API: :func:`render_graph_html` writes ``runs/<run_id>/graph_interactive.html``
after every search cycle. The layout is a vertical hierarchical mindmap
(Top-to-Bottom, fixed 180 px cards, ``cubicBezier`` edges, physics off);
PyVis/vis.js loads from a CDN so the HTML bundle stays far below the 1.5 MB /
50 MB browser-RAM targets. Cards show ``short_label`` only; a custom hover
tooltip and 360 px inspector drawer expose the full label, grounding quotes,
source URLs, and critic suggestions. When PyVis is missing a styled static
fallback is emitted instead (the run never breaks). Mode Web App
(``web_api=True``) wires the action bar to the local REST endpoints
(``/api/pivot``, ``/api/expand``, ``/api/stop``).
"""
from __future__ import annotations

from drx.visualizer.graph_view import (
    ACCENT,
    BG,
    BODY_TEXT,
    BORDER,
    EDGE_COLORS,
    GRID,
    NODE_BADGES,
    NODE_COLORS,
    NODE_ICONS,
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

__all__ = [
    "ACCENT",
    "BG",
    "BODY_TEXT",
    "BORDER",
    "EDGE_COLORS",
    "GRID",
    "NODE_BADGES",
    "NODE_COLORS",
    "NODE_ICONS",
    "PANEL",
    "SURFACE",
    "TEXT",
    "TEXT_MUTED",
    "TOOLTIP_BG",
    "TOOLTIP_BORDER",
    "build_payload",
    "edge_style",
    "node_badge",
    "node_color",
    "render_graph_html",
    "vis_elements",
]
