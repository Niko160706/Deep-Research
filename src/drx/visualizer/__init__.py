"""Interactive canvas renderer (PRD_design_new.md — clean minimalist light).

Public API: :func:`render_graph_html` writes ``runs/<run_id>/graph_interactive.html``
after every search cycle. The layout is a horizontal hierarchical mindmap
(Left-to-Right, ``cubicBezier`` edges, physics off); PyVis/vis.js loads from a
CDN so the HTML bundle stays far below the 1.5 MB / 50 MB browser-RAM targets.
When PyVis is missing a styled static fallback is emitted instead (the run
never breaks). Mode Web App (``web_api=True``) wires the action bar to the
local REST endpoints (``/api/pivot``, ``/api/expand``, ``/api/stop``).
"""
from __future__ import annotations

from drx.visualizer.graph_view import (
    BG,
    BORDER,
    EDGE_COLORS,
    GRID,
    NODE_BADGES,
    NODE_COLORS,
    NODE_ICONS,
    PANEL,
    TEXT,
    build_payload,
    edge_style,
    node_badge,
    node_color,
    render_graph_html,
    vis_elements,
)

__all__ = [
    "BG",
    "BORDER",
    "EDGE_COLORS",
    "GRID",
    "NODE_BADGES",
    "NODE_COLORS",
    "NODE_ICONS",
    "PANEL",
    "TEXT",
    "build_payload",
    "edge_style",
    "node_badge",
    "node_color",
    "render_graph_html",
    "vis_elements",
]
