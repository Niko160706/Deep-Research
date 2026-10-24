"""PyVis/vis.js renderer — strict compliance with ``PRD_design.md``.

Dark Neon Theme:

* Canvas background ``#0B0F17`` with a ``#1E293B`` dot-grid at 20 px spacing.
* Fixed-width 180 px node cards (``#151C2C``, 1 px ``#334155`` border,
  8 px radius). The visible title is ``short_label`` only (1-3 words, max
  20 chars); the full ``label`` lives in the hover tooltip + inspector drawer.
* Badge colors: INDIVIDU ``#C084FC``, INSTANSI ``#FCD34D``, AKTOR ``#FCA5A5``,
  VERIFIED ``#6EE7B7``, KUERI ``#38BDF8``.

Layout wajib: **Hierarchical Top-to-Bottom** (``direction: 'UD'``,
``sortMethod: 'directed'``, ``levelSeparation: 130``, ``nodeSpacing: 180``,
``treeSpacing: 200``, physics off) dengan edge ``cubicBezier``. Setiap node
membawa atribut ``level`` eksplisit per cabang (0 root, 1 pencarian
awal/Expand, lalu ``parent.level + 1`` untuk setiap pivot, tanpa batas level).
Cross-edge (relasi mundur ke level lebih awal) disembunyikan by default dan
bisa ditampilkan lewat toggle toolbar.

Interaction surface from ``PRD_design.md`` tetap berlaku: action bar
Pivot/Expand/Stop dengan status token/URL/waktu di baris terpisah, seleksi
**klik-toggle** multi-node, tray ``ID · label · level`` + penanda sudah-dipivot,
modal konfirmasi sebelum satu batch POST, hover tooltip 200 ms, dan Inspector
Drawer 360 px (ID, tipe, grounding quotes, URL sumber, saran critic).

Web App mode (``web_api=True``): tombol action bar POST ke server lokal
(``/api/pivot``, ``/api/expand``, ``/api/stop``) dan halaman polling
``/api/graph`` sehingga mindmap + statistik token/URL/waktu ter-update
real-time. CLI mode menampilkan petunjuk terminal seperti sebelumnya. Render
tidak pernah raise ke loop riset.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from drx.textutil import compact_short_label

log = logging.getLogger("drx.visualizer")

# ---- PRD_design.md design tokens (Dark Neon) ---------------------------------

BG = "#0B0F17"  # near-black canvas
PANEL = "#151C2C"  # node card / panel fill
SURFACE = "#0F172A"  # header + inspector drawer
BORDER = "#334155"  # default card border
GRID = "#1E293B"  # dot grid
TEXT = "#F8FAFC"  # primary text
TEXT_MUTED = "#94A3B8"  # header status / muted text
BODY_TEXT = "#CBD5E1"  # inspector body
TOOLTIP_BG = "#1E293B"
TOOLTIP_BORDER = "#475569"
SHADOW = "rgba(0,0,0,0.35)"
ACCENT = "#38BDF8"  # selected border / active edge
SELECTED_GLOW = "rgba(56,189,248,0.35)"
EDGE_BASE = "#334155"

#: Accent badges (bg / border / text / label) — PRD_design.md §3 / §4.
NODE_BADGES: dict[str, dict[str, str]] = {
    "root": {"bg": "#082F49", "border": "#0369A1", "text": "#38BDF8", "label": "KUERI"},
    "query": {"bg": "#082F49", "border": "#0369A1", "text": "#38BDF8", "label": "KUERI"},
    "individu": {"bg": "#2E1065", "border": "#581C87", "text": "#C084FC", "label": "INDIVIDU"},
    "instansi": {"bg": "#451A03", "border": "#78350F", "text": "#FCD34D", "label": "INSTANSI"},
    "perusahaan": {"bg": "#451A03", "border": "#78350F", "text": "#FCD34D", "label": "INSTANSI"},
    "aktor": {"bg": "#450A0A", "border": "#7F1D1D", "text": "#FCA5A5", "label": "AKTOR"},
    "peristiwa": {"bg": "#450A0A", "border": "#7F1D1D", "text": "#FCA5A5", "label": "AKTOR"},
    "verified": {"bg": "#022C22", "border": "#064E3B", "text": "#6EE7B7", "label": "VERIFIED"},
    "dokumen": {"bg": "#022C22", "border": "#064E3B", "text": "#6EE7B7", "label": "VERIFIED"},
    "lainnya": {"bg": "#1E293B", "border": "#475569", "text": "#94A3B8", "label": "LAINNYA"},
}

#: Backward-compatible accent (text) color map per node type.
NODE_COLORS: dict[str, str] = {
    key: value["text"] for key, value in NODE_BADGES.items()
}

NODE_ICONS: dict[str, str] = {
    "individu": "\U0001F464",  # person
    "instansi": "\U0001F3DB",  # building
    "perusahaan": "\U0001F3E2",  # office
    "dokumen": "\U0001F4C4",  # file-text
    "peristiwa": "\U0001F6A9",  # flag
    "lainnya": "\u25CF",
}

EDGE_COLORS: dict[str, str] = {
    "verified": "#22C55E",  # solid green (>= 2 independent sources)
    "single": "#94A3B8",  # dashed slate (1 source) — PRD §2.1 edge base
    "contradiction": "#EF4444",  # solid red (flagged by critic / conflict)
    "base": EDGE_BASE,  # neutral / cross-edge line
}


def node_badge(node_type: str, *, root: bool = False) -> dict[str, str]:
    """Accent badge for one node (root/target wins over its entity type)."""
    key = "root" if root else (node_type or "lainnya").lower()
    return NODE_BADGES.get(key, NODE_BADGES["lainnya"])


def node_color(node_type: str) -> str:
    """Accent text color for a node type (same value as ``NODE_COLORS``)."""
    return node_badge(node_type)["text"]


def edge_style(record: dict) -> dict:
    """Return ``{color, dashes}`` for one edge record (PRD_design.md §2.1.2)."""
    if record.get("contradiction"):
        return {"color": EDGE_COLORS["contradiction"], "dashes": False}
    if record.get("verified") or int(record.get("source_count") or 0) >= 2:
        return {"color": EDGE_COLORS["verified"], "dashes": False}
    return {"color": EDGE_COLORS["single"], "dashes": True}


# ---- payload assembly -----------------------------------------------------------


def _norm_label(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def build_payload(
    store: Any,
    *,
    target_entity: str = "",
    context: str = "",
    critic_suggestions: Sequence[dict] = (),
    cycle: int = 0,
    tokens: int = 0,
    cost_usd: float = 0.0,
    max_nodes: int = 200,
    run_id: str = "",
    budget: dict | None = None,
    pivoted_nodes: Sequence[str] = (),
) -> dict:
    """Collect node/edge/suggestion records for the canvas (pure Python)."""
    nodes = store.node_records() if store is not None else []
    edges = store.edge_records() if store is not None else []

    if max_nodes and len(nodes) > max_nodes:
        nodes.sort(key=lambda n: (-int(n.get("degree") or 0), str(n.get("id"))))
        nodes = nodes[:max_nodes]
    keep_ids = {str(n["id"]) for n in nodes}
    edges = [
        e
        for e in edges
        if str(e.get("source")) in keep_ids and str(e.get("target")) in keep_ids
    ]

    # Root input / prompt (PRD_design_new.md §1.1): node yang cocok dengan
    # target ATAU yang ditandai root oleh GraphStore.ensure_root().
    target_key = _norm_label(target_entity)
    pivoted = {str(node_id) for node_id in pivoted_nodes or ()}
    for node in nodes:
        matched = bool(target_key and _norm_label(node.get("label")) == target_key)
        node["root"] = bool(node.get("root") or matched)
        if node["root"]:
            node["level"] = 0
        else:
            node["level"] = max(1, int(node.get("level") or 1))
        node["pivoted"] = str(node.get("id")) in pivoted
        # Defensive normalization for legacy/custom stores: every canvas node
        # exposes a short_label, while the full label stays untouched.
        node["short_label"] = compact_short_label(
            str(node.get("short_label") or ""),
            fallback=str(node.get("label") or ""),
        )
        node.setdefault("summary", "")
        node.setdefault("grounding", list(node.get("quotes") or []))
        node.setdefault("critic_reason", "")
        node.setdefault("created_at", 0.0)

    # Cross-edge = relasi mundur (target berada di kolom level lebih awal).
    # Alur utama kiri->kanan tetap dominan; edge ini disembunyikan by default
    # dan bisa ditampilkan lewat toggle di toolbar.
    levels = {str(node["id"]): int(node.get("level") or 0) for node in nodes}
    for edge in edges:
        edge["cross"] = levels.get(str(edge.get("target")), 1) < levels.get(
            str(edge.get("source")), 1
        )
    cross_edges = sum(1 for edge in edges if edge.get("cross"))

    suggestions: list[dict] = []
    for item in critic_suggestions or ():
        node_id = str(item.get("node_id") or "")
        if not node_id or (keep_ids and node_id not in keep_ids):
            continue
        suggestions.append(
            {
                "node_id": node_id,
                "label": str(item.get("label") or ""),
                "reason": str(item.get("reason") or ""),
            }
        )
    return {
        "meta": {
            "run_id": run_id,
            "target": target_entity,
            "context": context,
            "cycle": int(cycle),
            "tokens": int(tokens),
            "cost_usd": float(cost_usd),
            "nodes": len(nodes),
            "edges": len(edges),
            "cross_edges": cross_edges,
            "pivoted_nodes": sorted(pivoted),
            "budget": dict(budget or {}),
        },
        "nodes": nodes,
        "edges": edges,
        "suggestions": suggestions,
    }


def vis_elements(payload: dict) -> dict:
    """vis.js-ready nodes/edges (initial render + ``/api/graph`` polling).

    Cards use the full 180 px width: ``short_label`` only is drawn on the
    canvas, while ``full_label`` / ``summary`` stay in the payload for the
    hover tooltip and inspector drawer. Every node carries an explicit
    hierarchical ``level`` (0 root, 1 initial search/Expand, then
    ``parent.level + 1`` per pivot branch) so the UD tree is stable without
    edge inference.
    """
    nodes: list[dict] = []
    for node in payload.get("nodes", []):
        level = max(0, int(node.get("level") or 0))
        short_label = compact_short_label(
            str(node.get("short_label") or ""),
            fallback=str(node.get("label") or ""),
        )
        nodes.append(
            {
                "id": str(node.get("id")),
                # Vis.js draws this inside the fixed 180px card. The full
                # label is never rendered on-canvas (PRD §4).
                "label": short_label,
                "full_label": str(node.get("label", "")),
                "short_label": short_label,
                "summary": str(node.get("summary") or ""),
                "pivoted": bool(node.get("pivoted")),
                "shape": "box",
                "level": level,
                "widthConstraint": {"minimum": 180, "maximum": 180},
                "shapeProperties": {"borderRadius": 8},
                "borderWidth": 1,
                "borderWidthSelected": 2,
                "color": {
                    "background": PANEL,
                    "border": BORDER,
                    "highlight": {"background": PANEL, "border": ACCENT},
                    "hover": {"background": PANEL, "border": ACCENT},
                },
                "font": {
                    "color": TEXT,
                    "size": 12,
                    "bold": False,
                    "face": (
                        "Inter, -apple-system, BlinkMacSystemFont, "
                        "Segoe UI, Roboto, sans-serif"
                    ),
                },
                "margin": {"top": 8, "bottom": 8, "left": 10, "right": 10},
                "shadow": {
                    "enabled": True,
                    "color": SHADOW,
                    "size": 8,
                    "x": 0,
                    "y": 3,
                },
            }
        )
    edges: list[dict] = []
    for index, edge in enumerate(payload.get("edges", [])):
        cross = bool(edge.get("cross"))
        style = edge_style(edge)
        status = (
            "kontradiksi"
            if edge.get("contradiction")
            else ("terverifikasi" if edge.get("verified") else "1 sumber")
        )
        title = (
            f"{edge.get('predicate', '')}<br>status: {status}<br>"
            f"sumber: {edge.get('source_count', 0)}"
            + ("<br>cross-edge (relasi mundur)" if cross else "")
        )
        edges.append(
            {
                "id": f"e{index}",
                "from": str(edge.get("source")),
                "to": str(edge.get("target")),
                "label": str(edge.get("predicate") or ""),
                "title": title,
                "hidden": cross,
                "cross": cross,
                "color": {
                    "color": EDGE_BASE if cross else style["color"],
                    "highlight": ACCENT,
                },
                "dashes": True if cross else bool(style["dashes"]),
                "width": 1.5,
                "font": {
                    "color": TEXT_MUTED,
                    "size": 10,
                    "face": "Inter, system-ui, sans-serif",
                    "strokeWidth": 3,
                    "strokeColor": BG,
                },
            }
        )
    return {"nodes": nodes, "edges": edges}


# ---- HTML assembly ---------------------------------------------------------------


def _data_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")


_CSS_TEMPLATE = """
:root {
  --drx-bg: __BG__; --drx-panel: __PANEL__; --drx-surface: __SURFACE__;
  --drx-border: __BORDER__; --drx-grid: __GRID__; --drx-text: __TEXT__;
  --drx-muted: __TEXT_MUTED__; --drx-body: __BODY_TEXT__;
  --drx-accent: __ACCENT__; --drx-tooltip: __TOOLTIP_BG__;
  --drx-tooltip-border: __TOOLTIP_BORDER__;
}
html, body {
  background-color: __BG__ !important;
  background-image: radial-gradient(__GRID__ 1.5px, transparent 1.5px);
  background-size: 20px 20px;
  color: __TEXT__;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 0;
}
#mynetwork, .card, .vis-network { background: transparent !important; }
.card { border: 0 !important; box-shadow: none !important; margin: 0 !important; padding: 0 !important; }
#mynetwork { border: 1px solid __GRID__; border-radius: 12px; }
.drx-mono, code {
  font-family: "Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace;
}
.drx-toolbar {
  position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
  display: flex; flex-direction: column; align-items: stretch; gap: 6px;
  z-index: 40; padding: 10px 12px; max-width: 96vw; box-sizing: border-box;
  background: __SURFACE__; border: 1px solid __GRID__; border-radius: 12px;
  backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px);
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.drx-toolbar-actions {
  display: flex; flex-wrap: wrap; align-items: center; justify-content: center;
  gap: 8px;
}
.drx-btn {
  border: 1px solid __BORDER__; border-radius: 8px; cursor: pointer;
  padding: 8px 12px; font-size: 12px; font-weight: 600; line-height: 1;
  letter-spacing: 0.01em; color: #FFFFFF; background: __PANEL__;
  transition: filter .15s ease, background .15s ease, border-color .15s ease;
}
.drx-btn:hover { filter: brightness(1.12); }
.drx-btn-pivot { background: #2563EB; border-color: #2563EB; }
.drx-btn-expand { background: #0D9488; border-color: #0D9488; }
.drx-btn-stop { background: #DC2626; border-color: #DC2626; }
.drx-btn-cross { background: __PANEL__; border-color: __BORDER__; color: __TEXT_MUTED__; }
.drx-btn-cross.drx-btn-active { background: #082F49; border-color: #0369A1; color: #38BDF8; }
.drx-btn-ready { box-shadow: 0 0 0 3px rgba(56,189,248,0.28); }
#drx-pivot-count { font-weight: 800; }
.drx-stats, .drx-status {
  width: 100%; box-sizing: border-box; margin-top: 6px; text-align: center;
  font-size: 11px; font-weight: 400; line-height: 1.2; letter-spacing: 0.02em;
  color: __TEXT_MUTED__; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.drx-status { margin-top: 0; min-height: 13px; }
.drx-tray {
  display: none; position: fixed; top: 120px; left: 14px; z-index: 41;
  width: 330px; max-width: 44vw; max-height: 34vh; overflow-y: auto;
  background: __SURFACE__; border: 1px solid __GRID__; border-radius: 12px;
  padding: 10px 12px; box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.drx-tray.drx-open { display: block; }
.drx-tray-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; margin-bottom: 6px; font-size: 12px; color: __TEXT__;
}
.drx-tray .drx-chip { cursor: pointer; }
.drx-tray-empty { color: __TEXT_MUTED__; font-size: 12px; }
.drx-modal-backdrop {
  display: none; position: fixed; inset: 0; z-index: 60;
  background: rgba(2,6,23,0.72); align-items: center; justify-content: center;
  padding: 18px;
}
.drx-modal-backdrop.drx-open { display: flex; }
.drx-modal {
  width: 460px; max-width: 94vw; max-height: 80vh; overflow-y: auto;
  background: __SURFACE__; border: 1px solid __BORDER__; border-radius: 12px;
  padding: 18px; color: __BODY_TEXT__; box-shadow: 0 24px 60px rgba(0,0,0,0.55);
}
.drx-modal h3 { margin: 0 0 8px; font-size: 15px; line-height: 1.3; color: __TEXT__; }
.drx-modal-list { margin: 8px 0 14px; }
.drx-modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
.drx-modal-confirm { background: #2563EB; border-color: #2563EB; color: #FFFFFF; }
.drx-level-pill {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  background: __PANEL__; border: 1px solid __BORDER__; font-size: 10px;
  color: __TEXT_MUTED__;
}
.drx-sidebar {
  position: fixed; top: 0; right: 0; width: 360px; max-width: 100vw; height: 100vh;
  background: __SURFACE__; border-left: 1px solid __GRID__; z-index: 45;
  transform: translateX(102%); transition: transform .22s ease;
  overflow-y: auto; padding: 18px; box-sizing: border-box;
  box-shadow: -18px 0 44px rgba(0,0,0,0.45); color: __BODY_TEXT__;
  font-size: 12px; font-weight: 400; line-height: 1.5;
}
.drx-sidebar.drx-open { transform: translateX(0); }
.drx-drawer-head {
  display: flex; justify-content: space-between; align-items: flex-start;
  gap: 10px; margin-bottom: 12px;
}
.drx-sidebar h3 {
  margin: 0; font-size: 15px; font-weight: 600; line-height: 1.3;
  letter-spacing: 0; color: __TEXT__;
}
.drx-eyebrow {
  margin-bottom: 3px; font-size: 9px; font-weight: 600; line-height: 1;
  letter-spacing: 0.12em; color: __TEXT_MUTED__;
}
.drx-inspector-name {
  font-size: 15px; font-weight: 600; line-height: 1.3; color: __TEXT__;
  margin-bottom: 8px;
}
.drx-chip {
  display: inline-block; margin: 3px 4px 3px 0; padding: 3px 9px;
  border-radius: 999px; border: 1px solid #0369A1; background: #082F49;
  font-size: 11px; cursor: pointer; color: #38BDF8;
  font-family: "Fira Code", monospace;
}
.drx-chip:hover { border-color: #38BDF8; color: #BAE6FD; }
.drx-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 9px; font-weight: 600; line-height: 1; letter-spacing: 0.05em;
  border: 1px solid __BORDER__;
}
.drx-badge-verified { color: #6EE7B7; background: #022C22; border-color: #064E3B; }
.drx-badge-single { color: #94A3B8; background: #1E293B; border-color: #334155; }
.drx-badge-contradiction { color: #FCA5A5; background: #450A0A; border-color: #7F1D1D; }
.drx-card {
  background: __PANEL__; border: 1px solid __BORDER__; border-radius: 8px;
  padding: 10px; margin: 8px 0; font-size: 12px; font-weight: 400;
  line-height: 1.5; color: __BODY_TEXT__;
}
.drx-card > b { display: block; margin-bottom: 4px; color: __TEXT__; }
.drx-muted { color: __TEXT_MUTED__; font-size: 11px; line-height: 1.4; }
.drx-grounding {
  margin: 7px 0; padding: 2px 0 2px 10px; border-left: 3px solid __ACCENT__;
  color: __BODY_TEXT__; font-size: 12px; line-height: 1.5;
}
.drx-source { margin: 5px 0; overflow-wrap: anywhere; }
.drx-source a, a { color: __ACCENT__; text-decoration: none; }
.drx-source a:hover, a:hover { text-decoration: underline; }
.drx-source-host { color: __TEXT__; font-weight: 600; }
.drx-legend {
  position: fixed; left: 14px; bottom: 14px; z-index: 40; padding: 10px 12px;
  background: __SURFACE__; border: 1px solid __GRID__; border-radius: 12px;
  font-size: 11px; color: __TEXT_MUTED__;
  box-shadow: 0 12px 30px rgba(0,0,0,0.35);
}
.drx-legend span { display: inline-flex; align-items: center; gap: 6px; margin-right: 10px; }
.drx-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.drx-hint { display: none; margin-top: 10px; }
.drx-hint.drx-open { display: block; }
.drx-tooltip {
  display: none; position: fixed; z-index: 80; max-width: 320px;
  padding: 10px 12px; pointer-events: none;
  background: __TOOLTIP_BG__; border: 1px solid __TOOLTIP_BORDER__;
  border-radius: 8px; color: __TEXT__; font-size: 11px; line-height: 1.4;
  box-shadow: 0 14px 34px rgba(0,0,0,0.5);
}
.drx-tooltip.drx-open { display: block; }
.drx-tooltip-title {
  font-size: 12px; font-weight: 600; line-height: 1.3; color: __TEXT__;
  margin-bottom: 4px;
}
.drx-tooltip-summary { color: __BODY_TEXT__; margin-bottom: 6px; }
.drx-tooltip-meta { color: #6EE7B7; font-weight: 600; font-size: 11px; }
blockquote { color: __TEXT_MUTED__; }
@media (max-width: 760px) {
  .drx-toolbar { width: 94vw; }
  .drx-stats, .drx-status { white-space: normal; }
  .drx-sidebar { width: 100vw; }
  .drx-tray { top: 140px; left: 3vw; width: 94vw; max-width: 94vw; max-height: 30vh; }
}
"""

def _theme_css() -> str:
    css = _CSS_TEMPLATE
    for token, value in (
        ("__BG__", BG),
        ("__PANEL__", PANEL),
        ("__SURFACE__", SURFACE),
        ("__BORDER__", BORDER),
        ("__GRID__", GRID),
        ("__TEXT_MUTED__", TEXT_MUTED),
        ("__TEXT__", TEXT),
        ("__BODY_TEXT__", BODY_TEXT),
        ("__ACCENT__", ACCENT),
        ("__TOOLTIP_BG__", TOOLTIP_BG),
        ("__TOOLTIP_BORDER__", TOOLTIP_BORDER),
    ):
        css = css.replace(token, value)
    return css


def _toolbar_html() -> str:
    return f"""
<div class="drx-toolbar" id="drx-toolbar">
  <div class="drx-toolbar-actions">
    <button class="drx-btn drx-btn-pivot" id="drx-btn-pivot" title="Batch query untuk semua node terpilih">
      \u26A1 Pivot Selected Nodes (<span id="drx-pivot-count">0</span> node terpilih)
    </button>
    <button class="drx-btn drx-btn-cross" id="drx-btn-cross" title="Tampilkan/sembunyikan relasi mundur (cross-edge)">
      \u2194 Cross (0)
    </button>
    <button class="drx-btn drx-btn-expand" id="drx-btn-expand" title="Cari gelombang dokumen level-1 baru (deep search + sudut baru)">
      \u2795 Expand Document Pool
    </button>
    <button class="drx-btn drx-btn-stop" id="drx-btn-stop" title="Akhiri pencarian &amp; susun laporan final">
      \u25A0 Stop &amp; Generate Final Report
    </button>
  </div>
  <div class="drx-stats" id="drx-stats"></div>
  <div class="drx-status" id="drx-status" role="status" aria-live="polite"></div>
</div>
<div class="drx-tray" id="drx-tray">
  <div class="drx-tray-head">
    <b>Node terpilih (<span id="drx-tray-count">0</span>)</b>
    <button class="drx-btn" id="drx-clear-selection" style="padding:3px 9px">Clear</button>
  </div>
  <div id="drx-tray-list" class="drx-tray-empty">Klik node untuk memilih; klik lagi untuk melepas.</div>
</div>
<div class="drx-modal-backdrop" id="drx-modal">
  <div class="drx-modal">
    <h3>Konfirmasi Pivot</h3>
    <div class="drx-muted">Periksa daftar ini. Satu klik akan mengirim semua node sekaligus.</div>
    <div class="drx-modal-list" id="drx-modal-list"></div>
    <div class="drx-modal-actions">
      <button class="drx-btn" id="drx-modal-cancel">Batal</button>
      <button class="drx-btn drx-modal-confirm" id="drx-modal-confirm">Kirim Pivot</button>
    </div>
  </div>
</div>
<div class="drx-legend" id="drx-legend">
  <div style="margin-bottom:6px;font-weight:600;color:{TEXT}">Legenda</div>
  <div>
    <span><i class="drx-dot" style="background:{NODE_BADGES['root']['bg']};border:2px solid {NODE_BADGES['root']['border']}"></i>Kueri</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['individu']['bg']};border:2px solid {NODE_BADGES['individu']['border']}"></i>Individu</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['instansi']['bg']};border:2px solid {NODE_BADGES['instansi']['border']}"></i>Instansi</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['aktor']['bg']};border:2px solid {NODE_BADGES['aktor']['border']}"></i>Aktor</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['verified']['bg']};border:2px solid {NODE_BADGES['verified']['border']}"></i>Verified</span>
  </div>
  <div style="margin-top:6px" class="drx-muted">
    <span style="border-top:3px solid {EDGE_COLORS['verified']};width:26px;display:inline-block"></span> terverifikasi
    <span style="border-top:3px dashed {EDGE_COLORS['single']};width:26px;display:inline-block;margin-left:8px"></span> 1 sumber
    <span style="border-top:3px solid {EDGE_COLORS['contradiction']};width:26px;display:inline-block;margin-left:8px"></span> kontradiksi
    <span style="border-top:2px dashed {EDGE_BASE};width:26px;display:inline-block;margin-left:8px"></span> cross-edge
  </div>
</div>
<div class="drx-tooltip" id="drx-tooltip" role="tooltip" aria-hidden="true"></div>
<div class="drx-sidebar" id="drx-sidebar">
  <div class="drx-drawer-head">
    <div>
      <div class="drx-eyebrow">INSPECTOR</div>
      <h3>Inspeksi Node</h3>
    </div>
    <button class="drx-btn" id="drx-close" style="padding:4px 9px" aria-label="Tutup inspector">\u2715</button>
  </div>
  <div id="drx-inspector">
    <p class="drx-muted">Klik node untuk membuka detail, atau klik untuk memilih/melepas (bisa banyak).</p>
  </div>
  <div id="drx-suggest-wrap">
    <h3>Saran Node (Critic)</h3>
    <div id="drx-suggestions" class="drx-muted">(tidak ada saran)</div>
  </div>
  <div id="drx-action-hint" class="drx-card drx-hint"></div>
</div>
"""

_JS_TEMPLATE = r"""
const DRX = __DRX_JSON__;
const DRX_WEB = __DRX_WEB__;
const DRX_API = "";
const DRX_BADGES = __DRX_BADGES__;
(function() {
  var $ = function(id) { return document.getElementById(id); };

  function esc(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function badgeHtml(badge, text) {
    return '<span class="drx-badge" style="background:' + badge.bg +
      ';border-color:' + badge.border + ';color:' + badge.text + '">' + text + '</span>';
  }
  function badgeKey(record) {
    var key = record.root ? "root" : String(record.type || "lainnya").toLowerCase();
    if (key === "perusahaan") return "instansi";
    if (key === "peristiwa") return "aktor";
    if (key === "dokumen") return "verified";
    return DRX_BADGES[key] ? key : "lainnya";
  }
  function badgeMeta(record) {
    return DRX_BADGES[badgeKey(record)] || DRX_BADGES.lainnya;
  }
  function typeBadge(record) {
    var badge = badgeMeta(record);
    return badgeHtml(badge, badge.label || badgeKey(record).toUpperCase());
  }
  function statusBadge(record) {
    if (record.contradiction) return '<span class="drx-badge drx-badge-contradiction">KONTRADIKSI</span>';
    if (record.verified) return '<span class="drx-badge drx-badge-verified">Verified (' + record.source_count + ' Sumber)</span>';
    return '<span class="drx-badge drx-badge-single">Belum terverifikasi (' + (record.source_count || 1) + ' sumber)</span>';
  }

  // ---- selection feedback + cross-edge toggle ------------------------------
  var DRX_SELECTED_BG = "#151C2C";
  var DRX_SELECTED_BORDER = "#38BDF8";
  var DRX_SELECTED_GLOW = "rgba(56,189,248,0.35)";
  var drxBaseStyles = {};
  var drxShowCross = false;
  var drxSelected = {};
  var drxPivoted = {};
  var drxSelectedSyncing = false;
  var drxSelectGuard = false;
  var drxModalIds = [];
  var drxCycle = (DRX.meta || {}).cycle || 0;
  var drxHoverTimer = null;
  var drxHoverNode = null;
  var drxPointer = { x: 0, y: 0 };
  var drxRootAnchored = false;
  var drxRootAnchorAttempts = 0;

  // Vis.js centers the complete hierarchy in the viewport. The PRD anchors the
  // level-0 root at the top center instead, so pan once after the first layout.
  function anchorRootTop() {
    if (drxRootAnchored || typeof network === "undefined" || !network) return;
    var root = null;
    (DRX.nodes || []).some(function(node) {
      if (node && node.root) { root = node; return true; }
      return false;
    });
    if (!root && DRX.nodes && DRX.nodes.length) root = DRX.nodes[0];
    if (!root) return;
    var positions = network.getPositions([root.id]) || {};
    var position = positions[root.id];
    if (!position || !isFinite(position.x) || !isFinite(position.y)) {
      if (drxRootAnchorAttempts++ < 6) setTimeout(anchorRootTop, 80);
      return;
    }
    var container = $("mynetwork");
    if (!container) return;
    var rect = container.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    var toolbar = $("drx-toolbar");
    var toolbarBottom = toolbar
      ? toolbar.getBoundingClientRect().bottom
      : rect.top + 80;
    var target = {
      x: rect.width / 2,
      y: toolbarBottom + 48 - rect.top
    };
    network.moveTo({
      position: { x: position.x, y: position.y },
      offset: { x: 0, y: 0 },
      scale: network.getScale(),
      animation: false
    });
    var center = network.canvasToDOM(
      network.getPositions([root.id])[root.id]
    );
    network.moveTo({
      position: { x: position.x, y: position.y },
      offset: { x: target.x - center.x, y: target.y - center.y },
      scale: network.getScale(),
      animation: false
    });
    drxRootAnchored = true;
  }

  function rememberNodeStyles(list) {
    (list || []).forEach(function(n) {
      drxBaseStyles[String(n.id)] = {
        color: n.color,
        borderWidth: n.borderWidth,
        font: n.font,
        shadow: n.shadow
      };
    });
  }

  function nodeRecord(id) {
    return (DRX.nodes || []).find(function(n) { return String(n.id) === String(id); });
  }

  function nodeShortLabel(record) {
    var value = String(record.short_label || record.label || "")
      .replace(/\s+/g, " ").trim();
    return value.length > 20 ? value.slice(0, 17).replace(/\s+$/, "") + "..." : value;
  }

  function drawRoundRect(ctx, x, y, width, height, radius) {
    var r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function makeNodeRenderer(record) {
    return function(options) {
      // vis-network calls drawNode() with no arguments; x/y/state are passed
      // to ctxRenderer itself (vis 9.1.x API).
      var x = Number(options.x || 0);
      var y = Number(options.y || 0);
      var state = options.state || {};
      return {
        nodeDimensions: { width: 180, height: 54 },
        drawNode: function() {
          var ctx = options.ctx;
          var selected = !!state.selected || !!drxSelected[String(record.id)];
          var hovered = !!state.hover;
          var left = x - 90;
          var top = y - 27;
          var badge = badgeMeta(record);

          ctx.save();
          drawRoundRect(ctx, left, top, 180, 54, 8);
          ctx.fillStyle = "#151C2C";
          if (selected || hovered) {
            ctx.shadowColor = selected
              ? "rgba(56,189,248,0.35)"
              : "rgba(56,189,248,0.22)";
            ctx.shadowBlur = selected ? 12 : 8;
            ctx.shadowOffsetY = 0;
          } else {
            ctx.shadowColor = "rgba(0,0,0,0.35)";
            ctx.shadowBlur = 8;
            ctx.shadowOffsetY = 3;
          }
          ctx.fill();
          ctx.shadowColor = "transparent";
          ctx.shadowBlur = 0;
          ctx.shadowOffsetY = 0;
          ctx.strokeStyle = selected ? "#38BDF8" : (hovered ? "#475569" : "#334155");
          ctx.lineWidth = selected ? 2 : 1;
          ctx.stroke();

          // Type badge (bg/border/text exactly from the PRD palette).
          var badgeText = String(badge.label || badgeKey(record).toUpperCase());
          ctx.font = "600 9px Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";
          var badgeWidth = Math.ceil(ctx.measureText(badgeText).width) + 16;
          var badgeHeight = 16;
          var badgeX = left + 10;
          var badgeY = top + 8;
          drawRoundRect(ctx, badgeX, badgeY, badgeWidth, badgeHeight, 8);
          ctx.fillStyle = badge.bg;
          ctx.fill();
          ctx.strokeStyle = badge.border;
          ctx.lineWidth = 1;
          ctx.stroke();
          ctx.fillStyle = badge.text;
          ctx.textAlign = "left";
          ctx.textBaseline = "middle";
          ctx.fillText(badgeText, badgeX + 8, badgeY + badgeHeight / 2 + 0.5);

          // Short title only: 1-3 words / max 20 chars.
          ctx.font = "500 12px Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif";
          ctx.fillStyle = "#F8FAFC";
          ctx.fillText(nodeShortLabel(record), left + 10, top + 38);

          if (record.pivoted) {
            ctx.font = "600 11px Inter, sans-serif";
            ctx.fillStyle = "#FCD34D";
            ctx.textAlign = "right";
            ctx.fillText("\u21BB", left + 170, top + 16);
          }
          if (record.verified && badgeKey(record) !== "verified") {
            ctx.font = "600 11px Inter, sans-serif";
            ctx.fillStyle = "#6EE7B7";
            ctx.textAlign = "right";
            ctx.fillText("\u2713", left + (record.pivoted ? 152 : 170), top + 16);
          }
          ctx.restore();
        }
      };
    };
  }

  function installNodeRenderers() {
    if (typeof nodes === "undefined" || !nodes || !nodes.update) return;
    try {
      nodes.update((DRX.nodes || []).map(function(record) {
        return {
          id: String(record.id),
          shape: "custom",
          ctxRenderer: makeNodeRenderer(record)
        };
      }));
    } catch (e) {}
  }

  function selectedIds() {
    return Object.keys(drxSelected).filter(function(id) {
      return !!nodeRecord(id);
    });
  }

  function setSelected(ids) {
    drxSelected = {};
    (ids || []).forEach(function(id) {
      if (nodeRecord(id)) drxSelected[String(id)] = true;
    });
  }

  function toggleSelected(id) {
    id = String(id);
    if (drxSelected[id]) delete drxSelected[id];
    else drxSelected[id] = true;
  }

  function syncNetworkSelection() {
    if (typeof network === "undefined" || !network) return;
    var ids = selectedIds();
    drxSelectedSyncing = true;
    drxSelectGuard = true;
    try {
      if (ids.length) network.selectNodes(ids);
      else network.unselectAll();
    } catch (e) {} finally {
      drxSelectedSyncing = false;
      // Defer releasing the guard: vis.js emits selectNode synchronously.
      setTimeout(function() { drxSelectGuard = false; }, 0);
    }
  }

  function positionTray() {
    var toolbar = $("drx-toolbar");
    var tray = $("drx-tray");
    if (!toolbar || !tray) return;
    var rect = toolbar.getBoundingClientRect();
    tray.style.top = Math.round(rect.bottom + 12) + "px";
  }

  function renderSelectionTray() {
    var ids = selectedIds();
    var count = $("drx-pivot-count");
    if (count) count.textContent = String(ids.length);
    var trayCount = $("drx-tray-count");
    if (trayCount) trayCount.textContent = String(ids.length);
    var btn = $("drx-btn-pivot");
    if (btn) {
      if (ids.length) btn.classList.add("drx-btn-ready");
      else btn.classList.remove("drx-btn-ready");
    }
    var tray = $("drx-tray");
    var list = $("drx-tray-list");
    if (tray) {
      if (ids.length) tray.classList.add("drx-open");
      else tray.classList.remove("drx-open");
    }
    positionTray();
    if (!list) return;
    if (!ids.length) {
      list.className = "drx-tray-empty";
      list.textContent = "Klik node untuk memilih; klik lagi untuk melepas.";
      return;
    }
    list.className = "";
    list.innerHTML = ids.map(function(id) {
      var node = nodeRecord(id) || {};
      var level = node.level == null ? "?" : node.level;
      var mark = drxPivoted[id] ? " \u00B7 sudah dipivot" : "";
      return '<span class="drx-chip" data-unselect="' + esc(id) + '" title="Klik untuk melepas">' +
        esc(id) + " \u00B7 " + esc(node.label || "") +
        ' <span class="drx-level-pill">L' + esc(level) + "</span>" +
        esc(mark) + " \u00d7</span>";
    }).join(" ");
    Array.prototype.forEach.call(list.querySelectorAll("[data-unselect]"), function(el) {
      el.addEventListener("click", function() {
        delete drxSelected[el.getAttribute("data-unselect")];
        syncNetworkSelection();
        applySelectionStyles();
        renderSelectionTray();
      });
    });
  }

  function updatePivotCount() {
    renderSelectionTray();
  }

  function applySelectionStyles() {
    if (typeof nodes === "undefined" || !nodes || !nodes.update) return;
    var selected = {};
    selectedIds().forEach(function(id) { selected[String(id)] = true; });
    var updates = [];
    Object.keys(drxBaseStyles).forEach(function(id) {
      var base = drxBaseStyles[id] || {};
      if (selected[id]) {
        var font = Object.assign({}, base.font || {});
        font.color = "#FFFFFF";
        updates.push({
          id: id,
          color: {
            background: DRX_SELECTED_BG,
            border: DRX_SELECTED_BORDER,
            highlight: { background: DRX_SELECTED_BG, border: DRX_SELECTED_BORDER },
            hover: { background: DRX_SELECTED_BG, border: DRX_SELECTED_BORDER }
          },
          borderWidth: 2,
          font: font,
          shadow: { enabled: true, color: DRX_SELECTED_GLOW, size: 12, x: 0, y: 0 }
        });
      } else {
        updates.push({
          id: id,
          color: base.color,
          borderWidth: base.borderWidth,
          font: base.font,
          shadow: base.shadow
        });
      }
    });
    if (updates.length) {
      try { nodes.update(updates); } catch (e) {}
    }
    updatePivotCount();
  }

  function crossEdgeCount() {
    return (DRX.edges || []).filter(function(edge) { return !!edge.cross; }).length;
  }

  function renderCrossButton() {
    var btn = $("drx-btn-cross");
    if (!btn) return;
    var count = crossEdgeCount();
    btn.textContent = "\u2194 Cross (" + count + ")";
    if (drxShowCross) btn.classList.add("drx-btn-active");
    else btn.classList.remove("drx-btn-active");
    btn.disabled = count === 0;
  }

  function applyCrossVisibility() {
    if (typeof edges === "undefined" || !edges || !edges.update) return;
    var updates = (DRX.edges || []).map(function(edge, index) {
      return { id: "e" + index, hidden: !!edge.cross && !drxShowCross };
    });
    if (updates.length) {
      try { edges.update(updates); } catch (e) {}
    }
  }

  function setStatus(text, ok) {
    var box = $("drx-status");
    if (!box) return;
    box.innerHTML = text;
    box.style.color = ok === false ? "#EF4444" : "";
  }
  function hint(text) {
    var box = $("drx-action-hint");
    box.innerHTML = text;
    box.classList.add("drx-open");
    $("drx-sidebar").classList.add("drx-open");
  }

  function closePivotModal() {
    var modal = $("drx-modal");
    if (modal) modal.classList.remove("drx-open");
    drxModalIds = [];
  }

  function openPivotModal(ids) {
    drxModalIds = (ids || []).slice();
    var list = $("drx-modal-list");
    var confirm = $("drx-modal-confirm");
    if (!list || !confirm) return;
    confirm.textContent = "Kirim Pivot (" + drxModalIds.length + " node)";
    list.innerHTML = drxModalIds.map(function(id) {
      var node = nodeRecord(id) || {};
      var level = node.level == null ? "?" : node.level;
      var mark = drxPivoted[id] ? " \u00B7 <b>sudah pernah dipivot</b>" : "";
      return '<div class="drx-card"><span class="drx-mono">[' + esc(id) + "]</span> " +
        esc(node.label || "") + ' <span class="drx-level-pill">level ' +
        esc(level) + "</span>" + mark + "</div>";
    }).join("");
    var modal = $("drx-modal");
    if (modal) modal.classList.add("drx-open");
  }

  function showNode(id) {
    var node = nodeRecord(id);
    if (!node) return;
    var badge = badgeMeta(node);
    var grounding = (node.grounding && node.grounding.length)
      ? node.grounding : (node.quotes || []);
    var quotes = grounding.map(function(q) {
      return '<blockquote class="drx-grounding">\u201C' + esc(q) + '\u201D</blockquote>';
    }).join("") || '<span class="drx-muted">(tanpa kutipan grounding)</span>';
    var sources = (node.sources || []).map(function(url) {
      var clean = esc(url);
      var host = "";
      try { host = esc(new URL(String(url)).host); } catch (err) {}
      return '<div class="drx-source"><a href="' + clean +
        '" target="_blank" rel="noopener"><span class="drx-source-host">' + host +
        '</span> ' + clean + "</a></div>";
    }).join("") || '<span class="drx-muted">(tanpa URL sumber)</span>';
    var aliases = (node.aliases || []).length
      ? '<div class="drx-muted">Alias: ' + esc((node.aliases || []).join(", ")) + "</div>" : "";
    var reasons = (DRX.suggestions || []).filter(function(s) {
      return String(s.node_id) === String(node.id);
    }).map(function(s) {
      return '<div class="drx-card"><b>' + esc(s.label || s.node_id) + "</b>" +
        '<div class="drx-muted">' + esc(s.reason || "") + "</div></div>";
    }).join("");
    if (!reasons && node.critic_reason) {
      reasons = '<div class="drx-card">' + esc(node.critic_reason) + "</div>";
    }
    if (!reasons) {
      reasons = '<span class="drx-muted">Belum ada saran critic untuk node ini.</span>';
    }
    var created = "";
    if (Number(node.created_at || 0) > 0) {
      try {
        created = new Date(Number(node.created_at) * 1000).toLocaleString("id-ID");
      } catch (err) {}
    }
    var confidence = Number(node.confidence || 0);
    $("drx-inspector").innerHTML =
      '<div class="drx-inspector-name">[' + esc(node.id) + "] " + esc(node.label) + "</div>" +
      '<div style="margin:8px 0">' + typeBadge(node) + " " + statusBadge(node) +
      ' <span class="drx-badge" style="background:#1E293B;border-color:#334155;color:#94A3B8">CONF ' +
      esc(confidence.toFixed(2)) + "</span>" +
      (node.pivoted
        ? ' <span class="drx-badge" style="background:#451A03;border-color:#78350F;color:#FCD34D">SUDAH DIPIVOT</span>'
        : "") + "</div>" +
      '<div class="drx-card"><div class="drx-muted">Node ID</div>' +
      '<div class="drx-mono" style="color:' + badge.text + '">' + esc(node.id) + "</div>" +
      '<div class="drx-muted" style="margin-top:6px">Short label: ' +
      esc(node.short_label || "") + " \u00B7 level " + esc(node.level) +
      " \u00B7 derajat " + esc(node.degree) + " \u00B7 confidence " +
      esc(confidence.toFixed(2)) + (created ? " \u00B7 dibuat " + esc(created) : "") +
      "</div>" + aliases + "</div>" +
      '<div class="drx-card"><b>Bukti Grounding</b>' + quotes + "</div>" +
      '<div class="drx-card"><b>URL Sumber</b>' + sources + "</div>" +
      '<div class="drx-card"><b>Saran Critic</b>' + reasons + "</div>";
    $("drx-sidebar").classList.add("drx-open");
  }

  function nodeSummary(node) {
    var summary = String(node.summary || "").replace(/\s+/g, " ").trim();
    if (summary) return summary;
    var quotes = node.grounding || node.quotes || [];
    if (quotes.length) {
      var first = String(quotes[0] || "").replace(/\s+/g, " ").trim();
      var match = first.match(/^[\s\S]*?[.!?](?:\s|$)/);
      var sentence = match ? match[0].trim() : first;
      if (sentence) return sentence.slice(0, 180);
    }
    return String(node.label || "Node") + " - " +
      Number(node.source_count || 0) + " sumber independen.";
  }

  function hideTooltip() {
    if (drxHoverTimer) {
      clearTimeout(drxHoverTimer);
      drxHoverTimer = null;
    }
    drxHoverNode = null;
    var box = $("drx-tooltip");
    if (box) {
      box.classList.remove("drx-open");
      box.setAttribute("aria-hidden", "true");
    }
  }

  function positionTooltip() {
    var box = $("drx-tooltip");
    if (!box || !box.classList.contains("drx-open")) return;
    var gap = 16;
    var left = drxPointer.x + gap;
    var top = drxPointer.y + gap;
    var width = box.offsetWidth || 280;
    var height = box.offsetHeight || 100;
    if (left + width > window.innerWidth - 12) left = drxPointer.x - width - gap;
    if (top + height > window.innerHeight - 12) top = drxPointer.y - height - gap;
    box.style.left = Math.max(12, Math.round(left)) + "px";
    box.style.top = Math.max(12, Math.round(top)) + "px";
  }

  function showTooltip(id) {
    var node = nodeRecord(id);
    var box = $("drx-tooltip");
    if (!node || !box) return;
    var count = Number(node.source_count || 0);
    var countText = node.verified
      ? "Verified " + count + " Sumber"
      : count + " Sumber";
    if (node.pivoted) countText += " \u00B7 sudah dipivot";
    box.innerHTML =
      '<div class="drx-tooltip-title">' + esc(node.label) + "</div>" +
      '<div class="drx-tooltip-summary">' + esc(nodeSummary(node)) + "</div>" +
      '<div class="drx-tooltip-meta">' + esc(countText) + "</div>";
    box.classList.add("drx-open");
    box.setAttribute("aria-hidden", "false");
    positionTooltip();
  }

  function showEdge(index) {
    var record = DRX.edges[index];
    if (!record) return;
    var sources = (record.sources || []).map(function(url) {
      return '<div style="margin:4px 0"><a href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(url) + "</a></div>";
    }).join("") || '<span class="drx-muted">(tanpa URL sumber)</span>';
    $("drx-inspector").innerHTML =
      '<div class="drx-card"><div class="drx-mono" style="color:#38BDF8">' + esc(record.source) + " \u2192 " + esc(record.target) + "</div>" +
      '<div style="margin:6px 0">' + statusBadge(record) + "</div>" +
      "<div>Predikat: <b>" + esc(record.predicate) + "</b></div>" +
      '<div class="drx-muted">confidence ' + esc(record.confidence) + "</div></div>" +
      '<div class="drx-card"><b>Sumber</b>' + sources + "</div>";
    $("drx-sidebar").classList.add("drx-open");
  }

  function renderSuggestions() {
    var box = $("drx-suggestions");
    if (!DRX.suggestions || !DRX.suggestions.length) { box.innerHTML = "(tidak ada saran)"; return; }
    box.innerHTML = DRX.suggestions.map(function(s) {
      var action = DRX_WEB
        ? '<button class="drx-btn" style="padding:3px 9px;margin-left:6px" data-pivot="' +
          esc(s.node_id) + '">Pivot</button>'
        : "";
      return '<div class="drx-card"><span class="drx-chip" data-focus="' + esc(s.node_id) + '">' +
        esc(s.node_id) + " \u2013 " + esc(s.label) + "</span>" + action +
        '<div class="drx-muted">' + esc(s.reason) + "</div></div>";
    }).join("");
    Array.prototype.forEach.call(box.querySelectorAll("[data-focus]"), function(el) {
      el.addEventListener("click", function() { window.DRX_FOCUS(el.getAttribute("data-focus")); });
    });
    Array.prototype.forEach.call(box.querySelectorAll("[data-pivot]"), function(el) {
      el.addEventListener("click", function() {
        var id = el.getAttribute("data-pivot");
        setSelected([id]);
        syncNetworkSelection();
        applySelectionStyles();
        renderSelectionTray();
        openPivotModal([id]);
      });
    });
  }

  function renderStats() {
    var meta = DRX.meta || {};
    var q = meta.budget || {};
    var stats = $("drx-stats");
    if (!stats) return;
    var tokens = Number(q.tokens == null ? (meta.tokens || 0) : q.tokens);
    var maxTokens = Number(q.max_tokens || 0);
    var remaining = q.tokens_remaining == null
      ? Math.max(0, maxTokens - tokens)
      : Number(q.tokens_remaining);
    var tokenText = maxTokens
      ? Number(tokens).toLocaleString("id-ID") + "/" + Number(maxTokens).toLocaleString("id-ID") +
        " token (sisa " + Number(remaining).toLocaleString("id-ID") + ")"
      : Number(tokens).toLocaleString("id-ID") + " token";
    var urls = Number(q.urls == null ? 0 : q.urls);
    var maxUrls = Number(q.max_urls || 0);
    var urlsRemaining = q.urls_remaining == null
      ? Math.max(0, maxUrls - urls)
      : Number(q.urls_remaining);
    var elapsed = Math.round(Number(q.elapsed_seconds || 0));
    var maxSeconds = Math.round(Number(q.max_runtime_seconds || 0));
    var secondsRemaining = q.seconds_remaining == null
      ? Math.max(0, maxSeconds - elapsed)
      : Math.round(Number(q.seconds_remaining));
    var over = (q.overrides || []).length > 0 ||
      (maxTokens > 0 && tokens >= maxTokens) ||
      (maxUrls > 0 && urls >= maxUrls) ||
      (maxSeconds > 0 && elapsed >= maxSeconds);
    var bits = [
      "Siklus " + esc(meta.cycle || 0),
      tokenText,
      "URL " + urls + "/" + maxUrls + " (sisa " + urlsRemaining + ")",
      "waktu " + elapsed + "s/" + maxSeconds + "s (sisa " + secondsRemaining + "s)",
      "$" + Number(meta.cost_usd || 0).toFixed(6)
    ];
    if (over) bits.push("\u26A0 over budget");
    if (meta.status) bits.push(esc(meta.status));
    stats.innerHTML = bits.join(" \u00B7 ");
  }

  window.DRX_FOCUS = function(nodeId) {
    setSelected([nodeId]);
    if (typeof network !== "undefined" && network) {
      try {
        var element = network.body.data.nodes.get(nodeId);
        if (element) {
          network.focus(nodeId, { scale: 1.1, animation: true });
          syncNetworkSelection();
        }
      } catch (e) {}
    }
    showNode(nodeId);
    applySelectionStyles();
    renderSelectionTray();
  };

  function postAction(path, body, okMessage) {
    if (!DRX_WEB) return;
    setStatus("Mengirim " + path + "\u2026");
    fetch(DRX_API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {})
    }).then(function(res) {
      return res.json().catch(function() { return {}; }).then(function(data) {
        return { ok: res.ok, data: data };
      });
    }).then(function(result) {
      if (!result.ok) {
        setStatus("Gagal: " + esc(result.data.error || "permintaan ditolak"), false);
        return;
      }
      setStatus(okMessage || esc(result.data.message || "Permintaan dikirim\u2026"));
    }).catch(function() {
      setStatus("Server web tidak merespons.", false);
    });
  }

  function applyPayload(data) {
    if (!data || !data.vis) return;
    try {
      if (typeof nodes !== "undefined" && nodes && nodes.update) {
        rememberNodeStyles(data.vis.nodes || []);
        nodes.update(data.vis.nodes);
      }
      if (typeof edges !== "undefined" && edges && edges.update) {
        var visEdges = (data.vis.edges || []).map(function(edge) {
          var copy = {};
          Object.keys(edge).forEach(function(key) { copy[key] = edge[key]; });
          copy.hidden = !!copy.cross && !drxShowCross;
          return copy;
        });
        edges.update(visEdges);
      }
    } catch (e) {}
    DRX.nodes = data.nodes || DRX.nodes;
    DRX.edges = data.edges || DRX.edges;
    DRX.suggestions = data.suggestions || [];
    DRX.meta = data.meta || DRX.meta;
    installNodeRenderers();
    drxPivoted = {};
    ((DRX.meta || {}).pivoted_nodes || []).forEach(function(id) {
      drxPivoted[String(id)] = true;
    });
    // Buang id yang sudah tidak ada di payload tetapi masih tersimpan lokal.
    Object.keys(drxSelected).forEach(function(id) {
      if (!nodeRecord(id)) delete drxSelected[id];
    });
    renderSuggestions();
    renderStats();
    renderCrossButton();
    syncNetworkSelection();
    applySelectionStyles();
    renderSelectionTray();
    if (data.meta && data.meta.cycle !== drxCycle) {
      drxCycle = data.meta.cycle;
      setStatus("Siklus " + drxCycle + " selesai - kanvas diperbarui.");
    }
  }

  function bind() {
    if (typeof network === "undefined" || !network) { setTimeout(bind, 150); return; }
    rememberNodeStyles(nodes.get());
    installNodeRenderers();
    drxPivoted = {};
    ((DRX.meta || {}).pivoted_nodes || []).forEach(function(id) {
      drxPivoted[String(id)] = true;
    });
    window.addEventListener("resize", positionTray);
    updatePivotCount();
    renderCrossButton();
    $("drx-btn-pivot").addEventListener("click", function() {
      var ids = selectedIds();
      if (!ids.length) {
        hint("Pilih minimal satu node dengan <b>klik</b> (klik lagi untuk melepas). " +
          "Daftar pilihan muncul di kiri sebelum dikirim.");
        return;
      }
      if (DRX_WEB) {
        openPivotModal(ids);
      } else {
        hint('<b>Pivot node terpilih:</b> di terminal pilih <span class="drx-mono">[1]</span>, lalu tempel: <span class="drx-mono">' +
             esc(ids.join(",")) + "</span>");
      }
    });
    $("drx-clear-selection").addEventListener("click", function() {
      drxSelected = {};
      syncNetworkSelection();
      applySelectionStyles();
      renderSelectionTray();
    });
    $("drx-btn-cross").addEventListener("click", function() {
      drxShowCross = !drxShowCross;
      applyCrossVisibility();
      renderCrossButton();
    });
    $("drx-btn-expand").addEventListener("click", function() {
      if (DRX_WEB) {
        postAction("/api/expand", {}, "Expand dikirim: mencari gelombang dokumen level-1 baru\u2026");
      } else {
        hint('Di terminal pilih <span class="drx-mono">[2] Expand</span> untuk mencari gelombang dokumen level-1 baru (deep search + sudut baru).');
      }
    });
    $("drx-btn-stop").addEventListener("click", function() {
      if (DRX_WEB) {
        postAction("/api/stop", {}, "Stop dikirim: menyusun laporan final\u2026");
      } else {
        hint('Di terminal pilih <span class="drx-mono">[3] Selesai</span> untuk menghentikan pencarian dan menyusun laporan final (sekali saja).');
      }
    });
    $("drx-close").addEventListener("click", function() {
      $("drx-sidebar").classList.remove("drx-open");
    });
    $("drx-modal-cancel").addEventListener("click", closePivotModal);
    $("drx-modal").addEventListener("click", function(event) {
      if (event.target === $("drx-modal")) closePivotModal();
    });
    $("drx-modal-confirm").addEventListener("click", function() {
      var ids = drxModalIds.slice();
      if (!ids.length) { closePivotModal(); return; }
      postAction("/api/pivot", { node_ids: ids }, "Pivot dikirim: " + ids.join(", "));
      closePivotModal();
    });
    document.addEventListener("mousemove", function(event) {
      drxPointer.x = event.clientX;
      drxPointer.y = event.clientY;
      positionTooltip();
    });
    var canvas = $("mynetwork");
    if (canvas) canvas.addEventListener("mouseleave", hideTooltip);
    network.on("hoverNode", function(params) {
      var id = String(params.node);
      drxHoverNode = id;
      if (drxHoverTimer) clearTimeout(drxHoverTimer);
      drxHoverTimer = setTimeout(function() {
        if (drxHoverNode === id) showTooltip(id);
      }, 200);
    });
    network.on("blurNode", hideTooltip);
    network.on("dragStart", hideTooltip);
    network.on("zoom", hideTooltip);
    // PRD: clicking/selecting a node opens the 360px inspector drawer.
    function handleNodeSelection(params) {
      hideTooltip();
      if (drxSelectGuard || drxSelectedSyncing) return;
      var selected = params.node != null
        ? params.node
        : (params.nodes && params.nodes.length ? params.nodes[0] : null);
      if (selected == null) return;
      var id = String(selected);
      if (nodeRecord(id)) showNode(id);
    }
    network.on("selectNode", handleNodeSelection);
    // Some vis-network builds emit ``select`` (with a nodes array) for
    // programmatic/custom-shape selection; keep the drawer in sync.
    network.on("select", handleNodeSelection);
    network.on("click", function(params) {
      hideTooltip();
      if (params.edges && params.edges.length) {
        var edgeId = String(params.edges[0]);
        var match = /^e(\d+)$/.exec(edgeId);
        showEdge(match ? Number(match[1]) : Number(edgeId) - 1);
        return;
      }
      if (!params.nodes || !params.nodes.length) return;
      var id = String(params.nodes[0]);
      toggleSelected(id);
      syncNetworkSelection();
      applySelectionStyles();
      renderSelectionTray();
      // Keep the drawer open even when the click toggles the node off.
      showNode(id);
    });
    document.addEventListener("keydown", function(event) {
      if (event.key === "Escape") {
        closePivotModal();
        var sidebar = $("drx-sidebar");
        if (sidebar) sidebar.classList.remove("drx-open");
      }
    });
    renderSuggestions();
    renderStats();
    renderSelectionTray();
    setTimeout(anchorRootTop, 80);
    if (DRX_WEB) {
      setInterval(function() {
        fetch(DRX_API + "/api/graph", { cache: "no-store" })
          .then(function(res) { return res.ok ? res.json() : null; })
          .then(function(data) { if (data) applyPayload(data); })
          .catch(function() {});
      }, 2000);
    }
  }
  if (document.readyState === "complete" || document.readyState === "interactive") {
    setTimeout(bind, 60);
  } else {
    window.addEventListener("DOMContentLoaded", function() { setTimeout(bind, 60); });
  }
})();
"""


def _inject(pyvis_html: str, payload: dict, *, web_api: bool = False) -> str:
    css = f'<style id="drx-theme">{_theme_css()}</style>'
    script = (
        _JS_TEMPLATE.replace("__DRX_JSON__", _data_json(payload))
        .replace("__DRX_WEB__", "true" if web_api else "false")
        .replace("__DRX_BADGES__", _data_json(NODE_BADGES))
    )
    html = pyvis_html
    if "</head>" in html:
        html = html.replace("</head>", css + "</head>", 1)
    else:
        html = css + html
    markup = _toolbar_html()
    if "</body>" in html:
        html = html.replace("</body>", markup + "<script id=\"drx-app\">" + script + "</script></body>", 1)
    else:
        html = html + markup + "<script id=\"drx-app\">" + script + "</script>"
    return html


# ---- PyVis rendering ---------------------------------------------------------------


def _load_pyvis():
    from pyvis.network import Network  # local import: heavy & optional

    return Network


def _render_with_pyvis(
    payload: dict,
    cdn_resources: str,
) -> str:
    Network = _load_pyvis()
    net = Network(
        height="100vh",
        width="100%",
        bgcolor=BG,
        font_color=TEXT,
        directed=True,
        cdn_resources=cdn_resources or "remote",
    )
    options = {
        "nodes": {
            "shape": "box",
            "widthConstraint": {"minimum": 180, "maximum": 180},
            "shapeProperties": {"borderRadius": 8},
            "font": {
                "face": "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
                "color": TEXT,
                "size": 12,
            },
            "color": {"background": PANEL, "border": BORDER},
        },
        # PRD_design.md §2.1: hierarchical Top-to-Bottom mindmap, physics off.
        # Node levels are explicit (0 root, 1 initial, 2+ pivot) so the tree is
        # stable even before all edges arrive.
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "directed",
                "nodeSpacing": 180,
                "levelSeparation": 130,
                "treeSpacing": 200,
                "blockShifting": True,
                "edgeMinimization": True,
                "parentCentralization": True,
            }
        },
        "physics": {"enabled": False},
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.55}},
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "vertical",
                "roundness": 0.45,
            },
            "color": {"color": EDGE_BASE, "highlight": ACCENT},
            "width": 1.5,
            "selectionWidth": 2.5,
            "font": {
                "color": TEXT_MUTED,
                "size": 10,
                "face": "Inter, system-ui, sans-serif",
                "strokeWidth": 2,
                "strokeColor": BG,
            },
        },
        "interaction": {
            "hover": True,
            # Multi-select is implemented by our click-to-toggle JS (plain
            # clicks, no Ctrl/drag), so vis.js's built-in multi-select is off.
            "multiselect": False,
            "selectable": True,
            "selectConnectedEdges": False,
            "tooltipDelay": 200,
            "navigationButtons": True,
            "keyboard": {"bindToWindow": False},
        },
    }
    net.set_options(_data_json(options))

    elements = vis_elements(payload)
    for node in elements["nodes"]:
        node_id = node.pop("id")
        net.add_node(node_id, **node)
    for edge in elements["edges"]:
        node_from = edge.pop("from")
        node_to = edge.pop("to")
        edge_id = edge.pop("id")
        net.add_edge(node_from, node_to, id=edge_id, **edge)
    return net.generate_html(notebook=False)


# ---- static fallback (PyVis unavailable) --------------------------------------------


def _render_fallback(payload: dict) -> str:
    def _badge(n: dict) -> dict:
        return node_badge(str(n.get("type") or ""), root=bool(n.get("root")))

    nodes_html = "".join(
        f"""<div class="drx-card">
  <div class="drx-mono" style="color:{_badge(n)['text']}">[{n.get('id')}] {n.get('short_label') or n.get('label', '')}
  <span class="drx-badge" style="background:{_badge(n)['bg']};border-color:{_badge(n)['border']};color:{_badge(n)['text']}">{_badge(n).get('label', str(n.get('type') or 'lainnya').upper())}</span></div>
  <div style="color:{TEXT};font-size:13px;margin-bottom:4px">{n.get('label', '')}</div>
  <div class="drx-muted">tipe: {n.get('type')} &middot; sumber: {n.get('source_count', 0)}
  &middot; derajat: {n.get('degree', 0)} &middot; confidence: {n.get('confidence', 0.0)}</div>
  <div>{''.join(f'<blockquote class="drx-grounding">&ldquo;{q}&rdquo;</blockquote>' for q in (n.get('quotes') or []))}</div>
  <div>{''.join(f'<div class="drx-source"><a href="{s}" target="_blank" rel="noopener">{s}</a></div>' for s in (n.get('sources') or []))}</div>
  <div><b>Saran Critic:</b> <span class="drx-muted">{n.get('critic_reason') or '(belum ada)'}</span></div>
</div>"""
        for n in payload.get("nodes", [])
    )
    edges_html = "".join(
        f"""<div class="drx-card">
  <span class="drx-mono">{e.get('source')} &rarr; {e.get('target')}</span>
  &middot; <b>{e.get('predicate', '')}</b>
  <span class="drx-badge {'drx-badge-verified' if e.get('verified') else ('drx-badge-contradiction' if e.get('contradiction') else 'drx-badge-single')}">
    {'Verified (' + str(e.get('source_count', 0)) + ' Sumber)' if e.get('verified') else ('KONTRADIKSI' if e.get('contradiction') else str(e.get('source_count', 1)) + ' sumber')}
  </span>
</div>"""
        for e in payload.get("edges", [])
    )
    suggestions_html = "".join(
        f'<span class="drx-chip">{s.get("node_id")} &ndash; {s.get("label")}</span>'
        f'<div class="drx-muted">{s.get("reason", "")}</div>'
        for s in payload.get("suggestions", [])
    )
    meta = payload.get("meta", {})
    return f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>Investigasi - {meta.get('target', '')}</title>
<style>{_theme_css()}</style></head>
<body>
{_toolbar_html()}
<main style="max-width:1080px;margin:130px auto 40px;padding:0 16px">
  <h2 style="font-size:18px">Kanvas statis (PyVis tidak terpasang)</h2>
  <p class="drx-muted">Target: {meta.get('target', '')} &middot; konteks: {meta.get('context', '') or '-'}
  &middot; siklus {meta.get('cycle', 0)} &middot; {meta.get('nodes', 0)} node / {meta.get('edges', 0)} edge
  &middot; {meta.get('tokens', 0)} token &middot; biaya ~${meta.get('cost_usd', 0.0):.6f}</p>
  <h3>Saran Node (Critic)</h3><div>{suggestions_html or '<span class="drx-muted">(tidak ada saran)</span>'}</div>
  <h3>Node ({len(payload.get('nodes', []))})</h3><div>{nodes_html}</div>
  <h3>Relasi ({len(payload.get('edges', []))})</h3><div>{edges_html}</div>
</main>
<script>const DRX = {_data_json(payload)};</script>
</body></html>"""


# ---- public entry point --------------------------------------------------------------


def render_graph_html(
    store: Any,
    output_path: str | Path,
    *,
    target_entity: str = "",
    context: str = "",
    critic_suggestions: Sequence[dict] = (),
    cycle: int = 0,
    tokens: int = 0,
    cost_usd: float = 0.0,
    max_nodes: int = 200,
    cdn_resources: str = "remote",
    run_id: str = "",
    web_api: bool = False,
    budget: dict | None = None,
    pivoted_nodes: Sequence[str] = (),
) -> Path | None:
    """Render ``graph_interactive.html`` (never raises; returns path or None).

    ``web_api=True`` (Mode Web App) menjadikan tombol action bar sebagai REST
    client (``/api/pivot``, ``/api/expand``, ``/api/stop``) plus polling
    ``/api/graph`` supaya mindmap ter-update real-time.
    """
    output = Path(output_path)
    try:
        payload = build_payload(
            store,
            target_entity=target_entity,
            context=context,
            critic_suggestions=critic_suggestions,
            cycle=cycle,
            tokens=tokens,
            cost_usd=cost_usd,
            max_nodes=max_nodes,
            run_id=run_id,
            budget=budget,
            pivoted_nodes=pivoted_nodes,
        )
        payload["meta"]["api"] = bool(web_api)
        try:
            html = _inject(
                _render_with_pyvis(payload, cdn_resources), payload, web_api=web_api
            )
        except Exception as exc:  # pyvis missing/broken -> static fallback
            log.warning("pyvis render gagal (%s) -> fallback HTML statis", exc)
            html = _render_fallback(payload)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
        log.info(
            "kanvas interaktif dirender: %s (%d node, %d edge, %.1f KB%s)",
            output,
            len(payload.get("nodes", [])),
            len(payload.get("edges", [])),
            output.stat().st_size / 1024.0,
            ", web app" if web_api else "",
        )
        return output
    except Exception as exc:  # pragma: no cover - never break the run
        log.warning("render kanvas gagal: %s", exc)
        return None
