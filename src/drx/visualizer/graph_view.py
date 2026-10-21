"""PyVis/vis.js renderer — strict compliance with ``PRD_design_new.md``.

Clean Minimalist Light Theme (inspired by Notion Light & modern flowcharts):

* Canvas background ``#F8FAFC`` (soft off-white) with a subtle ``#CBD5E1``
  dot-grid at 20 px spacing.
* Node cards solid white ``#FFFFFF`` with border ``#E2E8F0`` and a clean
  box-shadow (``0 4px 6px -1px rgba(0,0,0,0.08)``); primary text dark slate
  ``#0F172A`` (``Inter`` / system-ui).
* Accent badges (minimalist accent only): root input/prompt indigo
  ``#EEF2FF``/``#6366F1``/``#4338CA``; kueri/sub-kueri cyan
  ``#ECFEFF``/``#06B6D4``/``#0E7490``; entitas/individu purple
  ``#F3E8FF``/``#A855F7``/``#6B21A8``; instansi & perusahaan amber
  ``#FEF3C7``/``#F59E0B``/``#B45309``. Dokumen/peristiwa/lainnya memakai
  tint tipis dari semantic color lama supaya tetap konsisten.

Layout wajib: **Hierarchical Left-to-Right** (``direction: 'LR'``,
``sortMethod: 'directed'``, ``nodeSpacing: 180``, ``levelSeparation: 350``,
physics off) dengan edge ``cubicBezier`` + ``forceDirection: 'horizontal'``
sehingga terbentuk mindmap cabang kiri-ke-kanan. Setiap node membawa atribut
``level`` eksplisit per cabang (0 root, 1 pencarian awal/Expand, lalu
``parent.level + 1`` untuk setiap pivot, tanpa batas level). Cross-edge
(relasi mundur ke level lebih awal) disembunyikan by default dan bisa
ditampilkan lewat toggle toolbar; alur utama kiri->kanan tetap dominan.

``PRD_design_new.md`` tetap berlaku untuk interaksi: action bar Pivot/Expand/Stop,
seleksi **klik-toggle** multi-node (tanpa Ctrl/drag), tray daftar
``ID · label · level`` + penanda sudah-dipivot, modal konfirmasi sebelum satu
batch POST, sidebar provenance, dan semantik edge (verified ``#22C55E`` solid,
satu sumber ``#94A3B8`` dashed, kontradiksi ``#EF4444`` solid).

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

log = logging.getLogger("drx.visualizer")

# ---- PRD_design_new.md design tokens (Clean Minimalist Light) ------------------

BG = "#F8FAFC"  # soft off-white canvas
PANEL = "#FFFFFF"  # white cards/panels
BORDER = "#E2E8F0"  # subtle border
GRID = "#CBD5E1"  # dot grid
TEXT = "#0F172A"  # dark slate primary text
TEXT_MUTED = "#64748B"
SHADOW = "rgba(15,23,42,0.08)"
ACCENT = "#3B82F6"  # edge/node highlight (PRD §2.1)

#: Accent badges (bg / border / text) — PRD_design_new.md §1.1.
NODE_BADGES: dict[str, dict[str, str]] = {
    "root": {"bg": "#EEF2FF", "border": "#6366F1", "text": "#4338CA"},
    "query": {"bg": "#ECFEFF", "border": "#06B6D4", "text": "#0E7490"},
    "individu": {"bg": "#F3E8FF", "border": "#A855F7", "text": "#6B21A8"},
    "instansi": {"bg": "#FEF3C7", "border": "#F59E0B", "text": "#B45309"},
    "perusahaan": {"bg": "#FEF3C7", "border": "#F59E0B", "text": "#B45309"},
    "dokumen": {"bg": "#D1FAE5", "border": "#10B981", "text": "#047857"},
    "peristiwa": {"bg": "#FFE4E6", "border": "#F43F5E", "text": "#BE123C"},
    "lainnya": {"bg": "#F1F5F9", "border": "#CBD5E1", "text": "#475569"},
}

#: Backward-compatible accent (border) color map per node type.
NODE_COLORS: dict[str, str] = {key: value["border"] for key, value in NODE_BADGES.items()}

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
}


def node_badge(node_type: str, *, root: bool = False) -> dict[str, str]:
    """Accent badge for one node (root/target wins over its entity type)."""
    key = "root" if root else (node_type or "lainnya").lower()
    return NODE_BADGES.get(key, NODE_BADGES["lainnya"])


def node_color(node_type: str) -> str:
    return node_badge(node_type)["border"]


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

    Cards are intentionally minimal: white background, thin accent border,
    label only (ID/type/provenance live in tooltip + inspector sidebar).
    Every node carries an explicit hierarchical ``level`` (0 root, 1 initial
    search/Expand, then ``parent.level + 1`` per pivot branch) so the LR tree
    is stable without edge inference.
    """
    nodes: list[dict] = []
    for node in payload.get("nodes", []):
        node_type = str(node.get("type") or "lainnya").lower()
        root = bool(node.get("root"))
        badge = node_badge(node_type, root=root)
        level = max(0, int(node.get("level") or 0))
        status = "verified" if node.get("verified") else f"{node.get('source_count', 0)} sumber"
        title = (
            f"<b>[{node.get('id')}] {node.get('label', '')}</b><br>"
            f"Tipe: {node_type}{' (root)' if root else ''} &middot; level {level}<br>"
            f"Status: {status} &middot; derajat {node.get('degree', 0)} &middot; "
            f"confidence {node.get('confidence', 0.0)}"
            + ("<br><b>sudah pernah dipivot</b>" if node.get("pivoted") else "")
        )
        nodes.append(
            {
                "id": str(node.get("id")),
                "label": str(node.get("label", "")),
                "title": title,
                "shape": "box",
                "level": level,
                "borderWidth": 1,
                "borderWidthSelected": 4,
                "color": {
                    "background": PANEL,
                    "border": badge["border"],
                    "highlight": {"background": ACCENT, "border": "#60A5FA"},
                    "hover": {"background": PANEL, "border": badge["border"]},
                },
                "font": {
                    "color": TEXT,
                    "size": 14 if root else 13,
                    "bold": root,
                    "face": "Inter, system-ui, sans-serif",
                },
                "margin": {"top": 8, "bottom": 8, "left": 12, "right": 12},
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
                    "color": "#CBD5E1" if cross else style["color"],
                    "highlight": ACCENT,
                },
                "dashes": True if cross else bool(style["dashes"]),
                "width": 1 if cross else 2,
                "font": {
                    "color": TEXT_MUTED,
                    "size": 11,
                    "face": "Inter, system-ui, sans-serif",
                    "strokeWidth": 0,
                },
            }
        )
    return {"nodes": nodes, "edges": edges}


# ---- HTML assembly ---------------------------------------------------------------


def _data_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).replace("</", "<\\/")


_CSS_TEMPLATE = """
:root {
  --drx-bg: __BG__; --drx-panel: __PANEL__; --drx-border: __BORDER__;
  --drx-grid: __GRID__; --drx-text: __TEXT__; --drx-muted: __TEXT_MUTED__;
  --drx-accent: __ACCENT__;
}
html, body {
  background-color: __BG__ !important;
  background-image: radial-gradient(__GRID__ 1px, transparent 1px);
  background-size: 20px 20px;
  color: __TEXT__;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  margin: 0; padding: 0;
}
#mynetwork, .card, .vis-network { background: transparent !important; }
#mynetwork { border: 1px solid __BORDER__; border-radius: 12px; }
.drx-mono, code {
  font-family: "Fira Code", ui-monospace, SFMono-Regular, Menlo, monospace;
}
.drx-toolbar {
  position: fixed; top: 14px; left: 50%; transform: translateX(-50%);
  display: flex; flex-wrap: wrap; justify-content: center; gap: 10px;
  align-items: center; z-index: 40; padding: 8px 12px; max-width: 96vw;
  box-sizing: border-box;
  background: __PANEL__; border: 1px solid __BORDER__; border-radius: 12px;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
}
.drx-btn {
  border: 1px solid __BORDER__; border-radius: 8px; cursor: pointer;
  padding: 8px 12px; font-size: 13px; font-weight: 600; color: __TEXT__;
  background: __PANEL__; transition: filter .15s ease;
}
.drx-btn:hover { filter: brightness(0.97); }
.drx-btn-pivot { background: #6366F1; border-color: #6366F1; color: #FFFFFF; }
.drx-btn-expand { background: #06B6D4; border-color: #06B6D4; color: #FFFFFF; }
.drx-btn-stop { background: #EF4444; border-color: #EF4444; color: #FFFFFF; }
.drx-btn-cross { background: __BG__; border-color: __GRID__; color: #475569; }
.drx-btn-cross.drx-btn-active { background: #E0F2FE; border-color: #60A5FA; color: #1D4ED8; }
.drx-btn-ready { box-shadow: 0 0 0 3px rgba(59,130,246,0.28); }
#drx-pivot-count { font-weight: 800; }
.drx-stats {
  font-size: 12px; color: __TEXT_MUTED__; padding: 4px 10px;
  border-left: 1px solid __BORDER__; margin-left: 4px; white-space: nowrap;
}
.drx-tray {
  display: none; position: fixed; top: 78px; left: 14px; z-index: 41;
  width: 330px; max-width: 44vw; max-height: 34vh; overflow-y: auto;
  background: __PANEL__; border: 1px solid __BORDER__; border-radius: 12px;
  padding: 10px 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
}
.drx-tray.drx-open { display: block; }
.drx-tray-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 8px; margin-bottom: 6px; font-size: 13px;
}
.drx-tray .drx-chip { cursor: pointer; }
.drx-tray-empty { color: __TEXT_MUTED__; font-size: 12px; }
.drx-modal-backdrop {
  display: none; position: fixed; inset: 0; z-index: 60;
  background: rgba(15,23,42,0.38); align-items: center; justify-content: center;
  padding: 18px;
}
.drx-modal-backdrop.drx-open { display: flex; }
.drx-modal {
  width: 460px; max-width: 94vw; max-height: 80vh; overflow-y: auto;
  background: __PANEL__; border: 1px solid __BORDER__; border-radius: 14px;
  padding: 18px; box-shadow: 0 18px 48px rgba(15,23,42,0.22);
}
.drx-modal h3 { margin: 0 0 8px; font-size: 16px; }
.drx-modal-list { margin: 8px 0 14px; }
.drx-modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
.drx-modal-confirm { background: #6366F1; border-color: #6366F1; color: #FFFFFF; }
.drx-level-pill {
  display: inline-block; padding: 1px 7px; border-radius: 999px;
  background: __BG__; border: 1px solid __BORDER__; font-size: 11px;
}
.drx-sidebar {
  position: fixed; top: 0; right: 0; width: 360px; max-width: 92vw; height: 100vh;
  background: __PANEL__; border-left: 1px solid __BORDER__; z-index: 45;
  transform: translateX(102%); transition: transform .22s ease;
  overflow-y: auto; padding: 18px; box-sizing: border-box;
  box-shadow: -8px 0 24px rgba(15,23,42,0.06);
}
.drx-sidebar.drx-open { transform: translateX(0); }
.drx-sidebar h3 { margin: 6px 0 12px; font-size: 15px; color: __TEXT__; }
.drx-chip {
  display: inline-block; margin: 3px 4px 3px 0; padding: 3px 9px;
  border-radius: 999px; border: 1px solid #06B6D4; background: #ECFEFF;
  font-size: 12px; cursor: pointer; color: #0E7490; font-family: "Fira Code", monospace;
}
.drx-chip:hover { border-color: #6366F1; color: #4338CA; }
.drx-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 700; border: 1px solid __BORDER__;
}
.drx-badge-verified { color: #FFFFFF; background: #22C55E; border-color: #22C55E; }
.drx-badge-single { color: #475569; background: #F1F5F9; }
.drx-badge-contradiction { color: #FFFFFF; background: #EF4444; border-color: #EF4444; }
.drx-card {
  background: __PANEL__; border: 1px solid __BORDER__; border-radius: 10px;
  padding: 10px 12px; margin: 8px 0; font-size: 13px; line-height: 1.5;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
}
.drx-muted { color: __TEXT_MUTED__; font-size: 12px; }
.drx-legend {
  position: fixed; left: 14px; bottom: 14px; z-index: 40; padding: 10px 12px;
  background: __PANEL__; border: 1px solid __BORDER__; border-radius: 12px;
  font-size: 12px; color: __TEXT__;
  box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
}
.drx-legend span { display: inline-flex; align-items: center; gap: 6px; margin-right: 10px; }
.drx-dot { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.drx-hint { display: none; margin-top: 10px; }
.drx-hint.drx-open { display: block; }
a { color: #0E7490; }
blockquote { color: __TEXT_MUTED__; }
@media (max-width: 760px) {
  .drx-toolbar { flex-wrap: wrap; width: 94vw; justify-content: center; }
  .drx-stats {
    white-space: normal; width: 100%; text-align: center; border-left: 0;
    border-top: 1px solid __BORDER__; padding-top: 6px; margin-left: 0;
  }
  .drx-sidebar { width: 100vw; }
  .drx-tray { top: 118px; left: 3vw; width: 94vw; max-width: 94vw; max-height: 30vh; }
}
"""


def _theme_css() -> str:
    css = _CSS_TEMPLATE
    for token, value in (
        ("__BG__", BG),
        ("__PANEL__", PANEL),
        ("__BORDER__", BORDER),
        ("__GRID__", GRID),
        ("__TEXT_MUTED__", TEXT_MUTED),
        ("__TEXT__", TEXT),
        ("__ACCENT__", ACCENT),
    ):
        css = css.replace(token, value)
    return css


def _toolbar_html() -> str:
    return f"""
<div class="drx-toolbar" id="drx-toolbar">
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
  <span class="drx-stats" id="drx-stats"></span>
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
  <div style="margin-bottom:6px;font-weight:600">Legenda</div>
  <div>
    <span><i class="drx-dot" style="background:{NODE_BADGES['root']['bg']};border:2px solid {NODE_BADGES['root']['border']}"></i>Target / Root</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['individu']['bg']};border:2px solid {NODE_BADGES['individu']['border']}"></i>Individu</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['instansi']['bg']};border:2px solid {NODE_BADGES['instansi']['border']}"></i>Instansi/Perusahaan</span>
    <span><i class="drx-dot" style="background:{NODE_BADGES['query']['bg']};border:2px solid {NODE_BADGES['query']['border']}"></i>Kueri/Saran</span>
  </div>
  <div style="margin-top:6px" class="drx-muted">
    <span style="border-top:3px solid {EDGE_COLORS['verified']};width:26px;display:inline-block"></span> terverifikasi
    <span style="border-top:3px dashed {EDGE_COLORS['single']};width:26px;display:inline-block;margin-left:8px"></span> 1 sumber
    <span style="border-top:3px solid {EDGE_COLORS['contradiction']};width:26px;display:inline-block;margin-left:8px"></span> kontradiksi
    <span style="border-top:2px dashed #CBD5E1;width:26px;display:inline-block;margin-left:8px"></span> cross-edge
  </div>
</div>
<div class="drx-sidebar" id="drx-sidebar">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <h3 style="margin:0">Inspeksi Node</h3>
    <button class="drx-btn" id="drx-close" style="padding:4px 9px">\u2715</button>
  </div>
  <div id="drx-inspector">
    <p class="drx-muted">Klik node untuk memilih/melepas (bisa banyak), lalu cek daftar di kiri sebelum Pivot.</p>
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
  function typeBadge(record) {
    var key = record.root ? "root" : String(record.type || "lainnya").toLowerCase();
    var badge = DRX_BADGES[key] || DRX_BADGES.lainnya;
    var label = record.root ? "ROOT / PROMPT" : key.toUpperCase();
    return badgeHtml(badge, label);
  }
  function statusBadge(record) {
    if (record.contradiction) return '<span class="drx-badge drx-badge-contradiction">KONTRADIKSI</span>';
    if (record.verified) return '<span class="drx-badge drx-badge-verified">Verified (' + record.source_count + ' Sumber)</span>';
    return '<span class="drx-badge drx-badge-single">Belum terverifikasi (' + (record.source_count || 1) + ' sumber)</span>';
  }

  // ---- selection feedback + cross-edge toggle ------------------------------
  var DRX_SELECTED_BG = "#3B82F6";
  var DRX_SELECTED_BORDER = "#60A5FA";
  var DRX_SELECTED_GLOW = "rgba(96,165,250,0.85)";
  var drxBaseStyles = {};
  var drxShowCross = false;
  var drxSelected = {};
  var drxPivoted = {};
  var drxSelectedSyncing = false;
  var drxModalIds = [];
  var drxCycle = (DRX.meta || {}).cycle || 0;

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
    try {
      if (ids.length) network.selectNodes(ids);
      else network.unselectAll();
    } catch (e) {} finally {
      drxSelectedSyncing = false;
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
          borderWidth: 4,
          font: font,
          shadow: { enabled: true, color: DRX_SELECTED_GLOW, size: 24, x: 0, y: 0 }
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
    var node = DRX.nodes.find(function(n) { return String(n.id) === String(id); });
    if (!node) return;
    var key = node.root ? "root" : String(node.type || "lainnya").toLowerCase();
    var badge = DRX_BADGES[key] || DRX_BADGES.lainnya;
    var sources = (node.sources || []).map(function(url) {
      var clean = esc(url);
      return '<div style="margin:4px 0"><a href="' + clean + '" target="_blank" rel="noopener">' + clean + '</a></div>';
    }).join("") || '<span class="drx-muted">(tanpa URL sumber)</span>';
    var quotes = (node.quotes || []).map(function(q) {
      return '<blockquote style="margin:6px 0;padding-left:10px;border-left:3px solid ' +
        badge.border + '">\u201C' + esc(q) + '\u201D</blockquote>';
    }).join("") || '<span class="drx-muted">(tanpa kutipan)</span>';
    var aliases = (node.aliases || []).length
      ? '<div class="drx-muted">Alias: ' + esc((node.aliases || []).join(", ")) + "</div>" : "";
    $("drx-inspector").innerHTML =
      '<div class="drx-card"><div class="drx-mono" style="color:' + badge.text + '">[' +
      esc(node.id) + "] " + esc(node.label) + "</div>" +
      '<div style="margin:6px 0">' + typeBadge(node) + " " + statusBadge(node) + "</div>" + aliases +
      '<div class="drx-muted">Tipe: ' + esc(node.type) + " \u00B7 derajat " + esc(node.degree) +
      ' \u00B7 confidence ' + esc((node.confidence || 0).toFixed ? node.confidence.toFixed(2) : node.confidence) + "</div></div>" +
      '<div class="drx-card"><b>Bukti (Grounding)</b>' + quotes + "</div>" +
      '<div class="drx-card"><b>Sumber</b>' + sources + "</div>";
    $("drx-sidebar").classList.add("drx-open");
  }

  function showEdge(index) {
    var record = DRX.edges[index];
    if (!record) return;
    var sources = (record.sources || []).map(function(url) {
      return '<div style="margin:4px 0"><a href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(url) + "</a></div>";
    }).join("") || '<span class="drx-muted">(tanpa URL sumber)</span>';
    $("drx-inspector").innerHTML =
      '<div class="drx-card"><div class="drx-mono" style="color:#0E7490">' + esc(record.source) + " \u2192 " + esc(record.target) + "</div>" +
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
    network.on("click", function(params) {
      if (params.edges && params.edges.length) {
        showEdge(Number(params.edges[0]) - 1);
        return;
      }
      if (!params.nodes || !params.nodes.length) return;
      var id = String(params.nodes[0]);
      toggleSelected(id);
      syncNetworkSelection();
      applySelectionStyles();
      renderSelectionTray();
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
    markup = _toolbar_html() + '<span id="drx-status" class="drx-muted" style="position:fixed;top:64px;left:50%;transform:translateX(-50%);z-index:41"></span>'
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
            "font": {
                "face": "Inter, system-ui, sans-serif",
                "color": TEXT,
                "size": 14,
            },
        },
        # PRD_design_new.md §2.1: hierarchical Left-to-Right mindmap, physics off.
        # Node levels are explicit (0 root, 1 initial, 2+ pivot) so the tree is
        # stable even before all edges arrive.
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "LR",
                "sortMethod": "directed",
                "nodeSpacing": 180,
                "levelSeparation": 350,
            }
        },
        "physics": {"enabled": False},
        "edges": {
            "arrows": {"to": {"enabled": True, "scaleFactor": 0.6}},
            "smooth": {
                "type": "cubicBezier",
                "forceDirection": "horizontal",
                "roundness": 0.45,
            },
            "color": {"color": EDGE_COLORS["single"], "highlight": ACCENT},
            "width": 2,
            "selectionWidth": 3,
        },
        "interaction": {
            "hover": True,
            # Multi-select is implemented by our click-to-toggle JS (plain
            # clicks, no Ctrl/drag), so vis.js's built-in multi-select is off.
            "multiselect": False,
            "selectable": True,
            "selectConnectedEdges": False,
            "tooltipDelay": 120,
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
  <div class="drx-mono" style="color:{_badge(n)['text']}">[{n.get('id')}] {n.get('label', '')}
  <span class="drx-badge" style="background:{_badge(n)['bg']};border-color:{_badge(n)['border']};color:{_badge(n)['text']}">{'ROOT / PROMPT' if n.get('root') else str(n.get('type') or 'lainnya').upper()}</span></div>
  <div class="drx-muted">tipe: {n.get('type')} &middot; sumber: {n.get('source_count', 0)}
  &middot; derajat: {n.get('degree', 0)} &middot; confidence: {n.get('confidence', 0.0)}</div>
  <div>{''.join(f'<blockquote style="margin:6px 0;padding-left:10px;border-left:3px solid {_badge(n)["border"]}">&ldquo;{q}&rdquo;</blockquote>' for q in (n.get('quotes') or []))}</div>
  <div>{''.join(f'<div><a href="{s}" target="_blank" rel="noopener">{s}</a></div>' for s in (n.get('sources') or []))}</div>
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
<main style="max-width:1080px;margin:70px auto 40px;padding:0 16px">
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
