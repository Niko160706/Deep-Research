"""Mindmap construction for Internship Hunter.

The internship graph deliberately reuses the existing GraphStore + Vis.js
renderer, so it inherits the Dark Neon theme, hierarchical UD layout, fixed
cards, hover tooltip and inspector drawer without touching Mode 1.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from drx.config import Settings
from drx.models import Fact, SourceRef
from drx.store.graph_store import GraphStore
from drx.visualizer import render_graph_html

from .models import InternshipJob, InternshipPlan

log = logging.getLogger("drx.internship.graph")


_UNSET_VALUES = {"", "tidak diketahui", "tidak disebutkan", "-", "unknown", "n/a"}


def build_internship_store(
    run_dir: str | Path,
    plan: InternshipPlan,
    jobs: Sequence[InternshipJob],
) -> GraphStore:
    """Mindmap sesuai PRD: ``Magang [Role]`` → ``Perusahaan`` → ``Posisi``.

    Center node adalah kueri/saran (root, level 0).  Level 1 berisi
    perusahaan tujuan magang dan level 2 posisinya.  Ketika perusahaan tidak
    diketahui, posisi menempel langsung ke root agar tidak memunculkan node
    placeholder di level 1.
    """
    store = GraphStore(Path(run_dir))
    root_label = plan.display_title or "Internship Hunter"
    store.ensure_root(root_label)
    for job in jobs:
        posisi = job.role_title or "Posisi"
        quote = (
            f"Magang {posisi}"
            + (f" di {job.company_name}" if job.company_name else "")
            + (f" ({job.location})" if job.location and job.location not in _UNSET_VALUES else "")
            + f"; mode {job.work_mode or 'tidak disebutkan'}"
        )
        provenance = [
            SourceRef(
                url=job.apply_url,
                title=posisi,
                quote=quote[:400],
            )
        ]
        confidence = 0.72
        company = (job.company_name or "").casefold()
        if company and company not in _UNSET_VALUES:
            company_label = job.company_name
            if company_label.casefold() not in {root_label.casefold(), posisi.casefold()}:
                store.resolve_and_add(
                    [
                        Fact(
                            subject=root_label,
                            predicate="magang_di_perusahaan",
                            object=company_label,
                            confidence_score=confidence,
                            provenance=provenance,
                            subject_short_label=root_label[:20],
                            object_short_label=company_label[:20],
                        )
                    ],
                    level=1,
                )
            store.resolve_and_add(
                [
                    Fact(
                        subject=company_label,
                        predicate="menawarkan_posisi",
                        object=posisi,
                        confidence_score=confidence,
                        provenance=provenance,
                        subject_short_label=company_label[:20],
                        object_short_label=posisi[:20],
                    )
                ],
                level=2,
            )
        else:
            store.resolve_and_add(
                [
                    Fact(
                        subject=root_label,
                        predicate="saran_posisi",
                        object=posisi,
                        confidence_score=confidence,
                        provenance=provenance,
                        subject_short_label=root_label[:20],
                        object_short_label=posisi[:20],
                    )
                ],
                level=1,
            )
    return store


def render_internship_graph(
    store: GraphStore,
    output_path: str | Path,
    *,
    plan: InternshipPlan,
    settings: Settings,
    run_id: str,
    job_count: int,
) -> Path | None:
    """Render the static internship mindmap with the shared visualizer."""
    path = render_graph_html(
        store,
        output_path,
        target_entity=plan.display_title,
        context=plan.search_context,
        cycle=1,
        tokens=0,
        cost_usd=0.0,
        max_nodes=settings.visualizer.max_nodes,
        cdn_resources=settings.visualizer.cdn_resources,
        run_id=run_id,
        web_api=False,
        budget={
            "max_nodes": settings.visualizer.max_nodes,
            "max_urls": job_count,
            "max_tokens": 0,
            "max_runtime_seconds": 0,
        },
    )
    if path is not None:
        _add_internship_banner(path, plan=plan, job_count=job_count)
    return path


def _add_internship_banner(path: Path, *, plan: InternshipPlan, job_count: int) -> None:
    """Add a lightweight context banner without changing the shared renderer.

    The shared renderer still emits its Pivot/Expand/Stop wiring for other
    modes.  For a finished internship run those controls are hidden with CSS
    (their JS bindings stay intact), and a small context banner is inserted
    into the existing toolbar so the layout does not overlap the graph.
    """
    try:
        html = path.read_text(encoding="utf-8")
    except Exception:
        return
    title = _html_escape(plan.display_title)
    subtitle = _html_escape(
        f"{plan.approach} · {plan.location} · {job_count} lowongan tervalidasi"
    )
    banner = f"""
<style id="drx-internship-banner-style">
  .drx-internship-banner {{
    max-width: min(720px, calc(100vw - 42px));
    padding: 3px 4px 4px; color: #F8FAFC;
    font: 600 12px Inter, system-ui, sans-serif; text-align: center;
    line-height: 1.45;
  }}
  .drx-internship-banner span {{ color: #94A3B8; font-weight: 400; }}
  .drx-internship-badge {{
    display: inline-block; margin-right: 6px; padding: 2px 8px;
    border-radius: 999px; border: 1px solid #0369A1; background: #082F49;
    color: #38BDF8; font: 600 10px Inter, system-ui, sans-serif;
    letter-spacing: 0.04em; vertical-align: middle;
  }}
  /* Internship results are a finished report, not a live pivot session. */
  .drx-toolbar-actions, #drx-stats, #drx-status {{ display: none !important; }}
</style>
<div class="drx-internship-banner" role="status">
  <span class="drx-internship-badge">KUERI/SARAN</span>INTERNSHIP HUNTER · {title}<br><span>{subtitle}</span>
</div>
"""
    marker = '<div class="drx-toolbar" id="drx-toolbar">'
    if marker in html:
        html = html.replace(marker, marker + banner, 1)
        path.write_text(html, encoding="utf-8")
    elif "</body>" in html:
        html = html.replace("</body>", banner + "</body>", 1)
        path.write_text(html, encoding="utf-8")


def _html_escape(value: object) -> str:
    return (
        str(value or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
