"""Query Decomposer — 70/30 context diversification + transparency tree.

Splits the main query into sub-queries:
    * 70% Anchored    — locked to the user's ``--context`` terms via boolean AND/OR.
    * 30% Exploratory — free-form, no context lock, for broad outside coverage.
Without a context, all sub-queries are exploratory (100%).
"""
from __future__ import annotations

import logging
import re
import uuid

from drx.models import Mode, QueryTask, ResearchPlan, SurveyResult, TaskOrigin
from drx.textutil import keyword_query

log = logging.getLogger("drx.gate.decomposer")


def split_context(context: str) -> list[str]:
    """Split a context string (``Surabaya, Informatika``) into clean terms."""
    return [t.strip() for t in re.split(r"[,;|]", context or "") if t.strip()]


def build_anchored(query: str, terms: list[str]) -> str:
    """Build an anchored boolean query: ``"Query" AND ("t1" OR "t2")``.

    A query that already carries aliases/quotes (``"A" OR "B"``) is grouped
    instead of re-quoted, so the boolean syntax stays valid.
    """
    if not terms:
        return query
    or_terms = " OR ".join(f'"{t}"' for t in terms)
    base = query.strip()
    if '"' in base or re.search(r"\b(?:AND|OR|NOT)\b", base):
        return f"({base}) AND ({or_terms})"
    return f'"{base}" AND ({or_terms})'


def render_tree(root: str, tasks: list[QueryTask], aliases: list[str] | None = None) -> str:
    """Render a human-readable decomposition tree for ``--verbose``."""
    n_anchored = sum(1 for t in tasks if t.is_anchored)
    n_exploratory = len(tasks) - n_anchored
    lines = [
        "Query Decomposition Tree",
        f"Root: {root}",
    ]
    if aliases and len(aliases) > 1:
        lines.append(f"Entity aliases: {', '.join(aliases)} (semua dicari)")
    if tasks:
        pct_a = 100 * n_anchored / len(tasks)
        pct_e = 100 * n_exploratory / len(tasks)
        lines.append(
            f"Split: {n_anchored} anchored ({pct_a:.0f}%) / "
            f"{n_exploratory} exploratory ({pct_e:.0f}%)  [target 70/30]"
        )
    for t in tasks:
        tag = "A" if t.is_anchored else "E"
        lines.append(f"  [{tag}] {t.query}")
        lines.append(f"      rationale: {t.rationale}")
        lines.append(
            f"      backend: {t.backend} | depth={t.depth} | "
            f"status={t.status} | results={t.result_count} | score={t.score:.2f}"
        )
    return "\n".join(lines)


class Decomposer:
    """Decomposes a query into anchored + exploratory sub-queries (70/30)."""

    def __init__(self, context: str = "", backend: str = "hybrid") -> None:
        self.context = context
        self.backend = backend

    def decompose(
        self,
        plan: ResearchPlan,
        survey: SurveyResult,
        mode: Mode,
    ) -> list[QueryTask]:
        main = plan.hypothesis
        terms = split_context(self.context)
        kw_main = keyword_query(main) or main

        specs: list[tuple[str, bool, str]] = []  # (query, is_anchored, rationale)
        if terms:
            # ---- anchored branch (target ~70%) ----
            anchored: list[tuple[str, bool, str]] = [
                (build_anchored(main, terms), True, f"terikat context {terms}")
            ]
            if len(terms) > 1:
                for term in terms:
                    anchored.append(
                        (build_anchored(main, [term]), True, f"terikat context [{term}]")
                    )
            if kw_main != main:
                anchored.append(
                    (
                        build_anchored(kw_main, terms),
                        True,
                        f"terikat context {terms} (varian kata kunci)",
                    )
                )

            # ---- exploratory branch (target ~30%, no context lock) ----
            candidates = [kw_main] if kw_main != main else []
            if main not in candidates:
                candidates.append(main)
            target_expl = max(1, round(len(anchored) * 3 / 7))
            exploratory = [
                (q, False, "eksplorasi bebas (gambaran luas)")
                for q in candidates[:target_expl]
            ]
            # Interleave so the exploratory branch is sampled EARLY: if the loop
            # stops early (self-answer sufficient / stagnation / budget), the 30%
            # breadth sweep has already run instead of being starved at the end.
            if exploratory and len(anchored) > 1:
                specs = [anchored[0], exploratory[0], *anchored[1:], *exploratory[1:]]
            else:
                specs = anchored + exploratory
        else:
            # 100% exploratory.
            specs.append((main, False, "eksplorasi bebas (tanpa context)"))
            if kw_main != main:
                specs.append((kw_main, False, "varian kata kunci"))

        tasks: list[QueryTask] = []
        for q, anchored, rationale in specs:
            tasks.append(
                QueryTask(
                    id=str(uuid.uuid4()),
                    parent_id=None,
                    query=q,
                    mode=mode,
                    depth=1,
                    origin=TaskOrigin.INITIAL,
                    hypothesis=main,
                    rationale=rationale,
                    backend=self.backend,
                    is_anchored=anchored,
                    context=terms if anchored else [],
                )
            )
        return tasks
