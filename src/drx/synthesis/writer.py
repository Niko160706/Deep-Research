"""Synthesis Agent — writes the final Markdown report + JSON sidecar.

Phase 1 renders a NARRATIVE (not a link dump) from the packed context: an
executive summary, verbatim findings attributed to their sources, keyword
entities, and sources grouped by domain. Phase 2 wires ``deepseek-reasoner``
(roles.synthesis) for fluent multi-paragraph synthesis.
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from drx.models import Report

log = logging.getLogger("drx.synthesis.writer")


def _domain(url: str) -> str:
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


class SynthesisWriter:
    """Produces the run's final artifacts."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id

    async def synthesize(self, query: str, context: dict) -> Report:
        documents = context["documents"]
        keywords = context["keywords"]
        sentences = context["sentences"]
        graph = context["graph"]
        vector = context["vector"]

        n_docs = len(documents)
        domains = sorted({d["domain"] for d in documents})
        top_kw = [w for w, _ in keywords[:8]]

        lines = self._header(query, n_docs, len(domains), top_kw)
        lines += self._summary(query, n_docs, len(domains), top_kw)
        lines += self._findings(sentences)
        lines += self._keywords(keywords)
        lines += self._sources(documents)

        markdown = "\n".join(lines).strip() + "\n"
        sidecar = {
            "run_id": self.run_id,
            "query": query,
            "documents": documents,
            "keywords": [{"term": w, "count": c} for w, c in keywords],
            "entities": [w for w, _ in keywords],
            "findings": sentences,
            "edges": graph.get("triples", []),
            "graph": {"nodes": graph["nodes"], "edges": graph["edges"]},
            "evidence_cards": vector["cards"],
            "sources": domains,
        }
        return Report(run_id=self.run_id, query=query, markdown=markdown, sidecar=sidecar)

    @staticmethod
    def _header(query: str, n_docs: int, n_domains: int, top_kw: list[str]) -> list[str]:
        kw = ", ".join(top_kw) if top_kw else "—"
        return [
            f"# Laporan Investigasi: {query}",
            "",
            f"> **{n_docs}** dokumen · **{n_domains}** sumber · kata kunci: {kw}",
            "",
        ]

    @staticmethod
    def _summary(query: str, n_docs: int, n_domains: int, top_kw: list[str]) -> list[str]:
        out = ["## Ringkasan Eksekutif", ""]
        if n_docs == 0:
            out.append(
                "Penelusuran belum berhasil menjangkau dokumen yang relevan. "
                "Kemungkinan kueri terlalu spesifik, atau sumber pencarian "
                "yang tersedia tidak memuat materi terkait."
            )
        else:
            focus = ", ".join(f"**{w}**" for w in top_kw[:5]) if top_kw else "topik terkait"
            out.append(
                f"Investigasi atas “{query}” menelusuri **{n_docs}** dokumen dari "
                f"**{n_domains}** sumber. Temuan yang paling menonjol berkisar pada "
                f"{focus}. Berikut rangkuman naratif dari informasi yang berhasil dihimpun."
            )
        out.append("")
        return out

    @staticmethod
    def _findings(sentences: list[dict]) -> list[str]:
        out = ["## Temuan Utama", ""]
        if not sentences:
            out.append("- Belum ada kalimat relevan yang dapat dirangkai dari dokumen yang dijangkau.")
        else:
            for s in sentences:
                label = s.get("title") or _domain(s["source"])
                out.append(f"- {s['text']} _({label})_")
        out.append("")
        return out

    @staticmethod
    def _keywords(keywords: list[tuple[str, int]]) -> list[str]:
        out = ["## Kata Kunci & Entitas", ""]
        if keywords:
            out.append("  \n".join(f"- **{w}** ({c}×)" for w, c in keywords))
        else:
            out.append("- (tidak terdeteksi)")
        out.append("")
        return out

    @staticmethod
    def _sources(documents: list[dict]) -> list[str]:
        out = ["## Sumber", ""]
        if not documents:
            out.append("- (tidak ada dokumen)")
            return out
        by_domain: dict[str, list[dict]] = {}
        for d in documents:
            by_domain.setdefault(d["domain"], []).append(d)
        for domain in sorted(by_domain):
            out.append(f"### {domain}")
            for d in by_domain[domain]:
                title = d["title"] or d["url"]
                out.append(f"- [{title}]({d['url']})")
            out.append("")
        return out
