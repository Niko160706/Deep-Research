"""AutoSearch: heuristic URL scoring + redundancy filtering (async, 0 tokens).

Takes raw backend results, assigns a domain/keyword heuristic score, drops
already-seen and junk URLs, and returns a de-duplicated, ranked list.
"""
from __future__ import annotations

import logging
from urllib.parse import urlparse

from drx.models import SearchResult
from drx.muscle.backends import AbstractSearchBackend, build_backend, canonicalize_url
from drx.textutil import looks_like_name, sanitize_query, tokenize

log = logging.getLogger("drx.muscle.autosearch")

LOW_VALUE_TLDS = {"xyz", "info", "click", "top", "loan", "buzz"}
TRUSTED_HINTS = (
    "wikipedia.org",
    "arxiv.org",
    "nature.com",
    ".gov",
    ".edu",
    "ac.uk",
    ".org",
)


class AutoSearch:
    """Scores and filters search results before scraping."""

    def __init__(
        self,
        backend: AbstractSearchBackend | None = None,
        results_per_query: int = 10,
        min_score: float = 0.05,
        seen_urls: set[str] | None = None,
    ) -> None:
        self.backend: AbstractSearchBackend = backend or build_backend("hybrid")
        self.results_per_query = results_per_query
        self.min_score = min_score
        self.seen_urls: set[str] = seen_urls if seen_urls is not None else set()

    @staticmethod
    def is_anchored(query: str) -> bool:
        """True for 70/30 anchored queries: ``"Q" AND ("ctx" OR ...)``."""
        return " AND (" in query

    async def search(self, query: str, limit: int | None = None) -> list[SearchResult]:
        """Return scored, de-duplicated, relevant results (empty when none).

        ``limit`` overrides ``results_per_query`` for expansion searches that
        deliberately pull the next page/results beyond the initial top-N.
        Person-name queries are searched as exact matches (``"Nur Royul Amin"``)
        so the engine does not drift onto lookalike names. Anchored queries from
        the 70/30 strategy already carry their own quoting/boolean syntax, so
        they pass through untouched (semantic backends get a stripped variant
        inside the hybrid fan-out). Inputs with brackets (``Bayu Skak (Bayu Eko
        Moektito)``) are sanitized into ``"Bayu Skak" OR "Bayu Eko Moektito"``.
        """
        query = sanitize_query(query)
        effective_limit = int(limit) if limit and int(limit) > 0 else self.results_per_query
        if self.is_anchored(query):
            log.info("anchored query (70/30 strategy): %s", query)
            backend_query = query
        else:
            backend_query = f'"{query.strip()}"' if looks_like_name(query) else query
            if backend_query != query:
                log.info("name query detected -> exact-match: %s", backend_query)

        raw = await self.backend.search(backend_query, limit=effective_limit)
        if not raw:
            return []

        out: list[SearchResult] = []
        dupes = 0
        for r in raw:
            scored = self._score(r, query)
            if scored.score < self.min_score:
                continue  # drop results with no topical link to the query
            key = self._url_key(scored.url)
            if key in self.seen_urls:
                dupes += 1
                continue
            if self._is_junk(scored.url):
                continue
            self.seen_urls.add(key)
            out.append(scored)

        if dupes:
            log.info(
                "global dedup cache: %d/%d URLs already scraped -> skipped",
                dupes,
                len(raw),
            )
        out.sort(key=lambda r: r.score, reverse=True)
        return out

    # ---- helpers ----
    @staticmethod
    def _url_key(url: str) -> str:
        """Canonical dedup key: mobile/AMP variants collapse into one."""
        return canonicalize_url(url)

    @staticmethod
    def _is_junk(url: str) -> bool:
        netloc = (urlparse(url).netloc or "").lower()
        tld = netloc.rsplit(".", 1)[-1] if "." in netloc else ""
        return tld in LOW_VALUE_TLDS

    def _score(self, result: SearchResult, query: str) -> SearchResult:
        domain = result.domain or (urlparse(result.url).netloc or "").lower()
        qterms = set(tokenize(query))  # stopword-filtered, content words only

        title_hits = 0
        snip_hits = 0
        if qterms:
            title_lower = result.title.lower()
            title_hits = sum(1 for t in qterms if t in title_lower)
            if result.snippet:
                snip_hits = sum(1 for t in qterms if t in result.snippet.lower())

        # A result must have SOME topical link to the query; a trusted domain
        # alone is not enough to make an irrelevant page relevant.
        if title_hits == 0 and snip_hits == 0:
            return result.model_copy(update={"score": 0.0, "domain": domain, "reason": ""})

        score = 0.0
        reasons: list[str] = []
        if any(hint in domain for hint in TRUSTED_HINTS):
            score += 0.3
            reasons.append("trusted_domain")
        if title_hits:
            score += 0.4 * (title_hits / len(qterms))
            reasons.append("title_match")
        if snip_hits:
            score += 0.3 * (snip_hits / len(qterms))
            reasons.append("snippet_match")

        score = round(min(score, 1.0), 3)
        return result.model_copy(
            update={"score": score, "domain": domain, "reason": ",".join(reasons)}
        )
