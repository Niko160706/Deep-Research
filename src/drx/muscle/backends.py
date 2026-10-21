"""Pluggable search backends.

Default backend is DuckDuckGo's HTML endpoint (no API key, free). Backends
implement :class:`AbstractSearchBackend` so a paid/self-hosted backend
(SearXNG, Tavily, SerpAPI) can be swapped in via config without touching the
orchestrator.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from abc import ABC, abstractmethod
from urllib.parse import parse_qs, quote, urlencode, urlparse, urlunparse

import aiohttp

from drx.models import SearchResult
from drx.textutil import strip_boolean

log = logging.getLogger("drx.muscle.backends")

# Query parameters that carry no semantic value for de-duplication.
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "gclsrc", "dclid", "msclkid", "mc_cid",
    "mc_eid", "igshid", "ref", "ref_src", "cmpid", "hmb_campaign", "_ga",
    "_gl", "yclid", "vero_id", "vero_conv", "wickedid",
}

# Site-navigation noise params: same page, different URL (AMP/mobile/locale).
# NOTE: no pagination params here ("page"/"p") — those select *different*
# content and must stay part of the URL key.
NOISE_PARAMS = {
    "amp", "amp_js_v", "output", "showpage", "hl", "lang", "locale", "m",
    "mobile", "nomobile", "redirect", "share", "utm", "print",
}

# Mobile/AMP host prefixes that point at the same page as the bare domain.
_MOBILE_PREFIXES = ("m.", "mobile.", "amp.", "wap.", "touch.", "www.")

# Region code -> SearXNG language code (used for Dynamic Region Lock).
REGION_LANGUAGES = {
    "id": "id-ID",
    "us": "en-US",
    "gb": "en-GB",
    "au": "en-AU",
    "jp": "ja-JP",
    "kr": "ko-KR",
    "cn": "zh-CN",
    "tw": "zh-TW",
    "de": "de-DE",
    "fr": "fr-FR",
    "es": "es-ES",
    "br": "pt-BR",
    "in": "hi-IN",
    "nl": "nl-NL",
    "ru": "ru-RU",
    "it": "it-IT",
    "tr": "tr-TR",
    "sa": "ar-SA",
}


def region_to_language(region: str) -> str:
    """Map a region code (``id``, ``us``, ``jp``, ``global``) to a language code."""
    r = (region or "").strip().lower()
    if not r or r == "global":
        return ""
    return REGION_LANGUAGES.get(r, "")


def canonicalize_url(url: str) -> str:
    """Canonical URL key: no mobile subdomain, no noise params, no fragment.

    ``m.antaranews.com/x?amp=1`` and ``antaranews.com/x`` collapse to the same
    key, so the anchored (70%) and exploratory (30%) paths can never scrape the
    same page twice. Mobile/AMP prefixes are stripped and tracking/noise query
    params (``?amp=1``, ``?showpage=all``, ``?hl=en``, UTM, …) are removed.
    """
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    host = (parsed.netloc or "").lower()
    for prefix in _MOBILE_PREFIXES:
        if host.startswith(prefix):
            host = host[len(prefix) :]
            break
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = ""
    if parsed.query:
        kept = [
            (k, v)
            for k, v in parse_qs(parsed.query, keep_blank_values=True).items()
            if k.lower() not in TRACKING_PARAMS and k.lower() not in NOISE_PARAMS
        ]
        query = urlencode(kept, doseq=True) if kept else ""
    return urlunparse((scheme, host, path, "", query, ""))


def normalize_url(url: str) -> str:
    """Strip tracking params + fragment; return a canonical key for dedup."""
    return canonicalize_url(url)


def dedupe_results(results: list[SearchResult]) -> list[SearchResult]:
    """Collapse duplicate URLs (after tracking-param stripping), keeping the
    richest snippet / highest score."""
    by_key: dict[str, SearchResult] = {}
    for r in results:
        key = normalize_url(r.url)
        existing = by_key.get(key)
        if existing is None or len(r.snippet) > len(existing.snippet) or r.score > existing.score:
            by_key[key] = r
    return list(by_key.values())

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class AbstractSearchBackend(ABC):
    """Contract every search backend must satisfy."""

    name = "abstract"
    #: Boolean operators (AND/OR/NOT) are meaningful for this backend.
    supports_boolean = False

    @abstractmethod
    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Return up to `limit` results for `query` (empty on failure)."""


class DuckDuckGoBackend(AbstractSearchBackend):
    """Scrape DuckDuckGo's server-rendered HTML endpoint."""

    name = "duckduckgo"
    endpoint = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        headers = {"User-Agent": _UA}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(self.endpoint, params={"q": query}) as resp:
                    if resp.status != 200:
                        log.warning("DDG returned HTTP %s", resp.status)
                        return []
                    html = await resp.text()
        except Exception as exc:  # network failure -> empty, not crash
            log.warning("DDG request failed: %s", exc)
            return []
        return self._parse(html, limit)

    @staticmethod
    def _parse(html: str, limit: int) -> list[SearchResult]:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            results: list[SearchResult] = []
            for link in soup.select("a.result__a"):
                url = DuckDuckGoBackend._clean_url(link.get("href", ""))
                title = link.get_text(strip=True)
                if not url or not title:
                    continue
                container = link.find_parent("div", class_="result")
                snippet = ""
                if container is not None:
                    snip = container.select_one(".result__snippet")
                    if snip is not None:
                        snippet = snip.get_text(strip=True)
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        domain=(urlparse(url).netloc or "").lower(),
                    )
                )
                if len(results) >= limit:
                    break
            return results
        except ImportError:  # pragma: no cover - bs4/lxml are base deps
            return []

    @staticmethod
    def _clean_url(href: str) -> str:
        """Resolve DDG redirect wrapper (``//duckduckgo.com/l/?uddg=...``)."""
        if not href:
            return ""
        if href.startswith("//"):
            href = "https:" + href
        parsed = urlparse(href)
        if "duckduckgo.com" in (parsed.netloc or ""):
            uddg = parse_qs(parsed.query).get("uddg", [""])[0]
            if uddg:
                return uddg
        return href


class BingSearchBackend(AbstractSearchBackend):
    """Scrape Bing's server-rendered SERP (no API key).

    Bing wraps result URLs in a ``/ck/a?...&u=<base64url>`` redirect; we decode
    the ``u`` parameter to recover the real destination URL.
    """

    name = "bing"
    endpoint = "https://www.bing.com/search"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        headers = {"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(
                    self.endpoint, params={"q": query, "count": limit}
                ) as resp:
                    if resp.status != 200:
                        log.warning("Bing returned HTTP %s", resp.status)
                        return []
                    html = await resp.text()
        except Exception as exc:
            log.warning("Bing request failed: %s", exc)
            return []
        return self._parse(html, limit)

    @staticmethod
    def _parse(html: str, limit: int) -> list[SearchResult]:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            results: list[SearchResult] = []
            for algo in soup.select("li.b_algo"):
                anchor = algo.select_one("h2 a")
                if anchor is None:
                    continue
                title = anchor.get_text(strip=True)
                url = BingSearchBackend._extract_url(anchor.get("href", ""))
                if not url or not title:
                    continue
                snippet = ""
                snip = algo.select_one("p.b_lineclamp2, p.b_lineclamp3, .b_caption p")
                if snip is not None:
                    snippet = snip.get_text(strip=True)
                results.append(
                    SearchResult(
                        url=url,
                        title=title,
                        snippet=snippet,
                        domain=(urlparse(url).netloc or "").lower(),
                    )
                )
                if len(results) >= limit:
                    break
            return results
        except ImportError:  # pragma: no cover - bs4/lxml are base deps
            return []

    @staticmethod
    def _extract_url(href: str) -> str:
        parsed = urlparse(href)
        if "bing.com" in (parsed.netloc or "") or parsed.path.startswith("/ck/"):
            encoded = parse_qs(parsed.query).get("u", [""])[0]
            if encoded:
                return BingSearchBackend._decode_u(encoded)
        return href

    @staticmethod
    def _decode_u(encoded: str) -> str:
        """Decode Bing's ``u`` param (base64 with an ``a1`` organic prefix)."""
        encoded = encoded.removeprefix("a1")
        if len(encoded) % 4 == 1:
            return ""  # malformed base64 length -> not a usable URL
        try:
            padded = encoded + "=" * (-len(encoded) % 4)
            return base64.b64decode(padded).decode("utf-8")
        except Exception:  # malformed payload -> skip
            return ""


class WikipediaSearchBackend(AbstractSearchBackend):
    """Wikipedia full-text search (MediaWiki API, no key, clean on-topic results).

    Wikipedia requires a descriptive User-Agent per its robot policy, so a
    dedicated UA (with contact) is set. Handles 429 rate-limit with one retry.
    """

    name = "wikipedia"
    endpoint = "https://en.wikipedia.org/w/api.php"

    def __init__(
        self,
        timeout: float = 15.0,
        user_agent: str = "DeepResearch/0.1 (research agent; contact: https://github.com/deep-research)",
    ) -> None:
        self.timeout = timeout
        self.user_agent = user_agent

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        headers = {"User-Agent": self.user_agent, "Accept-Encoding": "gzip"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(self.endpoint, params=params) as resp:
                    if resp.status == 429:
                        # Rate-limited: brief backoff, one retry.
                        import asyncio

                        await asyncio.sleep(1.5)
                        async with session.get(self.endpoint, params=params) as retry:
                            if retry.status != 200:
                                log.warning("Wikipedia returned HTTP %s", retry.status)
                                return []
                            data = await retry.json()
                    elif resp.status != 200:
                        log.warning("Wikipedia returned HTTP %s", resp.status)
                        return []
                    else:
                        data = await resp.json()
        except Exception as exc:
            log.warning("Wikipedia request failed: %s", exc)
            return []
        return self._parse(data, limit)

    @staticmethod
    def _parse(data: dict, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for hit in data.get("query", {}).get("search", []):
            title = hit.get("title", "")
            if not title:
                continue
            snippet = re.sub(r"<[^>]+>", "", hit.get("snippet", ""))
            url = "https://en.wikipedia.org/wiki/" + quote(title.replace(" ", "_"))
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    domain="en.wikipedia.org",
                )
            )
            if len(results) >= limit:
                break
        return results


class BraveSearchBackend(AbstractSearchBackend):
    """Brave Search API — clean, stable broad-web search (requires API key).

    Returns ``[]`` (never raises) when the key is missing or the request fails,
    so the failover chain can fall through to Wikipedia / other backends.
    """

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 15.0,
        country: str = "",
    ) -> None:
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY")
        self.timeout = timeout
        self.country = country

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if not self.api_key:
            log.warning("Brave Search API key missing; set BRAVE_SEARCH_API_KEY")
            return []

        headers = {
            "X-Subscription-Token": self.api_key,
            "Accept": "application/json",
        }
        params: dict[str, object] = {"q": query, "count": min(max(limit, 1), 30)}
        if self.country:
            params["country"] = self.country

        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(self.endpoint, params=params) as resp:
                    if resp.status in (401, 403):
                        log.warning("Brave API auth failed (HTTP %s); check API key", resp.status)
                        return []
                    if resp.status == 429:
                        log.warning("Brave API quota/rate limit reached (HTTP 429)")
                        return []
                    if resp.status != 200:
                        log.warning("Brave API returned HTTP %s", resp.status)
                        return []
                    data = await resp.json()
        except Exception as exc:
            log.warning("Brave API request failed: %s", exc)
            return []
        return self._parse(data, limit)

    @staticmethod
    def _parse(data: dict, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("web", {}).get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=item.get("description", ""),
                    domain=(urlparse(url).netloc or "").lower(),
                )
            )
            if len(results) >= limit:
                break
        return results


class TavilySearchBackend(AbstractSearchBackend):
    """Tavily Search API — clean, LLM-oriented web search (requires API key).

    Uses HTTP POST with a JSON body (Tavily passes the key in the body, not a
    header). Returns ``[]`` (never raises) on missing key / auth failure /
    quota, so the failover chain can fall through gracefully.
    """

    name = "tavily"
    endpoint = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 15.0,
        search_depth: str = "basic",
        include_answer: bool = True,
    ) -> None:
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        self.timeout = timeout
        self.search_depth = search_depth
        self.include_answer = include_answer

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if not self.api_key:
            log.warning("Tavily API key missing; set TAVILY_API_KEY")
            return []

        headers = {"Content-Type": "application/json"}
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": self.search_depth,
            "max_results": min(max(limit, 1), 30),
            "include_answer": self.include_answer,
            "include_raw_content": False,
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                    if resp.status in (401, 403):
                        log.warning("Tavily API auth failed (HTTP %s); check API key", resp.status)
                        return []
                    if resp.status == 429:
                        log.warning("Tavily API quota/rate limit reached (HTTP 429)")
                        return []
                    if resp.status != 200:
                        log.warning("Tavily API returned HTTP %s", resp.status)
                        return []
                    data = await resp.json()
        except Exception as exc:
            log.warning("Tavily API request failed: %s", exc)
            return []
        return self._parse(data, limit)

    @staticmethod
    def _parse(data: dict, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=item.get("content", ""),
                    domain=(urlparse(url).netloc or "").lower(),
                    score=float(item.get("score", 0.0)),
                )
            )
            if len(results) >= limit:
                break
        return results


class SearXNGBackend(AbstractSearchBackend):
    """Local SearXNG meta-search REST API (JSON). No API key needed.

    Requires the SearXNG Docker service (``docker-compose.yml``). Region lock is
    applied via SearXNG's ``language`` parameter.
    """

    name = "searxng"
    supports_boolean = True

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 15.0,
        region: str = "global",
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("SEARXNG_URL", "http://127.0.0.1:8080")
        ).rstrip("/")
        self.timeout = timeout
        self.region = region

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        language = region_to_language(self.region)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        results: list[SearchResult] = []
        pageno = 1
        # SearXNG returns ~10 results per page; paginate to reach `limit`.
        max_pages = max(1, (limit + 9) // 10)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                while len(results) < limit and pageno <= max_pages:
                    params: dict[str, object] = {
                        "q": query,
                        "format": "json",
                        "pageno": pageno,
                    }
                    if language:
                        params["language"] = language
                    async with session.get(
                        f"{self.base_url}/search", params=params
                    ) as resp:
                        if resp.status != 200:
                            log.warning("SearXNG returned HTTP %s", resp.status)
                            break
                        data = await resp.json()
                    page = self._parse(data, limit - len(results))
                    if not page:
                        break
                    results.extend(page)
                    pageno += 1
        except Exception as exc:
            log.warning("SearXNG request failed: %s", exc)
            return []
        return results[:limit]

    @staticmethod
    def _parse(data: dict, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=item.get("content", ""),
                    domain=(urlparse(url).netloc or "").lower(),
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
            if len(results) >= limit:
                break
        return results


class ExaSearchBackend(AbstractSearchBackend):
    """Exa.ai semantic web search (requires ``EXA_API_KEY``).

    Returns ``[]`` (never raises) on missing key / auth failure / quota so the
    hybrid aggregator degrades gracefully.
    """

    name = "exa"
    endpoint = "https://api.exa.ai/search"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 15.0,
        region: str = "global",
    ) -> None:
        self.api_key = api_key or os.environ.get("EXA_API_KEY")
        self.timeout = timeout
        self.region = region

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        if not self.api_key:
            log.warning("Exa API key missing; set EXA_API_KEY")
            return []

        headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        payload = {
            "query": query,
            "numResults": min(max(limit, 1), 30),
            "type": "auto",
            "contents": {"text": {"maxCharacters": 500}},
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                    if resp.status in (401, 403):
                        log.warning("Exa API auth failed (HTTP %s); check API key", resp.status)
                        return []
                    if resp.status == 429:
                        log.warning("Exa API quota/rate limit reached (HTTP 429)")
                        return []
                    if resp.status != 200:
                        log.warning("Exa API returned HTTP %s", resp.status)
                        return []
                    data = await resp.json()
        except Exception as exc:
            log.warning("Exa API request failed: %s", exc)
            return []
        return self._parse(data, limit)

    @staticmethod
    def _parse(data: dict, limit: int) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in data.get("results", []):
            url = item.get("url", "")
            title = item.get("title", "")
            if not url or not title:
                continue
            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=item.get("text", "") or "",
                    domain=(urlparse(url).netloc or "").lower(),
                    score=float(item.get("score", 0.0) or 0.0),
                )
            )
            if len(results) >= limit:
                break
        return results


class HybridSearchBackend(AbstractSearchBackend):
    """Tri-backend aggregator: SearXNG + Tavily + Exa, run in parallel.

    Merges the three sources, then applies a **URL de-duplication gate** that
    strips tracking params and collapses duplicate URLs before handing results
    to the scraper.
    """

    name = "hybrid"

    def __init__(
        self,
        backends: list[AbstractSearchBackend] | None = None,
        timeout: float = 15.0,
        region: str = "global",
    ) -> None:
        self.backends = backends or [
            SearXNGBackend(timeout=timeout, region=region),
            TavilySearchBackend(timeout=timeout),
            ExaSearchBackend(timeout=timeout, region=region),
        ]
        self.region = region

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        """Fan out to all sources; boolean syntax is stripped for semantic ones."""
        shaped = [
            (backend, query if getattr(backend, "supports_boolean", False) else strip_boolean(query))
            for backend in self.backends
        ]
        outcomes = await asyncio.gather(
            *(backend.search(q, limit) for backend, q in shaped),
            return_exceptions=True,
        )
        merged: list[SearchResult] = []
        for outcome in outcomes:
            if isinstance(outcome, list):
                merged.extend(outcome)
            elif isinstance(outcome, Exception):
                log.debug("hybrid backend error: %s", outcome)

        deduped = dedupe_results(merged)
        deduped.sort(key=lambda r: r.score, reverse=True)
        log.info(
            "hybrid: %d raw -> %d unique URLs across %d sources",
            len(merged),
            len(deduped),
            len(self.backends),
        )
        return deduped[:limit]


class FailoverSearchBackend(AbstractSearchBackend):
    """Tries a chain of backends and returns the first non-empty result.

    Uses a simple circuit breaker: a backend that returns empty ``max_failures``
    times in a row is marked "down" and skipped for the rest of the run. This
    stops a bot-blocked backend (e.g. DuckDuckGo's 202 anomaly page) from being
    hit on every single query.
    """

    name = "failover"

    def __init__(self, backends: list[AbstractSearchBackend], max_failures: int = 2) -> None:
        self.backends = backends
        self.max_failures = max_failures
        self._failures: dict[str, int] = {}
        self._down: set[str] = set()

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        for backend in self.backends:
            if backend.name in self._down:
                continue
            results = await backend.search(query, limit)
            if results:
                self._failures[backend.name] = 0
                log.info("backend '%s' returned %d results", backend.name, len(results))
                return results

            fails = self._failures.get(backend.name, 0) + 1
            self._failures[backend.name] = fails
            if fails >= self.max_failures:
                self._down.add(backend.name)
                log.info(
                    "backend '%s' marked down after %d empty responses",
                    backend.name,
                    fails,
                )
            else:
                log.info("backend '%s' returned 0 results", backend.name)
        return []


def build_backend(
    name: str, timeout: float = 15.0, region: str = "global"
) -> AbstractSearchBackend:
    """Resolve a backend name from config into a concrete backend instance."""
    key = (name or "hybrid").strip().lower()
    if key == "hybrid":
        return HybridSearchBackend(timeout=timeout, region=region)
    if key == "searxng":
        return SearXNGBackend(timeout=timeout, region=region)
    if key == "exa":
        return ExaSearchBackend(timeout=timeout, region=region)
    if key == "tavily":
        return TavilySearchBackend(timeout=timeout)
    if key == "brave":
        return BraveSearchBackend(timeout=timeout)
    if key == "duckduckgo":
        return DuckDuckGoBackend(timeout=timeout)
    if key == "bing":
        return BingSearchBackend(timeout=timeout)
    if key == "wikipedia":
        return WikipediaSearchBackend(timeout=timeout)
    if key == "failover":
        return FailoverSearchBackend(
            [
                TavilySearchBackend(timeout=timeout),
                BraveSearchBackend(timeout=timeout),
                WikipediaSearchBackend(timeout=timeout),
                DuckDuckGoBackend(timeout=timeout),
                BingSearchBackend(timeout=timeout),
            ]
        )
    if key == "mock":
        return MockSearchBackend()
    # Default: hybrid tri-backend aggregator (SearXNG + Tavily + Exa).
    return HybridSearchBackend(timeout=timeout, region=region)


class MockSearchBackend(AbstractSearchBackend):
    """Deterministic offline backend for tests and dry-runs."""

    name = "mock"

    def __init__(self) -> None:
        self._samples = [
            SearchResult(
                url="https://example.com/entity-a",
                title="Entity A overview",
                snippet="An authoritative page about entity A.",
                domain="example.com",
            ),
            SearchResult(
                url="https://example.com/entity-b",
                title="Entity B relations",
                snippet="How entity A relates to entity B.",
                domain="example.com",
            ),
        ]

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return [r.model_copy() for r in self._samples[:limit]]
