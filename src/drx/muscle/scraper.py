"""Async scraper: aiohttp primary, Playwright (headless) fallback. 0 tokens.

Resilience features:
    * PDF documents are converted to text (pypdf), not discarded.
    * 403/429/999 (LinkedIn) and CAPTCHA pages fall back to Playwright over the
      TOR SOCKS proxy with a real desktop User-Agent.
    * Playwright launches a FRESH browser per attempt (no shared singleton) so a
      crashed context can never leak into the next fetch, with a generous
      timeout and a single retry on crash.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import aiohttp

from drx.models import Document, SearchResult
from drx.muscle.pdf_reader import (
    extract_pdf_text,
    extract_with_pdfplumber,
    extract_with_pypdf,
)

log = logging.getLogger("drx.muscle.scraper")

DEFAULT_HEADERS = {
    # A descriptive UA (with contact) is required by Wikipedia's robot policy;
    # it also works for the vast majority of general web pages.
    "User-Agent": (
        "DeepResearch/0.1 (research agent; contact: https://github.com/deep-research)"
    ),
    "Accept-Language": "en-US,en;q=0.9,id;q=0.8",
}

# A realistic desktop browser UA for Playwright (avoids "bot" UA blocking).
DESKTOP_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Optional TOR SOCKS support via aiohttp-socks (imported lazily).
try:
    from aiohttp_socks import ProxyConnector as _ProxyConnector

    _HAS_SOCKS = True
except ImportError:  # pragma: no cover
    _ProxyConnector = None  # type: ignore[assignment]
    _HAS_SOCKS = False

_CAPTCHA_MARKERS = (
    "captcha",
    "verify you are human",
    "cloudflare",
    "access denied",
    "challenge",
    "too many requests",
)

# HTTP statuses treated as "blocked" (999 = LinkedIn request-denied).
_BLOCKED_STATUS = {403, 429, 999}

# High-value research domains: worth a JavaScript render even for short pages.
_HIGH_VALUE_HOST_SUFFIXES = (".ac.id", ".go.id")
_HIGH_VALUE_HOSTS = (
    "imdb.com",
    "letterboxd.com",
    "wikipedia.org",
    "sinta.kemdiktisaintek.go.id",
)

# Government/academic sites often serve a broken certificate chain; TLS
# verification is waived for them (content risk is acceptable, fetch must work).
_INSECURE_TLS_SUFFIXES = (".ac.id", ".go.id")
_INSECURE_TLS_HOSTS = ("sinta.kemdiktisaintek.go.id",)

_PDF_PATH_RE = re.compile(r"\.pdf/?$", re.IGNORECASE)

_SCRIPT_BLOCK_RE = re.compile(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>")
_TAG_STRIP_RE = re.compile(r"(?s)<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _host_of(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def is_pdf_url(url: str, content_type: str = "") -> bool:
    """True for PDF documents (by extension or ``Content-Type``)."""
    if "application/pdf" in (content_type or "").lower():
        return True
    return bool(_PDF_PATH_RE.search((urlparse(url).path or "").lower()))


def is_high_value(url: str) -> bool:
    """True for academic/government/reference domains worth JS rendering."""
    host = _host_of(url)
    if not host:
        return False
    if any(host == h or host.endswith("." + h) for h in _HIGH_VALUE_HOSTS):
        return True
    return any(host.endswith(suffix) for suffix in _HIGH_VALUE_HOST_SUFFIXES)


def needs_insecure_tls(url: str) -> bool:
    """True when TLS verification should be waived for this host."""
    host = _host_of(url)
    if not host:
        return False
    if host in _INSECURE_TLS_HOSTS:
        return True
    return any(host.endswith(suffix) for suffix in _INSECURE_TLS_SUFFIXES)


def _looks_blocked(html: str) -> bool:
    """Heuristic: does this (usually short) page look like a block/CAPTCHA?"""
    sample = html[:2000].lower()
    return any(marker in sample for marker in _CAPTCHA_MARKERS)


def _visible_text(html: str) -> str:
    """Strip scripts/styles + tags to estimate real visible text length."""
    html = _SCRIPT_BLOCK_RE.sub(" ", html)
    text = _TAG_STRIP_RE.sub(" ", html)
    return _WS_RE.sub(" ", text).strip()


def _is_linkedin(url: str) -> bool:
    return "linkedin.com" in (urlparse(url).netloc or "").lower()


def _chromium_deps_dir() -> Path | None:
    """Locate bundled Chromium system libs (``.chromium-deps``) if present."""
    base = Path(__file__).resolve().parents[3] / ".chromium-deps" / "usr" / "lib"
    if not base.exists():
        return None
    for sub in base.iterdir():
        if sub.is_dir():
            return sub
    return None


def _ensure_chromium_libs() -> None:
    """Prepend bundled Chromium libs to ``LD_LIBRARY_PATH`` (no sudo needed)."""
    deps = _chromium_deps_dir()
    if deps is None:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    if str(deps) not in existing:
        os.environ["LD_LIBRARY_PATH"] = str(deps) + (f":{existing}" if existing else "")


class Scraper:
    """Fetches page HTML/PDF concurrently, degrading to Playwright / TOR."""

    def __init__(
        self,
        max_aiohttp: int = 10,
        max_playwright: int = 2,
        timeout: float = 15.0,
        enable_playwright: bool = True,
        tor_proxy: str = "socks5://127.0.0.1:9050",
        enable_tor: bool = True,
        min_content_chars: int = 500,
        high_value_min_chars: int = 200,
        playwright_timeout_ms: int = 30000,
        playwright_tor_timeout_ms: int = 45000,
        content_retry_backoff: float = 1.5,
    ) -> None:
        self.timeout = timeout
        self.enable_playwright = enable_playwright
        self.tor_proxy = tor_proxy
        self.enable_tor = enable_tor
        self.min_content_chars = min_content_chars
        # High-value domains (ac.id / go.id / wikipedia / imdb / letterboxd /
        # sinta) get a JS render when the static text is under 200 chars.
        self.high_value_min_chars = high_value_min_chars
        self.playwright_timeout_ms = playwright_timeout_ms
        # TOR circuits are slow: a blocked page needs a longer render budget.
        self.playwright_tor_timeout_ms = playwright_tor_timeout_ms
        self.content_retry_backoff = content_retry_backoff
        self._sem_aiohttp = asyncio.Semaphore(max_aiohttp)
        self._sem_playwright = asyncio.Semaphore(max_playwright)  # strict RAM guard

    async def fetch(self, results: list[SearchResult]) -> list[Document]:
        """Fetch all results concurrently; failures degrade to ``[]``."""
        if not results:
            return []
        outcomes = await asyncio.gather(
            *(self._fetch_one(r) for r in results), return_exceptions=True
        )
        docs: list[Document] = []
        for outcome in outcomes:
            if isinstance(outcome, Document):
                docs.append(outcome)
            elif isinstance(outcome, Exception):
                log.debug("fetch failed: %s", outcome)
        return docs

    async def _fetch_one(self, result: SearchResult) -> Document:
        url = result.url
        pdf = is_pdf_url(url)
        needs_tor = _is_linkedin(url)

        blocked, doc = await self._fetch_aiohttp(url)

        # PDFs must never be handed to Playwright: the browser would start a
        # download and crash with "Download is starting" instead of returning
        # content. If the PDF could not be parsed, the document is dropped.
        if pdf or (doc is not None and doc.method == "pdf"):
            if doc is None:
                log.debug("PDF without extractable text -> drop %s", url)
                raise RuntimeError(f"unreachable: {url}")
            if result.title and not doc.title:
                doc = doc.model_copy(update={"title": result.title})
            return doc

        if self._needs_render(doc, url):
            # aiohttp failed / blocked / thin / high-value short text -> render.
            if self.enable_playwright:
                rendered = await self._fetch_playwright(url, use_tor=(blocked or needs_tor))
                if rendered is not None and (doc is None or not self._is_thin(rendered)):
                    doc = rendered

        if doc is None:
            raise RuntimeError(f"unreachable: {url}")
        if result.title and not doc.title:
            doc = doc.model_copy(update={"title": result.title})
        return doc

    async def _fetch_aiohttp(self, url: str) -> tuple[bool, Document | None]:
        """Return ``(was_blocked, doc)``. Tries direct, then TOR on block."""
        async with self._sem_aiohttp:
            blocked, doc = await self._aiohttp_once(url, use_tor=False)
            if doc is not None:
                return blocked, doc
            if blocked and self._tor_available():
                log.info("blocked on %s -> retrying via TOR", url)
                _, doc = await self._aiohttp_once(url, use_tor=True)
                if doc is not None:
                    return True, doc
            return blocked, None

    async def _aiohttp_once(self, url: str, use_tor: bool) -> tuple[bool, Document | None]:
        """One HTTP attempt; each TOR attempt owns its connector + session.

        The connector must never be shared between sessions: ``ClientSession``
        closes the connector it was handed, so a shared SOCKS connector would die
        with the first session and every later TOR retry would fail with
        "Connector is closed." / "Session is closed." Each attempt therefore
        creates its own connector and closes it only after its own response has
        been fully read — no task can close another task's session mid-flight.
        """
        connector = self._new_connector(url, use_tor)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=DEFAULT_HEADERS, connector=connector
            ) as session:
                async with session.get(url) as resp:
                    if resp.status in _BLOCKED_STATUS:
                        log.debug("aiohttp %s -> HTTP %s (blocked)", url, resp.status)
                        return True, None
                    if resp.status != 200:
                        log.debug("aiohttp %s -> HTTP %s", url, resp.status)
                        return False, None

                    ctype = resp.headers.get("Content-Type", "").lower()
                    if is_pdf_url(url, ctype):
                        pdf_bytes = await resp.read()
                        text = self._extract_pdf_text(pdf_bytes)
                        if not text:
                            log.debug("PDF text extraction failed: %s", url)
                            return False, None
                        # ``method="pdf"`` marks the document as terminal: it is
                        # never escalated to Playwright ("Download is starting").
                        return False, Document(
                            url=url,
                            text=text,
                            fetched_at=time.time(),
                            method="pdf",
                        )

                    if "text/html" not in ctype and "text/plain" not in ctype:
                        log.debug("aiohttp %s non-HTML (%s)", url, ctype)
                        return False, None
                    html = await resp.text()
                    if _looks_blocked(html):
                        log.debug("aiohttp %s looks like a CAPTCHA/block page", url)
                        return True, None
                    return False, Document(
                        url=url, text=html, fetched_at=time.time(), method="aiohttp"
                    )
        except Exception as exc:
            log.debug("aiohttp error %s: %s", url, exc)
            return False, None
        finally:
            if connector is not None:
                try:
                    await connector.close()
                except Exception:  # pragma: no cover - close is best-effort
                    pass

    @staticmethod
    def _extract_pdf_text(pdf_bytes: bytes) -> str:
        """Raw PDF text: pdfplumber first (layout), pypdf fallback."""
        return extract_pdf_text(pdf_bytes)

    @staticmethod
    def _pdf_text_pypdf(pdf_bytes: bytes) -> str:
        return extract_with_pypdf(pdf_bytes)

    @staticmethod
    def _pdf_text_pdfplumber(pdf_bytes: bytes) -> str:
        return extract_with_pdfplumber(pdf_bytes)

    def _tor_available(self) -> bool:
        return self.enable_tor and _HAS_SOCKS and bool(self.tor_proxy)

    def _new_tor_connector(self, insecure_tls: bool = False):
        """Create a FRESH SOCKS connector for a single attempt (never shared)."""
        if not _HAS_SOCKS:
            return None
        kwargs = {"ssl": False} if insecure_tls else {}
        return _ProxyConnector.from_url(self.tor_proxy, **kwargs)

    def _new_connector(self, url: str, use_tor: bool):
        """Per-attempt connector: SOCKS for TOR, plain TLS-waived for ac.id/go.id."""
        insecure = needs_insecure_tls(url)
        if use_tor:
            return self._new_tor_connector(insecure_tls=insecure)
        if insecure:
            log.info("waiving TLS verification (broken chain) for %s", _host_of(url))
            return aiohttp.TCPConnector(ssl=False)
        return None

    def _is_thin(self, doc: Document) -> bool:
        """True when the static HTML has little visible text (JS app shell)."""
        return len(_visible_text(doc.text)) < self.min_content_chars

    def _needs_render(self, doc: Document | None, url: str) -> bool:
        """Decide whether Playwright should re-fetch this page.

        High-value domains (ac.id/go.id/wikipedia/imdb/letterboxd/sinta) escalate
        with a lower bar (<200 chars static text) because their content is often
        client-side rendered; every other domain keeps the general threshold.
        """
        if doc is None:
            return True
        chars = len(_visible_text(doc.text))
        if is_high_value(url):
            if chars < self.high_value_min_chars:
                log.info(
                    "high-value domain %s: only %d static chars -> playwright",
                    _host_of(url),
                    chars,
                )
                return True
            return False
        return chars < self.min_content_chars

    async def _fetch_playwright(self, url: str, use_tor: bool = False) -> Document | None:
        async with self._sem_playwright:
            try:
                from playwright.async_api import async_playwright  # noqa: F401
            except ImportError:
                log.debug("playwright not installed; skip fallback for %s", url)
                return None
            for attempt in (1, 2):
                try:
                    return await self._playwright_attempt(url, use_tor)
                except Exception as exc:
                    if "download" in str(exc).lower():
                        # A PDF/attachment reached the browser: retrying would
                        # only crash again ("Download is starting").
                        log.info("playwright skipped, URL is a download: %s", url)
                        return None
                    if attempt == 1:
                        log.warning(
                            "playwright attempt 1 failed for %s (%s); restarting context",
                            url,
                            exc,
                        )
                    else:
                        log.warning("playwright failed for %s: %s", url, exc)
            return None

    def _playwright_timeout(self, via_tor: bool) -> int:
        """Playwright render budget: 45s over the TOR SOCKS5 proxy, else 30s."""
        return self.playwright_tor_timeout_ms if via_tor else self.playwright_timeout_ms

    async def _playwright_attempt(self, url: str, use_tor: bool) -> Document:
        from playwright.async_api import async_playwright

        _ensure_chromium_libs()
        via_tor = use_tor and self._tor_available()
        timeout_ms = self._playwright_timeout(via_tor)
        async with async_playwright() as p:
            launch_kwargs: dict = {"headless": True}
            if via_tor:
                launch_kwargs["proxy"] = {"server": self.tor_proxy}
            browser = await p.chromium.launch(**launch_kwargs)
            try:
                context = await browser.new_context(user_agent=DESKTOP_UA)
                try:
                    page = await context.new_page()
                    page.set_default_timeout(timeout_ms)
                    try:
                        await page.goto(
                            url,
                            timeout=timeout_ms,
                            wait_until="domcontentloaded",
                        )
                    except Exception as exc:  # PDF/attachment -> not a page
                        if "download" in str(exc).lower():
                            raise RuntimeError(f"download: {url}") from exc
                        raise
                    # JS apps keep navigating after DOMContentLoaded; let the page
                    # settle before reading it (avoids "page is navigating").
                    await asyncio.sleep(2)
                    html = await self._safe_content(page)
                    return Document(
                        url=url, text=html, fetched_at=time.time(), method="playwright"
                    )
                finally:
                    await context.close()
            finally:
                await browser.close()

    async def _safe_content(self, page, retries: int = 3) -> str:
        """Read page HTML, tolerating transient "page is navigating" errors."""
        last_exc: Exception | None = None
        for attempt in range(retries):
            try:
                return await page.content()
            except Exception as exc:  # navigation race mid-read
                last_exc = exc
                log.debug("page.content attempt %d failed: %s", attempt + 1, exc)
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=5000)
                except Exception:
                    pass  # still navigating; fall through to the backoff
                await asyncio.sleep(self.content_retry_backoff)
        raise RuntimeError(f"page.content failed after {retries} attempts: {last_exc}")
