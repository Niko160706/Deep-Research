"""Site-specific direct-URL parsers for the Internship Hunter engine.

Contract (from the internship brief):

* every source is fetched by building its URL directly — no UI clicking;
* Playwright is used when the browser is available, plain HTTP +
  BeautifulSoup is the fallback;
* card extraction is anchored on stable attributes / semantic CSS classes;
  hashed utility CSS class names are never used as selectors;
* when the site-specific selector finds nothing, the parser falls back to
  the ``<body>`` inner text split into lines and ships that to the LLM;
* Playwright pages always wait ``page.wait_for_timeout(3000)``;
* NodeFlair (source G) is fetched with browser-stealth hardening;

Supported sources (A-G)
-----------------------
A. JobStreet  — https://id.jobstreet.com/id/{QUERY-WITH-HYPHENS}-intern-jobs/in-Indonesia?daterange=31
B. Glints     — https://glints.com/id/en/opportunities/jobs/explore?keyword={QUERY}&country=ID&jobTypes=INTERNSHIP
C. Kalibrr    — https://www.kalibrr.id/id-ID/home/w/100-internship-_-ojt/k/{QUERY}
D. ITS        — https://careers.its.ac.id/lowongan?keywords={QUERY}
E. MSIB       — https://simbelmawa.kemdiktisaintek.go.id/magang/lowongan?cari={QUERY}
F. LinkedIn   — Tavily/Exa API with the Google dork
                ``site:linkedin.com/jobs "internship" "{QUERY}" Indonesia``
G. NodeFlair  — https://nodeflair.com/jobs?query={QUERY}&seniorities%5B%5D=intern&countries%5B%5D=Indonesia

This module intentionally has no dependency on ``drx.orchestrator`` — Mode 1
is never imported or modified here.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Protocol
from urllib.parse import parse_qs, quote, quote_plus, unquote, urljoin, urlparse

from .models import RawJob, SourceContent

log = logging.getLogger("drx.internship.site_parsers")

# ---- exact targeted URLs (A-G) ----------------------------------------------


def jobstreet_url(query: str) -> str:
    """A JobStreet: query becomes a lowercase-hyphenated path slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", _pure_query(query).strip().casefold()).strip("-")
    return (
        "https://id.jobstreet.com/id/"
        f"{quote(slug, safe='')}-intern-jobs/in-Indonesia?daterange=31"
    )


def glints_url(query: str) -> str:
    """B Glints: keyword percent-encoded as a query string."""
    return (
        "https://glints.com/id/en/opportunities/jobs/explore"
        f"?keyword={quote_plus(_pure_query(query))}&country=ID&jobTypes=INTERNSHIP"
    )


def kalibrr_url(query: str) -> str:
    """C Kalibrr: keyword percent-encoded into the ``k/`` path."""
    return (
        "https://www.kalibrr.id/id-ID/home/w/100-internship-_-ojt/k/"
        f"{quote(_pure_query(query), safe='')}"
    )


def its_career_url(query: str) -> str:
    """D ITS Career Center: keyword percent-encoded into ``keywords=``."""
    return f"https://careers.its.ac.id/lowongan?keywords={quote_plus(_pure_query(query))}"


def msib_url(query: str) -> str:
    """E MSIB (simbelmawa): keyword percent-encoded into ``cari=``."""
    return (
        "https://simbelmawa.kemdiktisaintek.go.id/magang/lowongan"
        f"?cari={quote_plus(_pure_query(query))}"
    )


def nodeflair_url(query: str) -> str:
    """G NodeFlair: query percent-encoded; intern + Indonesia from spec."""
    return (
        f"https://nodeflair.com/jobs?query={quote_plus(_pure_query(query))}"
        "&seniorities%5B%5D=intern&countries%5B%5D=Indonesia"
    )


LINKEDIN_DORK_TEMPLATE = 'site:linkedin.com/jobs "internship" "{query}" Indonesia'


def linkedin_dork(query: str) -> str:
    """F The exact Google dork used for the LinkedIn source."""
    return LINKEDIN_DORK_TEMPLATE.format(query=_pure_query(query))


GOOGLE_DORK_URL_TEMPLATE = "https://www.google.com/search?q={query}"


def google_dork_url(dork: str) -> str:
    """Diagnostic URL for a dork query (not fetched; LinkedIn uses the APIs)."""
    return GOOGLE_DORK_URL_TEMPLATE.format(query=quote_plus(dork.strip()))


def linkedin_dork_url(query: str) -> str:
    return google_dork_url(linkedin_dork(query))


def _pure_query(query: str) -> str:
    """Collapse whitespace; planner queries are already pure role names."""
    return " ".join(str(query or "").strip().split())


# ---- source specifications ---------------------------------------------------


@dataclass(frozen=True)
class SourceSpec:
    """One direct-URL source plus how it must be fetched/parsed."""

    key: str          # canonical short key: jobstreet | glints | ...
    name: str         # human/terminal label
    kind: str         # html | linkedin
    url: str          # template with {query} resolved by url_builder
    url_builder: str  # name of the local URL builder function
    requires_playwright: bool = False
    requires_stealth: bool = False
    internship_filtered: bool = True


_URL_BUILDERS: dict[str, object] = {
    "jobstreet_url": jobstreet_url,
    "glints_url": glints_url,
    "kalibrr_url": kalibrr_url,
    "its_career_url": its_career_url,
    "msib_url": msib_url,
    "nodeflair_url": nodeflair_url,
    "linkedin_dork_url": linkedin_dork_url,
}


def _spec(key: str, name: str, kind: str, builder: str, **flags: bool) -> SourceSpec:
    spec = SourceSpec(key=key, name=name, kind=kind, url="", url_builder=builder, **flags)
    return spec


def spec_url(spec: SourceSpec, query: str) -> str:
    builder = _URL_BUILDERS[spec.url_builder]
    return str(builder(query))  # type: ignore[operator]


JOBSTREET = _spec("jobstreet", "JobStreet", "html", "jobstreet_url")
GLINTS = _spec("glints", "Glints", "html", "glints_url")
KALIBRR = _spec("kalibrr", "Kalibrr", "html", "kalibrr_url")
ITS_CAREER = _spec("its_career", "ITS Career Center", "html", "its_career_url", internship_filtered=False)
MSIB = _spec("msib", "MSIB", "html", "msib_url")
NODEFLAIR = _spec("nodeflair", "NodeFlair", "html", "nodeflair_url", requires_playwright=True, requires_stealth=True)
LINKEDIN = _spec("linkedin", "LinkedIn Jobs", "linkedin", "linkedin_dork_url")

#: Sources A-G in the required order.  LinkedIn is API/dork based; the six
#: others are direct HTML pages.
HTML_SOURCES: tuple[SourceSpec, ...] = (
    JOBSTREET, GLINTS, KALIBRR, ITS_CAREER, MSIB, NODEFLAIR,
)

ALL_SOURCES: tuple[SourceSpec, ...] = HTML_SOURCES + (LINKEDIN,)

# ---- fetch layer -------------------------------------------------------------


@dataclass
class FetchResponse:
    """Raw fetch result; never raises out of the fetchers below."""

    url: str
    status: int = 0
    text: str = ""
    content_type: str = ""
    final_url: str = ""
    elapsed_ms: int = 0
    error: str = ""
    error_kind: str = ""  # timeout | network | ssl | browser | unknown


class Fetcher(Protocol):
    async def fetch(self, url: str) -> FetchResponse: ...


class AiohttpFetcher:
    """Small, dependency-light direct HTTP fetcher.

    A browser-like User-Agent is used because several Indonesian job boards
    return a different page (or a 403) for Python's default agent.  403/429
    and CAPTCHA pages surface later as ``BLOCKED`` status, never "FAILED".
    """

    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    )

    def __init__(self, timeout_seconds: float = 15.0, max_bytes: int = 2_500_000) -> None:
        self.timeout_seconds = max(1.0, float(timeout_seconds or 15.0))
        self.max_bytes = max(100_000, int(max_bytes))

    async def fetch(self, url: str) -> FetchResponse:
        import aiohttp

        started = time.perf_counter()
        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,application/json;q=0.9,"
                "text/plain;q=0.8,*/*;q=0.5"
            ),
            "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
            "Cache-Control": "no-cache",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url, allow_redirects=True) as response:
                    raw = await response.content.read(self.max_bytes)
                    charset = response.charset or "utf-8"
                    try:
                        text = raw.decode(charset, errors="ignore")
                    except LookupError:
                        text = raw.decode("utf-8", errors="ignore")
                    return FetchResponse(
                        url=url,
                        status=int(response.status),
                        text=text,
                        content_type=str(response.headers.get("Content-Type") or ""),
                        final_url=str(response.url),
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                    )
        except asyncio.TimeoutError:
            return FetchResponse(
                url=url,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"aiohttp timeout {self.timeout_seconds:.0f}s exceeded",
                error_kind="timeout",
            )
        except Exception as exc:  # aiohttp.ClientError, SSL, DNS, ...
            kind = "network"
            lower = f"{exc.__class__.__name__} {exc}".lower()
            if "ssl" in lower or "certificate" in lower:
                kind = "ssl"
            elif "timed out" in lower or "timeout" in lower:
                kind = "timeout"
            return FetchResponse(
                url=url,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"{exc.__class__.__name__}: {exc}",
                error_kind=kind,
            )


#: Extra hardening applied per page.  This is a focused subset of classical
#: stealth patches (no external dependency): hide ``navigator.webdriver``,
#: fake a real plugin/window-chrome surface and align languages/timezone so
#: Cloudflare-style probes see a normal visitor.
STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['id-ID', 'id', 'en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
window.chrome = {runtime: {}, loadTimes: () => ({}), csi: () => ({}), app: {name: 'Google Chrome'}};
const originalQuery = window.navigator.permissions.query;
window.navigator.permissions.query = (parameters) => (
  parameters.name === 'notifications'
    ? Promise.resolve({state: Notification.permission})
    : originalQuery(parameters)
);
"""


class PlaywrightFetcher:
    """Chromium-backed fetcher with a fixed 3-second SPA wait.

    One browser process serves a whole run; every ``fetch`` opens a fresh page
    in a shared context.  ``wait_ms`` defaults to 3000 per the internship
    brief and must not be reduced.  Launch failures surface as
    ``error_kind="browser"`` so the engine can fall back to HTTP cleanly.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 30.0,
        wait_ms: int = 3000,
        max_bytes: int | None = None,
    ) -> None:
        # SPA-heavy job boards frequently answer slowly behind anti-bot
        # tiers; keep a generous browser budget regardless of the plain-HTTP
        # timeout (NodeFlair has been observed taking >30 s to navigate).
        self.timeout_seconds = max(40.0, float(timeout_seconds or 40.0))
        self.wait_ms = wait_ms if wait_ms and wait_ms >= 1000 else 3000
        self.max_bytes = max_bytes or 4_000_000
        self._playwright = None
        self._browser = None
        self._context = None
        self._lock = asyncio.Lock()

    # -- lifecycle ---------------------------------------------------------

    async def _ensure(self) -> None:
        async with self._lock:
            if self._context is not None:
                return
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            try:
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=["--disable-blink-features=AutomationControlled"],
                )
            except Exception:
                await self._teardown()
                raise
            self._context = await self._browser.new_context(
                user_agent=AiohttpFetcher.USER_AGENT,
                locale="id-ID",
                viewport={"width": 1366, "height": 900},
                timezone_id="Asia/Jakarta",
            )

    async def _teardown(self) -> None:
        for closer in (
            getattr(self, "_context", None),
            getattr(self, "_browser", None),
            getattr(self, "_playwright", None),
        ):
            if closer is None:
                continue
            try:
                await closer.close()
            except Exception:  # pragma: no cover - shutdown best effort
                pass
        self._context = None
        self._browser = None
        self._playwright = None

    async def close(self) -> None:
        await self._teardown()

    # -- fetching ----------------------------------------------------------

    async def fetch(
        self,
        url: str,
        *,
        stealth: bool = True,
        wait_ms: int | None = None,
    ) -> FetchResponse:
        started = time.perf_counter()
        try:
            await self._ensure()
        except Exception as exc:
            return FetchResponse(
                url=url,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"Playwright tidak tersedia: {exc.__class__.__name__}: {exc}",
                error_kind="browser",
            )
        page_wait = int(wait_ms if wait_ms is not None else self.wait_ms)
        try:
            page = await self._context.new_page()
            if stealth:
                try:
                    await page.add_init_script(STEALTH_INIT_SCRIPT)
                except Exception:  # pragma: no cover - defensive
                    pass
            try:
                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self.timeout_seconds * 1000,
                )
                # Mandatory settle time for SPA hydration (per the contract).
                await page.wait_for_timeout(page_wait)
                status = int(response.status) if response is not None else 0
                text = await page.content()
                if len(text) > self.max_bytes:
                    text = text[: self.max_bytes]
                return FetchResponse(
                    url=url,
                    status=status,
                    text=text,
                    content_type="text/html",
                    final_url=page.url,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            except Exception as exc:
                lower = f"{exc.__class__.__name__} {exc}".lower()
                if "timeout" in lower:
                    kind = "timeout"
                elif "net::" in lower or "name not resolved" in lower or "refused" in lower:
                    kind = "network"
                else:
                    kind = "browser"
                return FetchResponse(
                    url=url,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                    error=f"{exc.__class__.__name__}: {exc}",
                    error_kind=kind,
                )
            finally:
                try:
                    await page.close()
                except Exception:  # pragma: no cover - already closing
                    pass
        except Exception as exc:  # context-level failure (browser died, ...)
            return FetchResponse(
                url=url,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"{exc.__class__.__name__}: {exc}",
                error_kind="browser",
            )


class MockFetcher:
    """Deterministic offline responses for every source A-G.

    The mock pages reproduce the *exact* card markup each parser looks for
    (attributes like ``data-automation="normalJob"``, ``itemprop="name"``,
    ``inner-box``, ``border border-gray-200``) — no hashed CSS classes.
    """

    async def fetch(self, url: str) -> FetchResponse:
        started = time.perf_counter()
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        key = self._source_key(host)
        if key == "linkedin":
            text = _mock_linkedin_results()
            content_type = "application/json"
        elif key:
            text = _mock_page(key, self._query_of(key, parsed))
            content_type = "text/html"
        else:
            return FetchResponse(
                url=url,
                status=404,
                text="",
                error="MockFetcher has no fixture for this URL",
                elapsed_ms=max(1, int((time.perf_counter() - started) * 1000)),
            )
        return FetchResponse(
            url=url,
            status=200,
            text=text,
            content_type=content_type,
            final_url=url,
            elapsed_ms=max(1, int((time.perf_counter() - started) * 1000)),
        )

    @staticmethod
    def _source_key(host: str) -> str:
        if host.endswith("jobstreet.com"):
            return "jobstreet"
        if "glints.com" in host:
            return "glints"
        if "kalibrr" in host:
            return "kalibrr"
        if host.endswith("its.ac.id") or "careers.its.ac.id" in host:
            return "its"
        if "kemdiktisaintek.go.id" in host:
            return "msib"
        if "nodeflair.com" in host:
            return "nodeflair"
        if "google." in host or "tavily" in host:
            return "linkedin"
        return ""

    @staticmethod
    def _query_of(key: str, parsed) -> str:
        qs = parse_qs(parsed.query or "")
        for param in ("keyword", "query", "cari", "keywords", "q"):
            if param in qs:
                return unescape(qs[param][0])
        path = parsed.path or ""
        if key == "jobstreet":
            match = re.match(r"/id/(.+)-intern-jobs", path)
            if match:
                return _smart_title(match.group(1).replace("-", " "))
        if key == "kalibrr":
            parts = [part for part in path.split("/") if part]
            if parts:
                tail = parts[-1]
                if "-intern-jobs" not in tail:
                    return _smart_title(unquote(tail).replace("-", " "))
        return "AI Engineer"


# ---- block detection ---------------------------------------------------------

_BLOCK_MARKERS = (
    "cloudflare",
    "cf-chl",
    "captcha",
    "verify you are human",
    "verify that you are not a robot",
    "access denied",
    "just a moment",
    "challenge-platform",
    "invalid browser",
    " Suspect automated",
    "ddos protection",
)
_BLOCK_STATUS = {403, 407, 429, 460, 509, 999}


def detect_block_reason(status: int, text: str) -> str:
    """Human-readable block reason, or empty string when not blocked."""
    lowered = (text or "")[:8000].casefold()
    if status in _BLOCK_STATUS:
        label = {
            403: "HTTP 403 Forbidden/CAPTCHA challenge",
            407: "HTTP 407 proxy auth required",
            429: "HTTP 429 rate limited",
            460: "HTTP 460 (anti-bot/Cloudflare)",
            509: "HTTP 509 bandwidth limited",
            999: "HTTP 999 LinkedIn anti-bot response",
        }[status]
        return label
    marker = next((m for m in _BLOCK_MARKERS if m.casefold() in lowered), None)
    if marker:
        return f"Blocked/anti-bot marker detected: {marker.strip()}"
    if status and status >= 400 and "cloudflare" in lowered:
        return f"HTTP {status} behind Cloudflare challenge"
    return ""


def _status_reason(status: int) -> str:
    labels = {
        400: "Bad request",
        401: "Authentication required",
        402: "Payment required",
        404: "HTTP 404 Not Found",
        410: "HTTP 410 Gone",
        511: "Network authentication required",
    }
    base = labels.get(status, f"HTTP {status} server error" if status >= 400 else "")
    return base


# ---- HTML parsing ------------------------------------------------------------


def _clean(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _element_lines(element) -> list[str]:
    """Visible inner text of one element split into non-empty lines."""
    if element is None:
        return []
    clone = element
    try:
        for tag in clone(["script", "style", "noscript", "svg"]):
            tag.decompose()
    except Exception:
        pass
    text = clone.get_text("\n", strip=True) or ""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _body_lines(html: str) -> list[str]:
    """Fallback contract: ``body`` inner text split into lines."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    body = soup.find("body") or soup
    return _element_lines(body)


def body_raw_text(html: str, limit: int = 16000) -> str:
    return "\n".join(_body_lines(html))[:limit]


def _first_text(element, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        found = element.select_one(selector)
        if found is not None:
            text = _clean(found.get_text(" ", strip=True))
            if text:
                return text
    return ""


def _ancestor_card(anchor, *, roles: tuple[str, ...] = ("article", "li")):
    """Nearest ancestor that looks like a listing card (has extra text)."""
    node = anchor
    for _ in range(6):
        if node.parent is None:
            break
        node = node.parent
        if getattr(node, "name", "") in ("html", "body", "[document]") or node.name is None:
            break
        lines = _element_lines(node)
        if len(lines) >= 2:
            return node
    return None


def _smart_title(text: str) -> str:
    """Title-case that preserves common tech acronyms (AI, ML, UI, ...)."""
    acronyms = {"ai", "ml", "ui", "ux", "qa", "ios", "sql", "html", "css", "api", "llm", "nlp"}
    words = []
    for word in str(text or "").split(" "):
        if word.lower() in acronyms:
            words.append(word.upper())
        else:
            words.append(word.title())
    return " ".join(words)


_CTA_LINE_RE = re.compile(
    r"^\s*(?:detail|details|selengkapnya|lihat|lihat detail|buka|apply|daftar|"
    r"view|view post|more|open)\b",
    re.IGNORECASE,
)


def _is_cta_line(line: str) -> bool:
    clean = _clean(line)
    return bool(len(clean) <= 40 and _CTA_LINE_RE.search(clean))


_SALARY_RE = re.compile(
    r"^(Rp|IDR|SGD|USD|MYR|THB|VND|PHP)\s?\d|^[\d.,]+\s?(juta|k\b)|^Not\s?Disclosed",
    re.IGNORECASE,
)


def _is_noise_line(line: str) -> bool:
    clean = _clean(line)
    if _is_cta_line(clean):
        return True
    return bool(_SALARY_RE.match(clean))


def _card_job(
    *,
    role: str,
    apply_url: str,
    card_lines: list[str],
    source_name: str,
    page_url: str,
    internship_hint: bool,
    company: str = "",
    location: str = "",
) -> RawJob:
    description = "\n".join(card_lines)[:1200]
    other_lines = [
        line for line in card_lines if line != role and not _is_noise_line(line)
    ]
    if not company and other_lines:
        company = other_lines[0]
    if not location and len(other_lines) > 1:
        location = other_lines[1]
    lowered = f"{role} {company} {description}".lower()
    if not internship_hint:
        internship_hint = any(w in lowered for w in ("intern", "magang", "internship"))
    mode = _work_mode_hint(card_lines)
    return RawJob(
        role_title=role,
        company_name=company,
        location=location,
        work_mode=mode,
        apply_url=apply_url,
        description=description,
        source_name=source_name,
        source_url=page_url,
        is_internship_hint=internship_hint,
    )


def _work_mode_hint(lines: list[str]) -> str:
    text = " ".join(lines).lower()
    if "hybrid" in text:
        return "Hybrid"
    if "remote" in text or "wfh" in text:
        return "Remote"
    if "wfo" in text or "on-site" in text or "onsite" in text:
        return "On-site"
    return ""


def _dedupe(jobs: list[RawJob]) -> list[RawJob]:
    seen: set[tuple[str, str, str]] = set()
    out: list[RawJob] = []
    for job in jobs:
        if not job.role_title or not job.apply_url:
            continue
        key = (
            job.role_title.casefold(),
            job.company_name.casefold(),
            job.apply_url.casefold(),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


def _role_from_text(value: str) -> str:
    """Normalise a candidate role line (salary noise, pipe separators)."""
    role = _clean(value)
    role = re.split(r"\s*\|\s*", role, maxsplit=1)[0].strip()
    # Glints prints salaries like "Rp 3-3.5Mio" right after/with the title.
    role = re.sub(r"\bRp\s?[\d.,]+[A-Za-z]*\b.*$", "", role).strip(" ·,-|")
    return role


def _role_from_lines(lines: list[str]) -> str:
    """First plausible role line of a card (skips salary/CTA lines)."""
    for line in lines:
        clean = _clean(line)
        if not clean or _is_cta_line(clean) or len(clean) > 120:
            continue
        if re.match(r"^Rp\s?\d", clean):
            continue
        return _role_from_text(clean)
    return ""


def parse_jobstreet(html: str, page_url: str, source_name: str) -> list[RawJob]:
    """A JobStreet: ``article[data-automation="normalJob"]`` cards."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    jobs: list[RawJob] = []
    for card in soup.select('article[data-automation="normalJob"]'):
        anchor = card.find("a", href=True) or card.select_one("a[aria-label][href]")
        if anchor is None:
            continue
        apply_url = str(urljoin(page_url or "", str(anchor.get("href") or "")))
        role = _first_text(
            card,
            (
                '[data-automation="jobTitle"]',
                'a[data-automation="jobTitle"]',
                "h3 a",
                "h2 a",
                "a[aria-label]",
            ),
        ) or _clean(anchor.get_text(" ", strip=True) or anchor.get("aria-label"))
        if not role:
            lines = _element_lines(card)
            role = lines[0] if lines else ""
        company = _first_text(card, ('[data-automation="jobCompany"]',)) or ""
        location = _first_text(card, ('[data-automation="jobLocation"]',)) or ""
        lines = _element_lines(card)
        jobs.append(
            _card_job(
                role=role,
                apply_url=apply_url,
                card_lines=lines,
                source_name=source_name,
                page_url=page_url,
                internship_hint=True,
                company=company,
                location=location,
            )
        )
    return _dedupe(jobs)


def parse_glints(html: str, page_url: str, source_name: str) -> list[RawJob]:
    """B Glints: job links under ``/opportunities/jobs/`` (never ``/explore``)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    jobs: list[RawJob] = []
    for anchor in soup.find_all("a", href=True):
        raw_href = str(anchor.get("href") or "")
        href = str(urljoin(page_url or "", raw_href))
        path = urlparse(href).path
        if "/opportunities/jobs/" not in path or path.rstrip("/").endswith("/explore"):
            continue
        if "explore" in path and "/jobs/" not in path:
            continue
        # Glints job links carry tracking params (utm_referrer, traceInfo);
        # canonicalise the apply URL so duplicates dedupe correctly.
        href = urlparse(href)._replace(query="").geturl()
        role = _role_from_lines(_element_lines(anchor))
        if not role:
            role = _role_from_text(
                _first_text(
                    anchor.parent if anchor.parent else anchor,
                    ('h3[data-automation="job-item-title"]', "h3", "h2", "h4"),
                )
                or _clean(anchor.get_text(" ", strip=True))
            )
        slug_company = ""
        if not role and re.search(r"-at-", path):
            # Slug fallback keeps div-only cards working; the "at-" part of a
            # Glints slug is usually the employer name.
            slug = path.rstrip("/").split("/")[-1]
            role_company = re.split(r"-at-", re.sub(r"^\d+-", "", slug))
            role = _smart_title(role_company[0].replace("-", " "))
            if len(role_company) > 1:
                slug_company = _smart_title(role_company[-1].replace("-", " "))
        if not role or _is_cta_line(role):
            continue
        card = _ancestor_card(anchor)
        lines = _element_lines(card) if card is not None else []
        # The company line often follows the role line inside the card.
        clean_lines = [line for line in lines if _clean(line) != _clean(role)]
        jobs.append(
            _card_job(
                role=role,
                apply_url=href,
                card_lines=clean_lines,
                source_name=source_name,
                page_url=page_url,
                internship_hint=True,
                company=slug_company if not clean_lines else "",
            )
        )
    return _dedupe(jobs)


def parse_kalibrr(html: str, page_url: str, source_name: str) -> list[RawJob]:
    """C Kalibrr: cards around ``a[itemprop="name"]``."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    jobs: list[RawJob] = []
    # Prefer job-position anchors (Kalibrr's structured cards), but fall back
    # to every a[itemprop="name"] — real pages embed extra itemprop anchors
    # (e.g. LinkedIn "View Post" widgets) that are filtered below.
    anchors = soup.select('a[itemprop="name"][data-id="job-position"]')
    if not anchors:
        anchors = soup.select('a[itemprop="name"]')
    else:
        extra = soup.select('a[itemprop="name"]')
        anchors = [*anchors, *extra]
    seen_urls: set[str] = set()
    for anchor in anchors:
        href = anchor.get("href")
        if not href:
            continue
        apply_url = str(urljoin(page_url or "", str(href)))
        role = _role_from_text(_clean(anchor.get_text(" ", strip=True)))
        if not role or _is_cta_line(role):
            continue
        if apply_url.casefold() in seen_urls:
            continue
        seen_urls.add(apply_url.casefold())
        card = _ancestor_card(anchor)
        lines = _element_lines(card) if card is not None else []
        clean_lines = [line for line in lines if _clean(line) != _clean(role)]
        jobs.append(
            _card_job(
                role=role,
                apply_url=apply_url,
                card_lines=clean_lines,
                source_name=source_name,
                page_url=page_url,
                internship_hint=True,
            )
        )
    return _dedupe(jobs)


def parse_its_career(html: str, page_url: str, source_name: str) -> list[RawJob]:
    """D ITS Career Center: ``div.inner-box`` cards with ``h4 > a`` titles."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    jobs: list[RawJob] = []
    for card in soup.select("div.inner-box"):
        anchor = card.select_one("h4 > a[href]") or card.select_one("h3 > a[href]")
        role = _role_from_text(
            _clean(anchor.get_text(" ", strip=True)) if anchor is not None else ""
        )
        if not role or _is_cta_line(role):
            continue
        apply_url = str(urljoin(page_url or "", str(anchor.get("href") or ""))) if anchor else ""
        lines = [
            line
            for line in _element_lines(card)
            if line != role
        ]
        jobs.append(
            _card_job(
                role=role,
                apply_url=apply_url,
                card_lines=lines,
                source_name=source_name,
                page_url=page_url,
                internship_hint=False,
            )
        )
    return _dedupe(jobs)


def parse_msib(html: str, page_url: str, source_name: str) -> list[RawJob]:
    """E MSIB: ``div.border.border-gray-200`` listing cards."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    jobs: list[RawJob] = []
    for card in soup.select("div.border.border-gray-200"):
        anchor = card.select_one("a[href]")
        role = _first_text(card, ("h3", "h2", "h4", "h5 td", "td"))
        if not role:
            lines = _element_lines(card)
            role = lines[0] if lines else ""
        if not role:
            continue
        apply_url = str(urljoin(page_url or "", str(anchor.get("href") or ""))) if anchor else ""
        lines = [line for line in _element_lines(card) if line != role]
        jobs.append(
            _card_job(
                role=role,
                apply_url=apply_url,
                card_lines=lines,
                source_name=source_name,
                page_url=page_url,
                internship_hint=True,
            )
        )
    return _dedupe(jobs)


def parse_nodeflair(html: str, page_url: str, source_name: str) -> list[RawJob]:
    """G NodeFlair: job links under ``/jobs/`` (cards may use hashed classes)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "lxml")
    jobs: list[RawJob] = []
    seen_urls: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        href = str(urljoin(page_url or "", str(anchor.get("href") or "")))
        path = urlparse(href).path
        if not re.search(r"^/jobs/[^/?#]+$", path):
            continue
        if href.casefold() in seen_urls:
            continue
        # NodeFlair job links carry tracking params (itm_*); keep the clean
        # canonical job URL for the report/apply table.
        parsed = urlparse(href)
        apply_url = parsed._replace(query="").geturl()
        seen_urls.add(apply_url.casefold())
        role = _clean(anchor.get_text(" ", strip=True))
        if not role:
            # Hashed utility classes are not reliable; derive the title from
            # the trailing slug instead of trusting a class name.
            slug = path.split("/")[-1]
            role = _smart_title(re.sub(r"^\d+-", "", slug).replace("-", " "))
        role = re.split(r"\s*\|\s*", role, maxsplit=1)[0].strip()
        role = re.sub(r"\s+\d{4,}$", "", role)
        card = _ancestor_card(anchor)
        lines = _element_lines(card) if card is not None else []
        clean_lines = [line for line in lines if line != role]
        jobs.append(
            _card_job(
                role=role,
                apply_url=apply_url,
                card_lines=clean_lines,
                source_name=source_name,
                page_url=page_url,
                internship_hint=True,
            )
        )
    return _dedupe(jobs)


_PARSERS = {
    "jobstreet": parse_jobstreet,
    "glints": parse_glints,
    "kalibrr": parse_kalibrr,
    "its_career": parse_its_career,
    "msib": parse_msib,
    "nodeflair": parse_nodeflair,
}


def parse_site(
    source_key: str,
    html: str,
    page_url: str,
    source_name: str,
) -> tuple[list[RawJob], str]:
    """Parse one HTML page.

    Returns ``(raw_jobs, raw_text)``.  When no site-specific card selector
    matches, the body inner text split into lines is returned as ``raw_text``
    (the LLM curator works on it directly).
    """
    parser = _PARSERS.get(source_key)
    if parser is None:
        raise ValueError(f"Parser tidak dikenal untuk source key: {source_key}")
    raw_jobs = parser(html, page_url, source_name)
    raw_text = body_raw_text(html)
    return raw_jobs, raw_text


# ---- content classification --------------------------------------------------


def content_from_response(
    response: FetchResponse,
    *,
    source_name: str,
    source_key: str,
    target_url: str,
    query: str,
) -> SourceContent:
    """Turn one :class:`FetchResponse` into a classified :class:`SourceContent`."""
    content = SourceContent(
        source_name=source_name,
        target_url=target_url,
        query=query,
        http_status_code=response.status,
        content_type=response.content_type,
        response_bytes=len((response.text or "").encode("utf-8", errors="ignore")),
        latency_ms=response.elapsed_ms,
        notes=(
            response.final_url
            if response.final_url and response.final_url != response.url
            else ""
        ),
    )
    if response.error_kind == "timeout":
        content.access_status = "TIMEOUT"
        content.failure_reason = response.error or "Network timeout"
        return content
    if response.error_kind == "browser":
        content.access_status = "BLOCKED"
        content.failure_reason = response.error or "Playwright gagal"
        return content
    if response.error:
        content.access_status = "BLOCKED"
        content.failure_reason = f"Network error: {response.error}"
        return content
    if response.status == 0:
        content.access_status = "BLOCKED"
        content.failure_reason = "Tidak ada respons HTTP (koneksi gagal)"
        return content
    block_reason = detect_block_reason(response.status, response.text)
    if block_reason:
        content.access_status = "BLOCKED"
        content.failure_reason = block_reason
        return content
    if response.status in (404, 410):
        content.access_status = "EMPTY_RESULT"
        content.failure_reason = f"HTTP {response.status}: halaman/res sumber tidak ada"
        return content
    if response.status >= 400:
        content.access_status = "BLOCKED"
        content.failure_reason = _status_reason(response.status)
        return content

    try:
        if source_key == "linkedin":
            payload = _json_or_fail(response.text)
            content.raw_jobs = parse_linkedin_api_results(
                payload, response.final_url or target_url, source_name
            )
            content.raw_text = _linkedin_snippets_text(payload)
        else:
            content.raw_jobs, content.raw_text = parse_site(
                source_key, response.text, target_url, source_name
            )
        content.raw_chars_scraped = len(content.raw_text) or len(
            "\n".join(str(job.description) for job in content.raw_jobs)
        )
    except Exception as exc:
        content.access_status = "BLOCKED"
        content.failure_reason = f"Parser error: {exc.__class__.__name__}: {exc}"
        return content

    if not content.raw_jobs and not content.raw_text.strip():
        content.access_status = "EMPTY_RESULT"
        content.failure_reason = "Halaman berhasil diambil tetapi tidak ada item/teks"
    elif not content.raw_jobs:
        content.access_status = "SUCCESS"
        content.failure_reason = (
            "Tidak ada kartu lowongan terstruktur; teks body akan dikurasi LLM"
        )
    else:
        content.access_status = "SUCCESS"
        content.failure_reason = ""
    return content


def _json_or_fail(text: str) -> dict:
    payload = json.loads(text or "{}")
    if not isinstance(payload, dict):
        raise TypeError("Respons API bukan JSON objek")
    return payload


async def fetch_source(
    *,
    source_name: str,
    source_key: str,
    target_url: str,
    query: str,
    fetcher: Fetcher,
) -> SourceContent:
    """Fetch one URL through ``fetcher`` and classify/parse the result."""
    response = await fetcher.fetch(target_url)
    return content_from_response(
        response,
        source_name=source_name,
        source_key=source_key,
        target_url=target_url,
        query=query,
    )


# ---- LinkedIn via Tavily/Exa (source F) --------------------------------------

TAVILY_API_URL = "https://api.tavily.com/search"
EXA_API_URL = "https://api.exa.ai/search"


def parse_linkedin_api_results(
    payload: dict,
    page_url: str,
    source_name: str,
) -> list[RawJob]:
    """Extract raw candidates from a Tavily/Exa search payload.

    Snippets are the raw data (kept verbatim); only results pointing to a
    LinkedIn job page become candidate :class:`RawJob` items.
    """
    results = payload.get("results") or []
    if isinstance(results, dict):
        results = list(results.values())
    raw_jobs: list[RawJob] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        title = _clean(item.get("title") or item.get("name"))
        snippet = _clean(
            item.get("content")
            or item.get("text")
            or item.get("snippet")
            or item.get("raw_content")
        )
        if not title:
            continue
        role = re.sub(r"\s*\|\s*LinkedIn.*$", "", title, flags=re.IGNORECASE).strip()
        if not role:
            continue
        lowered_url = url.lower()
        if "linkedin.com/jobs" not in lowered_url:
            # Non-job result: its snippet still feeds the curation raw text.
            continue
        if "/jobs/view/" not in lowered_url:
            # Keep single vacancies only; "…-jobs" aggregate/listing pages and
            # "/jobs/search" URLs are not one apply-able internship.  Their
            # snippets remain in the raw text for LLM curation.
            continue
        raw_jobs.append(
            RawJob(
                role_title=role[:180],
                company_name="",
                apply_url=url,
                description=snippet[:1200],
                source_name=source_name,
                source_url=page_url,
                is_internship_hint=True,
                work_mode=_work_mode_hint([snippet, role]),
            )
        )
    return raw_jobs


def _linkedin_snippets_text(payload: dict, limit: int = 12000) -> str:
    results = payload.get("results") or []
    if isinstance(results, dict):
        results = list(results.values())
    blocks: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        title = _clean(item.get("title") or "")
        url = _clean(item.get("url") or "")
        snippet = _clean(
            item.get("content")
            or item.get("text")
            or item.get("snippet")
            or item.get("raw_content")
        )
        blocks.append(f"{title}\n{url}\n{snippet}".strip())
    return "\n\n".join(block for block in blocks if block)[:limit]


async def _http_json_post(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
    timeout_seconds: float,
) -> dict:
    import aiohttp

    timeout = aiohttp.ClientTimeout(total=max(5.0, timeout_seconds))
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {text[:200]}")
            return _json_or_fail(text)


async def search_linkedin_dork(
    query: str,
    *,
    tavily_key: str = "",
    exa_key: str = "",
    timeout_seconds: float = 30.0,
) -> tuple[list[RawJob], str, str]:
    """Run the LinkedIn dork through Tavily first, then Exa.

    Returns ``(raw_jobs, raw_text, provider)``.  Raises when both providers
    fail; the engine classifies that on its own.
    """
    dork = linkedin_dork(query)
    errors: list[str] = []
    if tavily_key:
        try:
            payload = await _http_json_post(
                TAVILY_API_URL,
                headers={"Content-Type": "application/json"},
                payload={
                    "api_key": tavily_key,
                    "query": dork,
                    "search_depth": "basic",
                    "max_results": 10,
                    "include_answer": False,
                    "include_raw_content": False,
                },
                timeout_seconds=timeout_seconds,
            )
            results = parse_linkedin_api_results(payload, linkedin_dork_url(query), "LinkedIn Jobs")
            return results, _linkedin_snippets_text(payload), "Tavily"
        except Exception as exc:
            errors.append(f"Tavily: {exc.__class__.__name__}: {exc}")
            log.warning("tavily linkedin search gagal: %s", exc)
    if exa_key:
        try:
            payload = await _http_json_post(
                EXA_API_URL,
                headers={
                    "x-api-key": exa_key,
                    "Content-Type": "application/json",
                },
                payload={
                    "query": dork,
                    "numResults": 10,
                    "type": "keyword",
                    "contents": {"text": {"maxCharacters": 800}},
                },
                timeout_seconds=timeout_seconds,
            )
            results = parse_linkedin_api_results(payload, linkedin_dork_url(query), "LinkedIn Jobs")
            return results, _linkedin_snippets_text(payload), "Exa"
        except Exception as exc:
            errors.append(f"Exa: {exc.__class__.__name__}: {exc}")
            log.warning("exa linkedin search gagal: %s", exc)
    raise RuntimeError(
        "LinkedIn dorking gagal tanpa API"
        + (f" ({'; '.join(errors)})" if errors else "Tanpa TAVILY_API_KEY/EXA_API_KEY")
    )


# ---- mock fixtures -----------------------------------------------------------


def _mock_page(key: str, query: str) -> str:
    clean_query = _clean(query).replace('"', "")
    company = "Nusantara Tech"
    location = "Jakarta"
    if key == "jobstreet":
        slug = re.sub(r"[^a-z0-9]+", "-", clean_query.casefold()).strip("-")
        return f"""<!doctype html><html><head><title>JobStreet mock</title></head>
<body class="brs-detail ss-container">
<div class="css-hardcode-decoy unrelated">Navigation filler text</div>
<article class="ss-card css-decoy" data-automation="normalJob">
  <h3 data-automation="jobTitle"><a href="https://id.jobstreet.com/id/job/74000001-{slug}">{clean_query}</a></h3>
  <span data-automation="jobCompany">{company}</span>
  <span data-automation="jobLocation">{location}</span>
  <a data-automation="jobAd" href="https://id.jobstreet.com/id/job/74000001-{slug}">Lihat detail</a>
</article>
<article class="ss-card" data-automation="normalJob">
  <div>
    <a href="https://id.jobstreet.com/id/job/74000002-{slug}-2">Data Analyst Intern</a>
  </div>
  <span data-automation="jobCompany">Speedy Apply</span>
  <span data-automation="jobLocation">Bandung</span>
</article>
</body></html>"""
    if key == "glints":
        return f"""<!doctype html><html><head><title>Glints mock</title></head>
<body><div class="css-hardcode-decoy">Explore-only link should be ignored</div>
<a href="https://glints.com/id/en/opportunities/jobs/explore?keyword=x">Explore</a>
<div class="job-card">
  <h3 data-automation="job-item-title"><a href="https://glints.com/id/opportunities/jobs/123-{re.sub('[^a-z0-9]+', '-', clean_query.casefold())}">{clean_query}</a></h3>
  <div>{company}</div>
  <div>{location}</div>
</div>
<div class="job-card">
  <h3><a href="https://glints.com/id/opportunities/jobs/124-data-analyst-intern-at-speedy">Data Analyst Intern</a></h3>
  <div>Speedy Apply</div>
</div>
</body></html>"""
    if key == "kalibrr":
        return f"""<!doctype html><html><head><title>Kalibrr mock</title></head>
<body><div class="j-card">
  <a itemprop="name" data-id="job-position" href="https://www.kalibrr.id/c/{company.lower().replace(' ', '')}/jobs/123-{re.sub('[^a-z0-9]+', '-', clean_query.casefold())}">{clean_query}</a>
  <div class="j-links"><span>{company}</span><span>{location}</span></div>
</div>
<div class="j-card">
  <a itemprop="name" href="https://www.kalibrr.id/c/speedy/jobs/124-data-analyst-intern">Data Analyst Intern</a>
  <div class="j-links"><span>Speedy Apply</span><span>Bandung</span></div>
</div>
</body></html>"""
    if key == "its":
        return f"""<!doctype html><html><head><title>ITS Career Center mock</title></head>
<body><div class="css-hardcode-decoy">
<div class="inner-box">
  <h4><a href="https://careers.its.ac.id/lowongan/detail/74000001">{clean_query}</a></h4>
  <p>{company}</p>
  <p>{location}</p>
</div>
<div class="inner-box">
  <h4><a href="https://careers.its.ac.id/lowongan/detail/74000002">Data Analyst Intern</a></h4>
  <p class="mt-1">Speedy Apply</p>
</div>
</div></body></html>"""
    if key == "msib":
        return f"""<!doctype html><html><head><title>MSIB mock</title></head>
<body>
<div class="border border-gray-200 rounded-2xl p-4">
  <h3>{clean_query}</h3>
  <p>{company}</p>
  <p>{location}</p>
  <a href="/magang/lowongan/74000001">Detail lowongan</a>
</div>
<div class="border border-gray-200">
  <h3>Data Analyst Intern</h3>
  <p>Speedy Apply</p>
  <a href="https://simbelmawa.kemdiktisaintek.go.id/magang/lowongan/74000002">Detail lowongan</a>
</div>
</body></html>"""
    if key == "nodeflair":
        return f"""<!doctype html><html><head><title>NodeFlair mock</title></head>
<body><div class="css-hardcode-decoy unrelated">Sitemap links</div>
<div class="job-listing">
  <div class="job-card css-decoy-hash">
    <a href="https://nodeflair.com/jobs/{'123-' + re.sub('[^a-z0-9]+', '-', clean_query.casefold())}">{clean_query}</a>
    <div>{company}</div>
  </div>
  <div class="job-card">
    <a href="https://nodeflair.com/jobs/124-data-analyst-intern">Data Analyst Intern</a>
    <div>Speedy Apply</div>
  </div>
  <a href="https://nodeflair.com/jobs?query=explore">Jobs search (bukan lowongan)</a>
</div>
</body></html>"""
    raise ValueError(f"mock tidak tersedia untuk source key: {key}")


def _mock_linkedin_results() -> str:
    payload = {
        "query": 'site:linkedin.com/jobs "internship" "AI Engineer" Indonesia',
        "results": [
            {
                "title": "AI Engineer Intern | LinkedIn",
                "url": "https://www.linkedin.com/jobs/view/4123456789",
                "content": "Nusantara Tech hyr AI Engineer Intern di Jakarta. "
                "Wajib Python, ML dasar. Magang 6 bulan, hybrid.",
            },
            {
                "title": "Data Analyst Intern | LinkedIn",
                "url": "https://www.linkedin.com/jobs/view/4123456790",
                "content": "Speedy Apply mencari Data Analyst Intern remote (WFH), "
                "skill SQL dan dashboard.",
            },
            {
                "title": "117 pekerjaan Data Analyst Intern di Indonesia",
                "url": "https://id.linkedin.com/jobs/data-analyst-intern-jobs",
                "content": "Halaman listing agregat (bukan satu lowongan).",
            },
            {
                "title": "Bukan LinkedIn Jobs",
                "url": "https://www.linkedin.com/feed/update/abc",
                "content": "Cuplikan tetap dikirim sebagai teks mentah ke kurator.",
            },
        ],
    }
    return json.dumps(payload, ensure_ascii=False)
