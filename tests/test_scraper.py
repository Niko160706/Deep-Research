"""Unit tests for scraper helpers (visible-text estimate + block detection)."""
from __future__ import annotations

import socket

import aiohttp
import pytest

from drx.models import Document, SearchResult
from drx.muscle.scraper import (
    Scraper,
    _looks_blocked,
    _visible_text,
    is_high_value,
    is_pdf_url,
    needs_insecure_tls,
)


def test_visible_text_strips_scripts_and_tags() -> None:
    html = (
        "<html><head><script>var x = 1;</script><style>.a{}</style></head>"
        "<body><nav>menu</nav><p>Hello world this is real content</p></body></html>"
    )
    text = _visible_text(html)
    assert "var x" not in text
    assert "Hello world this is real content" in text


def test_looks_blocked_detects_captcha() -> None:
    assert _looks_blocked("<html><body>Please verify you are human (captcha)</body></html>") is True


def test_looks_blocked_false_for_normal_page() -> None:
    assert _looks_blocked("<html><body><p>A long article about history.</p></body></html>") is False


# ---- Playwright timeout: longer budget over the TOR SOCKS5 proxy ----

def test_playwright_timeout_extends_for_tor() -> None:
    scraper = Scraper()
    assert scraper.playwright_timeout_ms == 30000
    assert scraper._playwright_timeout(via_tor=False) == 30000
    assert scraper._playwright_timeout(via_tor=True) == 45000


# ---- PDF handling: aiohttp only, never Playwright ----

def _sample_pdf(text: str = "Hello PDF Ainun Najib") -> bytes:
    """Build a tiny text-bearing PDF in memory (no extra dependency)."""
    import io as _io

    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    fonts = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): writer._add_object(fonts)}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 14 Tf 40 200 Td ({text}) Tj ET\n".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = _io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_pdf_detection_by_extension_and_content_type() -> None:
    assert is_pdf_url("https://x.go.id/dokumen/laporan.pdf") is True
    assert is_pdf_url("https://x.go.id/dokumen/laporan.PDF") is True
    assert is_pdf_url("https://x.go.id/unduh?id=7", "application/pdf") is True
    assert is_pdf_url("https://x.go.id/index.html") is False
    assert is_pdf_url("https://x.go.id/pdflike") is False


def test_extract_pdf_text_reads_real_pdf_and_survives_garbage() -> None:
    assert "Ainun Najib" in Scraper._extract_pdf_text(_sample_pdf())
    assert Scraper._extract_pdf_text(b"not a pdf at all") == ""


def test_high_value_domains_and_tls_waiver() -> None:
    assert is_high_value("https://sinta.kemdiktisaintek.go.id/authors/123") is True
    assert is_high_value("https://univ.ac.id/berita") is True
    assert is_high_value("https://sleman.go.id/profil") is True
    assert is_high_value("https://www.imdb.com/title/tt1/") is True
    assert is_high_value("https://letterboxd.com/film/x/") is True
    assert is_high_value("https://en.wikipedia.org/wiki/X") is True
    assert is_high_value("https://example.com/x") is False
    assert is_high_value("https://notimdb.com/title/tt1/") is False
    assert is_high_value("https://fake-ac.id.evil.com/x") is False
    assert needs_insecure_tls("https://univ.ac.id/x") is True
    assert needs_insecure_tls("https://sleman.go.id/x") is True
    assert needs_insecure_tls("https://example.com/x") is False


def test_needs_render_thresholds_are_domain_aware() -> None:
    """High-value domains escalate at <200 chars; others keep the 500 bar."""
    scraper = Scraper(min_content_chars=500, high_value_min_chars=200)
    short = Document(url="https://univ.ac.id/x", text="<html><body>hi</body></html>")
    mid = Document(
        url="https://univ.ac.id/x",
        text="<html><body>" + ("kata " * 60) + "</body></html>",  # ~300 chars
    )
    assert scraper._needs_render(short, short.url) is True
    assert scraper._needs_render(mid, mid.url) is False  # 300 >= 200 -> no render
    assert scraper._needs_render(mid, "https://example.com/x") is True  # < 500
    assert scraper._needs_render(None, "https://example.com/x") is True


@pytest.mark.asyncio
async def test_pdf_is_fetched_with_aiohttp_and_never_goes_to_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiohttp import web

    pdf_bytes = _sample_pdf()

    async def handler(_: web.Request) -> web.Response:
        return web.Response(body=pdf_bytes, content_type="application/pdf")

    app = web.Application()
    app.router.add_get("/dokumen.pdf", handler)
    app.router.add_get("/unduh", handler)  # PDF without a .pdf extension
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    scraper = Scraper(timeout=5.0, enable_playwright=True)
    playwright_calls: list[str] = []

    async def fake_playwright(url: str, use_tor: bool = False):
        playwright_calls.append(url)
        raise AssertionError("Playwright must never receive a PDF")

    monkeypatch.setattr(scraper, "_fetch_playwright", fake_playwright)
    try:
        for path in ("/dokumen.pdf", "/unduh"):
            doc = await scraper._fetch_one(
                SearchResult(url=f"http://127.0.0.1:{port}{path}", title="Laporan ASN")
            )
            assert doc.method == "pdf"
            assert "Ainun Najib" in doc.text
            assert doc.title == "Laporan ASN"
    finally:
        await runner.cleanup()

    assert playwright_calls == []


@pytest.mark.asyncio
async def test_unparsable_pdf_is_dropped_without_playwright(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aiohttp import web

    async def handler(_: web.Request) -> web.Response:
        return web.Response(body=b"%PDF-1.3 broken", content_type="application/pdf")

    app = web.Application()
    app.router.add_get("/rusak.pdf", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    scraper = Scraper(timeout=5.0, enable_playwright=True)
    playwright_calls: list[str] = []

    async def fake_playwright(url: str, use_tor: bool = False):
        playwright_calls.append(url)

    monkeypatch.setattr(scraper, "_fetch_playwright", fake_playwright)
    try:
        with pytest.raises(RuntimeError):
            await scraper._fetch_one(
                SearchResult(url=f"http://127.0.0.1:{port}/rusak.pdf")
            )
    finally:
        await runner.cleanup()

    assert playwright_calls == []


@pytest.mark.asyncio
async def test_tls_waived_connector_for_ac_id() -> None:
    """ac.id/go.id get a TLS-waiving connector; other hosts get the default."""
    scraper = Scraper(timeout=5.0)
    insecure = scraper._new_connector("https://univ.ac.id/x", use_tor=False)
    assert insecure is not None
    assert insecure._ssl is False
    assert scraper._new_connector("https://example.com/x", use_tor=False) is None
    await insecure.close()


# ---- "page is navigating" resilience ----

class _FakePage:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0
        self.load_state_calls = 0

    async def content(self) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(
                "Unable to retrieve content because the page is navigating"
            )
        return "<html>settled</html>"

    async def wait_for_load_state(self, state: str, timeout: int = 0) -> None:
        self.load_state_calls += 1


@pytest.mark.asyncio
async def test_safe_content_recovers_from_navigation_race() -> None:
    """Transient mid-navigation reads are retried, not surfaced as failures."""
    scraper = Scraper(content_retry_backoff=0.0)
    page = _FakePage(failures=2)
    html = await scraper._safe_content(page)
    assert html == "<html>settled</html>"
    assert page.calls == 3
    assert page.load_state_calls == 2


@pytest.mark.asyncio
async def test_safe_content_raises_when_page_never_settles() -> None:
    scraper = Scraper(content_retry_backoff=0.0)
    page = _FakePage(failures=99)
    with pytest.raises(RuntimeError, match="page.content failed"):
        await scraper._safe_content(page)
    assert page.calls == 3


# ---- aiohttp/TOR session lifecycle ----

@pytest.mark.asyncio
async def test_new_tor_connector_is_fresh_per_call() -> None:
    """A shared connector dies with the first session ("Session is closed")."""
    pytest.importorskip("aiohttp_socks")
    scraper = Scraper(tor_proxy="socks5://127.0.0.1:9050")
    first = scraper._new_tor_connector()
    second = scraper._new_tor_connector()
    try:
        assert first is not None
        assert second is not None
        assert first is not second
    finally:
        for connector in (first, second):
            if connector is not None:
                await connector.close()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.mark.asyncio
async def test_sequential_tor_attempts_survive_connector_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: retried TOR fetches must not inherit a closed session.

    Before the fix the SOCKS connector was shared, so the first session closed
    it and every later attempt failed with "Session is closed" /
    "Connector is closed." Each attempt must own (and close) its connector.
    """
    from aiohttp import web

    page_html = "<html><body><p>hello world this is real tor content</p></body></html>"

    async def handler(_: web.Request) -> web.Response:
        return web.Response(text=page_html, content_type="text/html")

    app = web.Application()
    app.router.add_get("/", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = _free_port()
    await web.TCPSite(runner, "127.0.0.1", port).start()

    scraper = Scraper(timeout=5.0)
    made: list[aiohttp.TCPConnector] = []

    def fresh_connector(insecure_tls: bool = False) -> aiohttp.TCPConnector:
        connector = aiohttp.TCPConnector(ssl=False if insecure_tls else None)
        made.append(connector)
        return connector

    monkeypatch.setattr(scraper, "_new_tor_connector", fresh_connector)
    try:
        for _ in range(2):
            blocked, doc = await scraper._aiohttp_once(
                f"http://127.0.0.1:{port}/", use_tor=True
            )
            assert blocked is False
            assert doc is not None
            assert "real tor content" in doc.text
    finally:
        await runner.cleanup()

    assert len(made) == 2
    assert all(connector.closed for connector in made)
