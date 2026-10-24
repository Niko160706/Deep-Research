"""Unit tests for the hybrid aggregator, dedup gate, and region mapping."""
from __future__ import annotations

import pytest

from drx.models import SearchResult
from drx.muscle.backends import (
    ExaSearchBackend,
    HybridSearchBackend,
    SearXNGBackend,
    dedupe_results,
    normalize_url,
    region_to_language,
)


# ---- URL normalization & dedup gate ----
def test_normalize_url_strips_tracking_params() -> None:
    norm = normalize_url("https://example.com/page?utm_source=x&utm_medium=y&id=123")
    assert "utm_source" not in norm
    assert "utm_medium" not in norm
    assert "id=123" in norm


def test_normalize_url_lowercases_host_and_strips_fragment() -> None:
    assert normalize_url("HTTPS://Example.COM/Path#frag") == "https://example.com/Path"


def test_dedupe_results_collapses_tracking_duplicates() -> None:
    a = SearchResult(url="https://example.com/a?utm_source=x", title="A", snippet="short")
    b = SearchResult(url="https://example.com/a?utm_source=y", title="A", snippet="a longer snippet")
    out = dedupe_results([a, b])
    assert len(out) == 1
    assert out[0].snippet == "a longer snippet"


# ---- region lock ----
def test_region_to_language() -> None:
    assert region_to_language("id") == "id-ID"
    assert region_to_language("us") == "en-US"
    assert region_to_language("jp") == "ja-JP"
    assert region_to_language("global") == ""
    assert region_to_language("") == ""


# ---- parsers ----
def test_searxng_parse() -> None:
    data = {"results": [
        {"url": "https://example.com/a", "title": "A", "content": "content A", "score": 0.5},
    ]}
    out = SearXNGBackend._parse(data, 10)
    assert len(out) == 1
    assert out[0].title == "A"
    assert out[0].snippet == "content A"
    assert out[0].score == 0.5


def test_exa_parse() -> None:
    data = {"results": [
        {"url": "https://example.com/a", "title": "A", "text": "text A", "score": 0.7},
    ]}
    out = ExaSearchBackend._parse(data, 10)
    assert len(out) == 1
    assert out[0].snippet == "text A"
    assert out[0].score == 0.7


# ---- hybrid aggregator ----
class _FakeBackend:
    name = "fake"

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        return self.results


@pytest.mark.asyncio
async def test_hybrid_merges_and_dedupes() -> None:
    b1 = _FakeBackend([SearchResult(url="https://a.com/x?utm_source=1", title="X", snippet="one")])
    b2 = _FakeBackend([
        SearchResult(url="https://a.com/x", title="X", snippet="one longer"),
        SearchResult(url="https://b.com/y", title="Y", snippet="two"),
    ])
    hybrid = HybridSearchBackend(backends=[b1, b2])
    out = await hybrid.search("q", limit=10)
    assert len(out) == 2  # a.com/x deduped
    assert {r.url for r in out} == {"https://a.com/x", "https://b.com/y"}


@pytest.mark.asyncio
async def test_hybrid_returns_empty_when_all_empty() -> None:
    hybrid = HybridSearchBackend(backends=[_FakeBackend([]), _FakeBackend([])])
    assert await hybrid.search("q", limit=10) == []


@pytest.mark.asyncio
async def test_searxng_disables_itself_after_connection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drx.muscle import backends as backend_module

    attempts = 0

    def _boom(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("connection refused")

    monkeypatch.setattr(backend_module.aiohttp, "ClientSession", _boom)
    backend = SearXNGBackend(base_url="http://127.0.0.1:8080", timeout=0.1)

    assert await backend.search("first") == []
    assert backend.disabled is True
    assert "connection refused" in backend.disabled_reason

    # No second connection attempt: the provider stays off for this session.
    assert await backend.search("second") == []
    assert attempts == 1


@pytest.mark.asyncio
async def test_searxng_disables_itself_after_5xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from drx.muscle import backends as backend_module

    class _Response:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class _Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def get(self, *args, **kwargs):
            return _Response()

    monkeypatch.setattr(backend_module.aiohttp, "ClientSession", _Session)
    backend = SearXNGBackend(base_url="http://127.0.0.1:8080", timeout=0.1)
    assert await backend.search("query") == []
    assert backend.disabled is True
    assert "503" in backend.disabled_reason


@pytest.mark.asyncio
async def test_hybrid_skips_a_disabled_searxng_backend() -> None:
    class _DisabledBackend:
        name = "searxng"
        disabled = True

        def __init__(self) -> None:
            self.calls = 0

        async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
            self.calls += 1
            return []

    disabled = _DisabledBackend()
    live = _FakeBackend(
        [SearchResult(url="https://example.com/a", title="A", snippet="A")]
    )
    hybrid = HybridSearchBackend(backends=[disabled, live])
    out = await hybrid.search("q", limit=10)
    assert [r.url for r in out] == ["https://example.com/a"]
    assert disabled.calls == 0
