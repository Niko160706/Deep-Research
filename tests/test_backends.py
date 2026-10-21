"""Unit tests for search backends: Bing URL decoding and failover."""
from __future__ import annotations

import pytest

from drx.models import SearchResult
from drx.muscle.backends import (
    BingSearchBackend,
    BraveSearchBackend,
    FailoverSearchBackend,
    MockSearchBackend,
    TavilySearchBackend,
    canonicalize_url,
    dedupe_results,
)


def test_tavily_parse_converts_json_to_results() -> None:
    payload = {
        "query": "Niko Rajani Syahputra Pane",
        "results": [
            {
                "title": "Niko - Wikipedia",
                "url": "https://en.wikipedia.org/wiki/Niko",
                "content": "Niko is a given name...",
                "score": 0.85,
            },
            {
                "title": "Niko Electronics",
                "url": "https://www.nikoelectronic.com/",
                "content": "Modern appliance brand...",
                "score": 0.42,
            },
        ],
    }
    results = TavilySearchBackend._parse(payload, 10)
    assert len(results) == 2
    assert results[0].title == "Niko - Wikipedia"
    assert results[0].domain == "en.wikipedia.org"
    assert results[0].snippet == "Niko is a given name..."
    assert results[0].score == 0.85


@pytest.mark.asyncio
async def test_tavily_missing_key_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    backend = TavilySearchBackend(api_key=None)
    assert await backend.search("anything") == []


def test_brave_parse_converts_json_to_results() -> None:
    payload = {
        "web": {
            "results": [
                {
                    "title": "Fall of the Western Roman Empire",
                    "url": "https://en.wikipedia.org/wiki/Fall_of_the_Western_Roman_Empire",
                    "description": "The fall of the Western Roman Empire...",
                },
                {
                    "title": "Roman Empire",
                    "url": "https://www.britannica.com/place/Roman-Empire",
                    "description": "Roman Empire, the ancient empire...",
                },
            ]
        }
    }
    results = BraveSearchBackend._parse(payload, 10)
    assert len(results) == 2
    assert results[0].title == "Fall of the Western Roman Empire"
    assert results[0].domain == "en.wikipedia.org"
    assert results[0].snippet == "The fall of the Western Roman Empire..."
    assert results[1].domain == "www.britannica.com"


@pytest.mark.asyncio
async def test_brave_missing_key_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("BRAVE_SEARCH_API_KEY", raising=False)
    backend = BraveSearchBackend(api_key=None)
    assert await backend.search("anything") == []


def test_bing_extract_url_decodes_redirect() -> None:
    # Real Bing organic results carry an 'a1' prefix before the base64 payload.
    href = (
        "https://www.bing.com/ck/a?!&&p=abc&u=a1aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvQWxhbl9UdXJpbmc"
        "&ntb=1"
    )
    assert BingSearchBackend._extract_url(href) == "https://en.wikipedia.org/wiki/Alan_Turing"


def test_bing_extract_url_decodes_without_prefix() -> None:
    href = (
        "https://www.bing.com/ck/a?!&&p=abc&u=aHR0cHM6Ly9lbi53aWtpcGVkaWEub3JnL3dpa2kvQWxhbl9UdXJpbmc"
        "&ntb=1"
    )
    assert BingSearchBackend._extract_url(href) == "https://en.wikipedia.org/wiki/Alan_Turing"


def test_bing_extract_url_passthrough_non_bing() -> None:
    assert BingSearchBackend._extract_url("https://example.com/x") == "https://example.com/x"


class _EmptyBackend:
    name = "empty"

    async def search(self, query: str, limit: int = 10) -> list:
        return []


@pytest.mark.asyncio
async def test_failover_falls_back_to_next_backend() -> None:
    backend = FailoverSearchBackend([_EmptyBackend(), MockSearchBackend()])
    results = await backend.search("anything")
    assert len(results) == 2


class _AlwaysEmptyBackend:
    name = "always_empty"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, limit: int = 10) -> list:
        self.calls += 1
        return []


@pytest.mark.asyncio
async def test_failover_circuit_breaker_marks_down() -> None:
    empty = _AlwaysEmptyBackend()
    backend = FailoverSearchBackend([empty, MockSearchBackend()], max_failures=2)
    first = await backend.search("x")
    assert len(first) == 2
    second = await backend.search("x")
    assert len(second) == 2
    assert "always_empty" in backend._down
    assert empty.calls == 2


# ---- URL canonicalization (mobile/AMP variants + noise params) ----

def test_canonicalize_url_strips_mobile_subdomains() -> None:
    assert (
        canonicalize_url("https://m.antaranews.com/berita/x?amp=1")
        == "https://antaranews.com/berita/x"
    )
    assert (
        canonicalize_url("https://www.antaranews.com/berita/x?showpage=all")
        == "https://antaranews.com/berita/x"
    )
    assert canonicalize_url("https://amp.kompas.com/x/") == "https://kompas.com/x"
    assert (
        canonicalize_url("https://mobile.detik.com/x?utm_source=wa&id=7")
        == "https://detik.com/x?id=7"
    )


def test_canonicalize_url_keeps_meaningful_params() -> None:
    assert canonicalize_url("https://site.id/berita?id=123") == "https://site.id/berita?id=123"
    # pagination selects different content -> must NOT be stripped
    assert "page=3" in canonicalize_url("https://site.id/daftar?page=3")


def test_dedupe_results_collapses_mobile_and_desktop_variants() -> None:
    mobile = SearchResult(url="https://m.antaranews.com/berita/x?amp=1", title="A", score=0.4)
    desktop = SearchResult(url="https://antaranews.com/berita/x", title="B", score=0.9)
    out = dedupe_results([mobile, desktop])
    assert len(out) == 1
    assert out[0].score == 0.9  # richest/highest-scored copy wins
