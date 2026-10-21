"""Unit tests for AutoSearch scoring, filtering, and dedup."""
from __future__ import annotations

import pytest

from drx.models import SearchResult
from drx.muscle.autosearch import AutoSearch
from drx.muscle.backends import MockSearchBackend


@pytest.mark.asyncio
async def test_mock_search_returns_results() -> None:
    autosearch = AutoSearch(backend=MockSearchBackend(), results_per_query=10)
    results = await autosearch.search("entity")
    assert len(results) == 2
    assert all(r.score > 0 for r in results)


@pytest.mark.asyncio
async def test_dedup_across_repeated_queries() -> None:
    autosearch = AutoSearch(backend=MockSearchBackend(), results_per_query=10)
    first = await autosearch.search("entity")
    assert len(first) == 2
    # Same backend returns identical URLs -> all filtered as seen.
    assert await autosearch.search("entity") == []


def test_is_anchored_detects_70_30_queries() -> None:
    assert AutoSearch.is_anchored('"Niko" AND ("Surabaya" OR "Informatika")')
    assert AutoSearch.is_anchored('"Niko Rajani" AND ("Surabaya")')
    assert not AutoSearch.is_anchored('"Nur Royul Amin"')
    assert not AutoSearch.is_anchored("roman empire fall")


@pytest.mark.asyncio
async def test_anchored_query_passes_through_unwrapped() -> None:
    seen: list[str] = []

    class _RecordingBackend:
        name = "recording"

        async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
            seen.append(query)
            return []

    autosearch = AutoSearch(backend=_RecordingBackend(), results_per_query=10)
    anchored = '"Nur Royul Amin" AND ("Surabaya" OR "Informatika")'
    await autosearch.search(anchored)
    assert seen == [anchored]  # not double-quoted as a name


@pytest.mark.asyncio
async def test_irrelevant_results_are_filtered_by_min_score() -> None:
    class _GarbageBackend:
        name = "garbage"

        async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
            return [
                SearchResult(
                    url="https://movies.example.com/warkop",
                    title="WARKOP DKI full movie",
                    snippet="Indonesian comedy film",
                    domain="movies.example.com",
                )
            ]

    autosearch = AutoSearch(backend=_GarbageBackend(), results_per_query=10)
    assert await autosearch.search("roman empire fall") == []


def test_junk_tld_filtered() -> None:
    assert AutoSearch._is_junk("http://spammy.xyz/page") is True
    assert AutoSearch._is_junk("https://en.wikipedia.org/wiki/Test") is False


# ---- Global URL dedup cache (shared between anchored 70% and exploratory 30%) ----

class _MobileVariantBackend:
    """Returns the same article as a mobile/AMP URL and a desktop URL."""

    name = "mobile-variant"
    supports_boolean = True

    async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
        _ = query, limit
        return [
            SearchResult(
                url="https://m.antaranews.com/berita/1?amp=1",
                title="entity berita",
                snippet="entity",
                domain="m.antaranews.com",
            ),
            SearchResult(
                url="https://antaranews.com/berita/1",
                title="entity berita",
                snippet="entity",
                domain="antaranews.com",
            ),
        ]


@pytest.mark.asyncio
async def test_mobile_and_desktop_variants_collapse_in_one_search() -> None:
    autosearch = AutoSearch(backend=_MobileVariantBackend(), results_per_query=10)
    results = await autosearch.search("entity")
    assert len(results) == 1
    assert results[0].url == "https://m.antaranews.com/berita/1?amp=1"


@pytest.mark.asyncio
async def test_shared_seen_urls_cache_across_paths() -> None:
    """An anchored (70%) URL is never fetched again on the exploratory (30%) path."""
    shared: set[str] = set()
    anchored = AutoSearch(
        backend=MockSearchBackend(), results_per_query=10, seen_urls=shared
    )
    exploratory = AutoSearch(
        backend=MockSearchBackend(), results_per_query=10, seen_urls=shared
    )
    first = await anchored.search("entity")
    assert first
    second = await exploratory.search("entity")
    assert second == []
    assert shared  # cache is populated and shared by reference


@pytest.mark.asyncio
async def test_sanitized_bracket_query_is_searched_as_or_aliases() -> None:
    captured: list[str] = []

    class _CapturingBackend:
        name = "capture"
        supports_boolean = True

        async def search(self, query: str, limit: int = 10) -> list[SearchResult]:
            captured.append(query)
            return []

    autosearch = AutoSearch(backend=_CapturingBackend(), results_per_query=5)
    await autosearch.search("Bayu Skak (Bayu Eko Moektito)")
    assert captured == ['"Bayu Skak" OR "Bayu Eko Moektito"']
