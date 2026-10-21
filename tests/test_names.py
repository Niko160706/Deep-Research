"""Unit tests for person-name detection, exact-match search, and name filtering."""
from __future__ import annotations

import pytest

from drx.models import Document
from drx.muscle.autosearch import AutoSearch
from drx.muscle.cleaner import Cleaner
from drx.textutil import has_all_tokens, looks_like_name, name_tokens


# ---- name detection ----
def test_looks_like_name() -> None:
    assert looks_like_name("Nur Royul Amin") is True
    assert looks_like_name("Alan Turing") is True
    assert looks_like_name("Niko Rajani Syahputra Pane") is True
    assert looks_like_name("why did the Roman Empire fall") is False
    assert looks_like_name("Alan") is False


def test_name_tokens() -> None:
    assert name_tokens("Nur Royul Amin") == ["nur", "royul", "amin"]


def test_has_all_tokens_requires_whole_word() -> None:
    assert has_all_tokens("Nur Royul Amin is here", ["royul"]) is True
    # "royul" absent
    assert has_all_tokens("Nurul Amin is here", ["royul"]) is False
    # "nur" is not a whole word inside "nurul"
    assert has_all_tokens("Nurul Amin", ["nur"]) is False


# ---- exact-match search ----
class _CapturingBackend:
    name = "capture"

    def __init__(self) -> None:
        self.last_query: str | None = None

    async def search(self, query: str, limit: int = 10) -> list:
        self.last_query = query
        return []


@pytest.mark.asyncio
async def test_name_query_is_quoted_for_exact_match() -> None:
    backend = _CapturingBackend()
    autosearch = AutoSearch(backend=backend)
    await autosearch.search("Nur Royul Amin")
    assert backend.last_query == '"Nur Royul Amin"'


@pytest.mark.asyncio
async def test_non_name_query_is_not_quoted() -> None:
    backend = _CapturingBackend()
    autosearch = AutoSearch(backend=backend)
    await autosearch.search("why did the Roman Empire fall")
    assert backend.last_query == "why did the Roman Empire fall"


# ---- cleaner name filtering ----
def test_cleaner_discards_lookalike_name_documents() -> None:
    cleaner = Cleaner(min_text_length=1, bm25_keep_top_k=5)
    good = Document(url="http://a", text="Nur Royul Amin adalah seorang peneliti muda.")
    bad = Document(url="http://b", text="Nurul Amin dari Pakistan adalah aktivis.")
    docs = cleaner.clean([good, bad], query="Nur Royul Amin")
    assert len(docs) == 1
    assert docs[0].url == "http://a"
