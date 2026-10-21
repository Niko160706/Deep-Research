"""Unit tests for the sliding-window chunker."""
from __future__ import annotations

import pytest

from drx.muscle.chunker import Chunker


def test_short_text_is_a_single_chunk() -> None:
    c = Chunker(chunk_size=1200, chunk_overlap=200)
    assert c.chunk_text("hi") == ["hi"]


def test_empty_text_returns_empty() -> None:
    c = Chunker(chunk_size=10, chunk_overlap=2)
    assert c.chunk_text("   ") == []


def test_long_text_splits_with_overlap() -> None:
    c = Chunker(chunk_size=10, chunk_overlap=2)
    chunks = c.chunk_text("0123456789abcdef")
    assert chunks == ["0123456789", "89abcdef"]
    assert chunks[0][-2:] == chunks[1][:2] == "89"


def test_invalid_overlap_rejected() -> None:
    with pytest.raises(ValueError):
        Chunker(chunk_size=10, chunk_overlap=10)
