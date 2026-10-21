"""Tests for the Phase 3 token guard (``drx.brain.batching``).

The guard is deliberately offline: tiktoken is used when available, otherwise
``len/4``; every helper must be deterministic and never raise.
"""
from __future__ import annotations

import pytest

from drx.brain import batching


def test_count_tokens_monotonic_and_positive() -> None:
    assert batching.count_tokens("") == 0
    short = batching.count_tokens("Halo dunia")
    long_text = batching.count_tokens("Halo dunia " * 50)
    assert 0 < short < long_text


def test_count_tokens_fallback_without_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batching, "get_encoder", lambda: None)
    text = "empat kata di sini"
    assert batching.count_tokens(text) == (len(text) + 3) // 4
    assert batching.count_tokens("") == 0


def test_truncate_tokens_respects_budget() -> None:
    text = " ".join(f"kata{i}" for i in range(400))
    out = batching.truncate_tokens(text, 50)
    assert out
    assert batching.count_tokens(out) <= 50
    # budget kebesaran -> teks utuh
    assert batching.truncate_tokens(text, 100_000) == text


def test_mmr_prefers_diverse_second_pick() -> None:
    relevance = [0.9, 0.89, 0.1]
    texts = [
        "alpha beta gamma delta",
        "alpha beta gamma delta",  # duplikat dari kandidat pertama
        "zzz xxx yyy",  # berbeda total
    ]
    chosen = batching.mmr_indices(relevance, texts, lambda_=0.5)
    assert chosen[0] == 0
    assert chosen[1] == 2  # diversitas menang atas skor 0.89 yang redundan


def test_pack_batches_respects_token_budget() -> None:
    a = "satu dua tiga " * 20
    b = "empat lima enam " * 20
    max_tokens = batching.count_tokens(a) + 2
    batches = batching.pack_batches([a, b], max_tokens)
    assert len(batches) == 2
    for batch in batches:
        assert batching.count_tokens(" ".join(batch)) <= max_tokens


def test_pack_batches_truncates_oversize_single_text() -> None:
    big = "kata " * 2000
    batches = batching.pack_batches([big], 100)
    assert len(batches) == 1
    assert batching.count_tokens(batches[0][0]) <= 100


def test_pack_batches_skips_blank_texts() -> None:
    batches = batching.pack_batches(["", "   ", "isi nyata"], 100)
    assert batches == [["isi nyata"]]


def test_estimate_cost_usd_math() -> None:
    cost = batching.estimate_cost_usd(
        1_000_000,
        500_000,
        price_input_per_1m=0.1,
        price_output_per_1m=0.4,
    )
    assert cost == pytest.approx(0.3)
    assert batching.estimate_cost_usd(10, 10) == 0.0
