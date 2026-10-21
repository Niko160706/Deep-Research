"""Tests for the Phase 3 triple-JSON critic (``drx.brain.critic``).

The critic must never read raw parquet text: it receives only triple JSON and
returns pivot suggestions whose node IDs are grounded in that JSON.
"""
from __future__ import annotations

import json

import pytest

from drx.brain import batching, critic
from drx.config import LLMConfig


def _triples(count: int = 3, *, quote_size: int = 20) -> list[dict]:
    rows: list[dict] = []
    for index in range(count):
        rows.append(
            {
                "subject": "Sri Lestari",
                "predicate": "bekerja_di",
                "object": "Dinas Kesehatan Sleman",
                "confidence": 0.8,
                "subject_id": "n1",
                "object_id": "n2",
                "sources": [f"https://news.contoh.id/{index}"],
                "quotes": ["bekerja di Dinas Kesehatan Sleman" * (quote_size // 20 + 1)],
                "source_count": 2,
                "verified": True,
                "contradiction": False,
            }
        )
    rows.append(
        {
            "subject": "PT Jaya Abadi",
            "predicate": "bekerja_sama",
            "object": "Dinas Kesehatan Sleman",
            "confidence": 0.6,
            "subject_id": "n3",
            "object_id": "n2",
            "sources": ["https://news.contoh.id/vendor"],
            "quotes": ["PT Jaya Abadi bermitra"],
            "source_count": 1,
            "verified": False,
            "contradiction": False,
        }
    )
    return rows


class _FakeClient:
    name = "fake"
    model = "fake-critic"

    def __init__(self, reply: dict | str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, str, bool]] = []

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]:
        self.calls.append((system, user, json_mode))
        text = self.reply if isinstance(self.reply, str) else json.dumps(self.reply)
        return text, {
            "model": self.model,
            "input_tokens": 200,
            "output_tokens": 80,
            "total_tokens": 280,
        }


GAP_VERDICT = {
    "is_sufficient": False,
    "critique_summary": "Jabatan dan kontak instansi belum lengkap.",
    "contradictions": ["Tanggal mulai berbeda di dua sumber"],
    "suggested_sub_queries": [
        "Sri Lestari Dinas Kesehatan jabatan",
        "Sri Lestari Sleman profil",
    ],
    "suggested_nodes": [
        {"node_id": "n2", "label": "label ngawur", "reason": "Perdalam instansi."},
        {"node_id": "n99", "label": "tidak ada", "reason": "halusinasi"},
    ],
}


@pytest.mark.asyncio
async def test_analyze_graph_gaps_grounds_suggested_node_ids() -> None:
    client = _FakeClient(GAP_VERDICT)

    result = await critic.analyze_graph_gaps(
        _triples(), "Sri Lestari", "Sleman", client=client
    )

    assert result.ok
    assert result.is_sufficient is False
    assert result.contradictions == ["Tanggal mulai berbeda di dua sumber"]
    assert len(result.suggested_sub_queries) == 2
    # n99 dibuang; label diganti label kanonik dari triple JSON
    assert len(result.suggested_nodes) == 1
    assert result.suggested_nodes[0].node_id == "n2"
    assert result.suggested_nodes[0].label == "Dinas Kesehatan Sleman"
    assert result.suggested_nodes[0].reason == "Perdalam instansi."
    assert result.nodes_used == 3


@pytest.mark.asyncio
async def test_triple_critic_prompt_contains_only_triple_json() -> None:
    client = _FakeClient(GAP_VERDICT)

    await critic.analyze_graph_gaps(_triples(), "Sri Lestari", client=client)

    system, user, json_mode = client.calls[0]
    assert "Saran ID Node untuk Pivot" in system
    assert json_mode is True
    assert '"subject_id": "n1"' in user
    # teks mentah (URL / kutipan panjang) tidak boleh dikirim ke critic
    assert "https://" not in user
    assert "short_quote" not in user


@pytest.mark.asyncio
async def test_triple_critic_batches_and_reduces_verdicts() -> None:
    triples = _triples(count=40, quote_size=200)
    client = _FakeClient(GAP_VERDICT)
    cfg = LLMConfig(batch_max_input_tokens=3200)

    result = await critic.analyze_graph_gaps(
        triples, "Sri Lestari", config=cfg, client=client
    )

    assert result.batches >= 2
    assert len(client.calls) == result.batches
    # reduce: semua batch tidak cukup -> hasil akhir tidak cukup
    assert result.is_sufficient is False
    assert len(result.suggested_nodes) == 1  # dedupe n2
    for system, user, _ in client.calls:
        total = batching.count_tokens(system) + batching.count_tokens(user)
        assert total <= cfg.batch_max_input_tokens


@pytest.mark.asyncio
async def test_triple_critic_sufficient_union_requires_all_batches() -> None:
    triples = _triples(count=40, quote_size=200)
    client = _FakeClient(
        {
            "is_sufficient": True,
            "critique_summary": "Cukup.",
            "contradictions": [],
            "suggested_sub_queries": [],
            "suggested_nodes": [],
        }
    )
    result = await critic.analyze_graph_gaps(triples, "Sri Lestari", client=client)
    assert result.batches >= 2
    assert result.is_sufficient is True


@pytest.mark.asyncio
async def test_triple_critic_empty_graph_is_skipped() -> None:
    result = await critic.analyze_graph_gaps([], "Sri Lestari")
    assert result.skipped_reason == "tidak ada triple di graph untuk dinilai"


@pytest.mark.asyncio
async def test_triple_critic_disabled_and_missing_key() -> None:
    cfg = LLMConfig().model_copy(update={"enabled": False})
    disabled = await critic.analyze_graph_gaps(_triples(), "Sri Lestari", config=cfg)
    assert "nonaktif" in disabled.skipped_reason

    no_key = await critic.analyze_graph_gaps(_triples(), "Sri Lestari", env={})
    assert "API key" in no_key.skipped_reason


@pytest.mark.asyncio
async def test_triple_critic_invalid_json_is_graceful() -> None:
    result = await critic.analyze_graph_gaps(
        _triples(), "Sri Lestari", client=_FakeClient("bukan json")
    )
    assert result.error
    assert result.ok is False


@pytest.mark.asyncio
async def test_triple_critic_caps_output_tokens_350_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq OTPM free tier 1000: output critic wajib 350-400 token."""
    from types import SimpleNamespace

    captured: list[int] = []

    async def fake_call(specs, cfg, system, user, *, json_mode=False, client=None):
        captured.append(cfg.max_output_tokens)
        return (
            json.dumps(GAP_VERDICT),
            {"model": "qwen"},
            SimpleNamespace(name="groq", model="qwen/qwen3.8-27b"),
        )

    monkeypatch.setattr(critic, "generate_with_failover", fake_call)

    result = await critic.analyze_graph_gaps(
        _triples(),
        "Sri Lestari",
        "Sleman",
        config=LLMConfig(provider="groq"),
        env={"GROQ_API_KEY": "qk"},
    )
    assert result.ok
    assert captured == [400]  # default critic_max_output_tokens

    # Nilai settings di luar rentang di-clamp ke 350-400.
    captured.clear()
    await critic.analyze_graph_gaps(
        _triples(),
        "Sri Lestari",
        config=LLMConfig(provider="groq", critic_max_output_tokens=9999),
        env={"GROQ_API_KEY": "qk"},
    )
    await critic.analyze_graph_gaps(
        _triples(),
        "Sri Lestari",
        config=LLMConfig(provider="groq", critic_max_output_tokens=10),
        env={"GROQ_API_KEY": "qk"},
    )
    assert captured == [400, 350]
