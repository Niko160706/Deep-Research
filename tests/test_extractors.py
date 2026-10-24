"""Tests for the Phase 3 grounded triple extractor (``drx.brain.extractors``).

Zero network: an injected fake client returns scripted JSON, while Python does
all grounding (subject/object/quote must exist in the referenced chunk).
"""
from __future__ import annotations

import json

import pytest

from drx.brain import batching, extractors
from drx.config import LLMConfig
from drx.models import Document

URL = "https://news.contoh.id/sri-lestari"
DOC = Document(
    url=URL,
    title="Sri Lestari perawat Sleman",
    text=(
        "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan Sleman. "
        "Ia menjabat sebagai koordinator imunisasi sejak tahun 2019. "
        "Dinas Kesehatan Sleman beralamat di Jalan Parasamya."
    ),
)


class _FakeClient:
    name = "fake"
    model = "fake-extractor"

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, bool]] = []

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]:
        self.calls.append((system, user, json_mode))
        payload = self.replies.pop(0) if self.replies else {"triples": []}
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return text, {
            "model": self.model,
            "input_tokens": 120,
            "output_tokens": 30,
            "total_tokens": 150,
        }


def _valid_payload() -> dict:
    return {
        "triples": [
            {
                "subject": "Sri Lestari",
                "subject_short_label": "Sri Lestari",
                "predicate": "bekerja_sebagai",
                "object": "perawat",
                "object_short_label": "Perawat",
                "source_id": "S1",
                "short_quote": "bekerja sebagai perawat",
                "confidence": 0.8,
            }
        ]
    }


@pytest.mark.asyncio
async def test_extract_triples_keeps_grounded_triple() -> None:
    client = _FakeClient([_valid_payload()])

    result = await extractors.extract_triples([DOC], "Sri Lestari", client=client)

    assert result.error == "" and result.skipped_reason == ""
    assert len(result.triples) == 1
    triple = result.triples[0]
    assert triple.source_id == "S1"
    assert triple.source_url == URL
    assert triple.short_quote == "bekerja sebagai perawat"
    assert triple.subject_short_label == "Sri Lestari"
    assert triple.object_short_label == "Perawat"
    assert "subject_short_label" in client.calls[0][0]
    assert result.rejected == 0
    assert result.documents == 1
    assert result.batches == 1


@pytest.mark.asyncio
async def test_extract_triples_falls_back_when_short_labels_missing() -> None:
    payload = _valid_payload()
    payload["triples"][0].pop("subject_short_label")
    payload["triples"][0].pop("object_short_label")
    result = await extractors.extract_triples(
        [DOC], "Sri Lestari", client=_FakeClient([payload])
    )

    assert len(result.triples) == 1
    assert result.triples[0].subject_short_label == "Sri Lestari"
    assert result.triples[0].object_short_label == "perawat"


@pytest.mark.asyncio
async def test_extract_triples_drops_ungrounded_claims() -> None:
    payload = {
        "triples": [
            _valid_payload()["triples"][0],
            {  # subjek tidak ada di chunk
                "subject": "Budi Santoso",
                "predicate": "bekerja_sebagai",
                "object": "perawat",
                "source_id": "S1",
                "short_quote": "bekerja sebagai perawat",
            },
            {  # kutipan tidak verbatim
                "subject": "Sri Lestari",
                "predicate": "bekerja_sebagai",
                "object": "perawat",
                "source_id": "S1",
                "short_quote": "kalimat yang tidak ada di sumber",
            },
            {  # source_id tidak dikenal
                "subject": "Sri Lestari",
                "predicate": "bekerja_sebagai",
                "object": "perawat",
                "source_id": "S99",
                "short_quote": "bekerja sebagai perawat",
            },
        ]
    }
    result = await extractors.extract_triples(
        [DOC], "Sri Lestari", client=_FakeClient([payload])
    )

    assert len(result.triples) == 1
    assert result.rejected == 3


def test_validate_triple_requires_source_match_and_grounding() -> None:
    chunk = extractors.build_evidence_chunks([DOC])[0]
    triple = extractors.GroundedTriple(
        subject="Sri Lestari",
        predicate="bekerja_sebagai",
        object="perawat",
        source_id=chunk.source_id,
        short_quote="bekerja sebagai perawat",
    )
    ok, reason = extractors.validate_triple(triple, chunk)
    assert ok is True and reason == ""

    triple.source_id = "S9"
    ok, reason = extractors.validate_triple(triple, chunk)
    assert ok is False and reason == "source_mismatch"


@pytest.mark.asyncio
async def test_extract_triples_dedupes_same_fact_same_source() -> None:
    payload = _valid_payload()
    payload["triples"].append(_valid_payload()["triples"][0])
    result = await extractors.extract_triples(
        [DOC], "Sri Lestari", client=_FakeClient([payload])
    )
    assert len(result.triples) == 1


@pytest.mark.asyncio
async def test_extract_triples_batches_long_documents_under_guard() -> None:
    long_doc = Document(
        url=URL,
        title="Berkas panjang",
        text=(
            "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan Sleman. "
            * 260
        ),
    )
    cfg = LLMConfig(batch_max_input_tokens=3200)
    client = _FakeClient([{"triples": []}] * 12)

    result = await extractors.extract_triples(
        [long_doc], "Sri Lestari", config=cfg, client=client
    )

    assert result.batches >= 2
    assert len(client.calls) == result.batches
    for system, user, _json_mode in client.calls:
        total = batching.count_tokens(system) + batching.count_tokens(user)
        assert total <= cfg.batch_max_input_tokens


@pytest.mark.asyncio
async def test_extract_triples_skips_without_api_key() -> None:
    result = await extractors.extract_triples([DOC], "Sri Lestari", env={})
    assert result.triples == []
    assert "API key" in result.skipped_reason


@pytest.mark.asyncio
async def test_extract_triples_disabled_returns_skip() -> None:
    cfg = LLMConfig().model_copy(update={"enabled": False})
    result = await extractors.extract_triples([DOC], "Sri Lestari", config=cfg)
    assert result.skipped_reason


def test_triples_to_facts_carries_provenance() -> None:
    triple = extractors.GroundedTriple(
        subject="Sri Lestari",
        predicate="bekerja_sebagai",
        object="perawat",
        source_id="S1",
        short_quote="bekerja sebagai perawat",
        source_url=URL,
        confidence=0.8,
        subject_short_label="Sri Lestari",
        object_short_label="Perawat",
    )
    facts = extractors.triples_to_facts([triple])

    assert len(facts) == 1
    fact = facts[0]
    assert fact.subject == "Sri Lestari"
    assert fact.subject_short_label == "Sri Lestari"
    assert fact.object_short_label == "Perawat"
    assert fact.confidence_score == pytest.approx(0.8)
    assert fact.provenance[0].url == URL
    assert fact.provenance[0].source_id == "S1"
    assert fact.provenance[0].quote == "bekerja sebagai perawat"


@pytest.mark.asyncio
async def test_extract_triples_caps_output_tokens_350_400(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq OTPM free tier 1000: output ekstraksi wajib 350-400 token."""
    from types import SimpleNamespace

    captured: list[int] = []

    async def fake_call(specs, cfg, system, user, *, json_mode=False, client=None):
        captured.append(cfg.max_output_tokens)
        return (
            json.dumps(_valid_payload()),
            {"model": "qwen"},
            SimpleNamespace(name="groq", model="qwen/qwen3.8-27b"),
        )

    monkeypatch.setattr(extractors, "generate_with_failover", fake_call)

    result = await extractors.extract_triples(
        [DOC],
        "Sri Lestari",
        config=LLMConfig(provider="groq"),
        env={"GROQ_API_KEY": "qk"},
    )
    assert len(result.triples) == 1
    assert captured == [400]  # default extraction_max_output_tokens

    captured.clear()
    await extractors.extract_triples(
        [DOC],
        "Sri Lestari",
        config=LLMConfig(provider="groq", extraction_max_output_tokens=9999),
        env={"GROQ_API_KEY": "qk"},
    )
    await extractors.extract_triples(
        [DOC],
        "Sri Lestari",
        config=LLMConfig(provider="groq", extraction_max_output_tokens=10),
        env={"GROQ_API_KEY": "qk"},
    )
    assert captured == [400, 350]
