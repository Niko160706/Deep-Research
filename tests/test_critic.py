"""Tests for the Phase 2 agentic critic (``drx.brain.critic``).

Offline by design: the parquet is written to ``tmp_path``, provider resolution
gets an injected env mapping, and every LLM call goes through a scriptable fake
client. No network, no API key.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from drx.brain import critic
from drx.brain import llm_synthesizer as syn
from drx.config import LLMConfig
from drx.models import Document

VERDICT_GAP = {
    "is_sufficient": False,
    "critique_summary": (
        "Identitas belum terverifikasi silang; jabatan terbaru belum ada."
    ),
    "suggested_sub_queries": [
        "Sri Lestari Sleman perawat",
        "Sri Lestari Dinas Kesehatan jabatan",
    ],
}
VERDICT_OK = {
    "is_sufficient": True,
    "critique_summary": "Identitas jelas dan tidak ada kontradiksi.",
    "suggested_sub_queries": [],
}


class _FakeCriticClient:
    """Fake provider that records calls and honours ``json_mode``."""

    name = "fake"
    model = "fake-critic"

    def __init__(
        self,
        replies: list[str],
        *,
        boom_first: bool = False,
        too_large_first: bool = False,
    ) -> None:
        self.replies = list(replies)
        self.boom_first = boom_first
        self.too_large_first = too_large_first
        self.calls: list[tuple[str, str]] = []
        self.json_modes: list[bool] = []

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]:
        self.calls.append((system, user))
        self.json_modes.append(json_mode)
        if self.boom_first and len(self.calls) == 1:
            raise RuntimeError("response_format not supported")
        if self.too_large_first and len(self.calls) == 1:
            raise RuntimeError(
                "Error code: 413 - Request too large for model "
                "`qwen/fake` ... ITPM: Limit 7000, Requested 8493 ... "
                "code: rate_limit_exceeded"
            )
        text = self.replies.pop(0) if self.replies else ""
        return text, {
            "model": self.model,
            "output_tokens": 40,
            "total_tokens": 120,
        }


def _docs() -> list[Document]:
    return [
        Document(
            url="https://news.contoh.id/sri-lestari",
            title="Sri Lestari perawat Sleman",
            text=(
                "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan "
                "Kab. Sleman dengan dedikasi tinggi. "
            )
            * 6,
        ),
        Document(
            url="https://blog.contoh.id/resep",
            title="Resep masakan rumahan",
            text="Artikel ini membahas resep masakan rumahan yang mudah. " * 6,
        ),
    ]


def _write_parquet(tmp_path: Path, docs: list[Document]) -> Path:
    target = tmp_path / "parquet" / "raw_documents.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([d.model_dump() for d in docs]).to_parquet(target, index=False)
    return target


# ---- prompt building ---------------------------------------------------------

def test_system_prompt_is_an_osint_fact_verifier() -> None:
    prompt = critic.build_critic_system_prompt("id")
    assert "OSINT Fact Verifier" in prompt
    assert "is_sufficient" in prompt
    assert "critique_summary" in prompt
    assert "suggested_sub_queries" in prompt
    assert "IDENTITAS" in prompt
    assert "KONTRADIKSI" in prompt
    assert "GAP DATA" in prompt


def test_system_prompt_english_language_variant() -> None:
    assert "in English" in critic.build_critic_system_prompt("en")
    assert "in English" not in critic.build_critic_system_prompt("id")


def test_critic_prompt_carries_target_context_and_evidence() -> None:
    segments = syn.select_segments(_docs(), "Sri Lestari", "Sleman, Kesehatan")
    prompt = critic.build_critic_prompt("Sri Lestari", "Sleman, Kesehatan", segments)

    assert "TARGET INVESTIGASI: Sri Lestari" in prompt
    assert "KONTEKS TAMBAHAN: Sleman, Kesehatan" in prompt
    assert "[S1] URL: https://news.contoh.id/sri-lestari" in prompt
    assert "AKHIR BUKTI" in prompt


# ---- verdict parsing (pydantic JSON contract) --------------------------------

def test_critic_result_json_roundtrip() -> None:
    result = critic.CriticResult(
        is_sufficient=False,
        critique_summary="gap",
        suggested_sub_queries=["q1"],
    )
    dumped = result.model_dump()
    assert {"is_sufficient", "critique_summary", "suggested_sub_queries"} <= set(dumped)
    assert dumped["is_sufficient"] is False
    assert dumped["suggested_sub_queries"] == ["q1"]


def test_parse_verdict_reads_plain_and_fenced_json() -> None:
    plain = critic._parse_verdict(json.dumps(VERDICT_GAP), limit=5)
    assert plain is not None
    assert plain.is_sufficient is False
    assert plain.critique_summary.startswith("Identitas belum")
    assert len(plain.suggested_sub_queries) == 2

    fenced = critic._parse_verdict(
        "```json\n" + json.dumps(VERDICT_OK) + "\n```", limit=5
    )
    assert fenced is not None and fenced.is_sufficient is True


def test_parse_verdict_rejects_non_json_and_missing_key() -> None:
    assert critic._parse_verdict("tidak bisa diparse", limit=3) is None
    assert critic._parse_verdict('{"foo": 1}', limit=3) is None
    assert critic._parse_verdict("", limit=3) is None


def test_sub_queries_are_cleaned_deduped_and_capped() -> None:
    data = {
        "is_sufficient": False,
        "critique_summary": "gap",
        "suggested_sub_queries": ["  a   b ", "a b", "", "c", "d", "e"],
    }
    verdict = critic._parse_verdict(json.dumps(data), limit=2)
    assert verdict is not None
    assert verdict.suggested_sub_queries == ["a b", "c"]


# ---- end-to-end analysis (fake client) ---------------------------------------

@pytest.mark.asyncio
async def test_analyze_fact_gaps_with_fake_client(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeCriticClient([json.dumps(VERDICT_GAP)])

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "Sleman", client=client, config=LLMConfig()
    )

    assert result.ok and not result.error
    assert result.is_sufficient is False
    assert len(result.suggested_sub_queries) == 2
    assert result.total_tokens == 120
    assert result.retries == 0
    assert client.json_modes == [True]  # percobaan pertama pakai native JSON mode
    system, user = client.calls[0]
    assert "Sri Lestari" in user
    assert "OSINT Fact Verifier" in system


@pytest.mark.asyncio
async def test_analyze_fact_gaps_retries_when_reply_unparseable(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeCriticClient(["bukan json", json.dumps(VERDICT_OK)])

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "", client=client
    )

    assert result.ok and result.is_sufficient
    assert result.retries == 1
    assert client.json_modes == [True, False]  # retry tanpa json mode
    assert result.total_tokens == 240  # dua percobaan diakumulasi
    assert "PENTING" in client.calls[1][1]  # instruksi penegas dikirim


@pytest.mark.asyncio
async def test_analyze_fact_gaps_retries_when_json_mode_rejected(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeCriticClient([json.dumps(VERDICT_OK)], boom_first=True)

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "", client=client
    )

    assert result.ok and result.is_sufficient
    assert client.json_modes == [True, False]


@pytest.mark.asyncio
async def test_analyze_fact_gaps_both_attempts_unparseable_returns_error(
    tmp_path: Path,
) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    client = _FakeCriticClient(["nope", "still nope"])

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "", client=client
    )

    assert not result.ok
    assert "JSON" in result.error
    assert len(client.calls) == 2


@pytest.mark.asyncio
async def test_analyze_fact_gaps_provider_error_is_graceful(tmp_path: Path) -> None:
    class _BoomClient:
        name = "boom"
        model = "boom-1"

        async def generate(self, system, user, *, json_mode=False):
            raise RuntimeError("network down")

    parquet = _write_parquet(tmp_path, _docs())
    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "", client=_BoomClient()
    )

    assert not result.ok
    assert "RuntimeError" in result.error
    assert "network down" in result.error


@pytest.mark.asyncio
async def test_analyze_fact_gaps_skips_without_api_key(tmp_path: Path) -> None:
    parquet = _write_parquet(tmp_path, _docs())
    result = await critic.analyze_fact_gaps(parquet, "Sri Lestari", "Sleman", env={})
    assert not result.ok
    assert "API key" in result.skipped_reason


@pytest.mark.asyncio
async def test_analyze_fact_gaps_fails_over_between_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Gemini 503 total -> critic tetap dapat verdict via Groq (qwen)."""

    class _FailingProvider:
        def __init__(self, spec):
            self.name = spec.name
            self.model = spec.model

        async def generate(self, system, user, *, json_mode=False):
            raise RuntimeError("503 UNAVAILABLE: high demand")

    parquet = _write_parquet(tmp_path, _docs())
    working = _FakeCriticClient([json.dumps(VERDICT_OK)])

    def fake_build(spec, cfg):
        if spec.name == "gemini":
            return _FailingProvider(spec)
        return working

    monkeypatch.setattr(syn, "build_client", fake_build)

    result = await critic.analyze_fact_gaps(
        parquet,
        "Sri Lestari",
        "Sleman",
        env={"GEMINI_API_KEY": "gk", "GROQ_API_KEY": "qk"},
    )

    assert result.ok and result.is_sufficient
    assert result.provider == "groq"
    assert working.calls


@pytest.mark.asyncio
async def test_analyze_fact_gaps_skips_without_documents(tmp_path: Path) -> None:
    result = await critic.analyze_fact_gaps(
        tmp_path / "missing.parquet", "Sri Lestari", ""
    )
    assert not result.ok
    assert "tidak ada dokumen" in result.skipped_reason


@pytest.mark.asyncio
async def test_analyze_fact_gaps_respects_critic_segment_caps(tmp_path: Path) -> None:
    docs = [
        Document(
            url=f"https://x.id/{i}",
            title="profil",
            text=f"Sri Lestari perawat Sleman nomor {i}. " * 8,
        )
        for i in range(6)
    ]
    parquet = _write_parquet(tmp_path, docs)
    client = _FakeCriticClient([json.dumps(VERDICT_OK)])
    cfg = LLMConfig(critic_max_segments=2, critic_max_input_tokens=10**9)

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "Sleman", client=client, config=cfg
    )

    assert result.segments_used == 2  # config critic benar-benar dipakai
    assert result.documents_used == 2


def test_critic_config_defaults() -> None:
    cfg = LLMConfig()
    assert cfg.critic_enabled is True
    assert cfg.critic_max_loops == 2
    assert cfg.critic_max_sub_queries == 3
    assert cfg.critic_max_segments == 8
    assert cfg.critic_max_input_tokens == 3400  # Phase 3: hard guard 3.0-3.5k


@pytest.mark.asyncio
async def test_analyze_fact_gaps_hard_caps_system_plus_user_tokens(
    tmp_path: Path,
) -> None:
    """Legacy path juga wajib ≤ critic_max_input_tokens (system + user)."""
    from drx.brain import batching

    docs = [
        Document(
            url=f"https://x.id/hard-{i}",
            title=f"berkas Sri Lestari {i}",
            text=(
                "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan Sleman. "
                + " ".join(
                    f"Catatan {i}-{j}: program imunisasi wilayah Sleman."
                    for j in range(60)
                )
            ),
        )
        for i in range(4)
    ]
    parquet = _write_parquet(tmp_path, docs)
    client = _FakeCriticClient([json.dumps(VERDICT_OK)])
    cfg = LLMConfig(critic_max_input_tokens=3400)

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "Sleman", client=client, config=cfg
    )

    assert result.ok
    system, user = client.calls[0]
    assert batching.count_tokens(system) + batching.count_tokens(user) <= 3400


@pytest.mark.asyncio
async def test_analyze_fact_gaps_shrinks_prompt_on_provider_size_limit(
    tmp_path: Path,
) -> None:
    docs = [
        Document(
            url=f"https://x.id/long-{i}",
            title=f"profil Sri Lestari {i}",
            text=(
                "Sri Lestari bekerja sebagai perawat di Dinas Kesehatan Kab. Sleman. "
                + " ".join(
                    f"Catatan {i}-{j}: program imunisasi dan posyandu di "
                    f"wilayah Sleman utara berjalan baik pada tahun {2000 + j}."
                    for j in range(45)
                )
            ),
        )
        for i in range(6)
    ]
    parquet = _write_parquet(tmp_path, docs)
    client = _FakeCriticClient([json.dumps(VERDICT_OK)], too_large_first=True)

    result = await critic.analyze_fact_gaps(
        parquet, "Sri Lestari", "Sleman", client=client
    )

    assert result.ok and result.is_sufficient
    assert len(client.calls) == 2  # 413 -> pangkas konteks -> retry
    assert len(client.calls[1][1]) < len(client.calls[0][1])
    assert client.json_modes == [True, True]
    assert result.retries == 1
