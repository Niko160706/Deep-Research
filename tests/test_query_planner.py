"""Tests for the P0 seeded query planner (``drx.brain.query_planner``).

One small (~400-token) call at run start; deterministic seeds whenever the LLM
is missing, errors, or answers garbage.
"""
from __future__ import annotations

import json

import pytest

from drx.brain import batching, query_planner
from drx.config import LLMConfig


class _FakeClient:
    name = "fake"
    model = "fake-planner"

    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[str, str, bool]] = []

    async def generate(
        self, system: str, user: str, *, json_mode: bool = False
    ) -> tuple[str, dict]:
        self.calls.append((system, user, json_mode))
        payload = self.replies.pop(0) if self.replies else {"queries": []}
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return text, {
            "model": self.model,
            "input_tokens": 90,
            "output_tokens": 40,
            "total_tokens": 130,
        }


class _BoomClient:
    name = "boom"
    model = "boom-1"

    async def generate(self, system: str, user: str, *, json_mode: bool = False):
        raise RuntimeError("network down")


@pytest.mark.asyncio
async def test_plan_queries_filters_and_dedupes() -> None:
    client = _FakeClient(
        [
            {
                "queries": [
                    "Sri Lestari profil lengkap",
                    "Sri Lestari jabatan perawat",
                    "profil orang lain tanpa target",
                    "Sri Lestari profil lengkap",  # duplikat
                ],
                "rationale": "Identitas dan jabatan belum lengkap.",
            }
        ]
    )

    result = await query_planner.plan_queries(
        "Sri Lestari", "Sleman", client=client
    )

    assert result.ok
    assert result.used_fallback is False
    assert result.queries == [
        "Sri Lestari profil lengkap",
        "Sri Lestari jabatan perawat",
    ]
    assert result.rationale.startswith("Identitas")
    assert result.total_tokens == 130


@pytest.mark.asyncio
async def test_plan_queries_prompt_stays_under_token_cap() -> None:
    client = _FakeClient([{"queries": ["Sri Lestari profil"]}])

    result = await query_planner.plan_queries("Sri Lestari", "Sleman", client=client)

    assert result.input_tokens_est <= query_planner.PLANNER_MAX_INPUT_TOKENS
    system, user, _ = client.calls[0]
    total = batching.count_tokens(system) + batching.count_tokens(user)
    assert total <= query_planner.PLANNER_MAX_INPUT_TOKENS


@pytest.mark.asyncio
async def test_plan_queries_falls_back_without_api_key() -> None:
    result = await query_planner.plan_queries("Sri Lestari", "Sleman", env={})

    assert result.used_fallback is True
    assert "API key" in result.skipped_reason
    assert result.queries
    assert all("Sri Lestari" in q for q in result.queries)


@pytest.mark.asyncio
async def test_plan_queries_falls_back_on_provider_error() -> None:
    result = await query_planner.plan_queries(
        "Sri Lestari", "Sleman", client=_BoomClient()
    )

    assert result.used_fallback is True
    assert "network down" in result.error
    assert result.queries


@pytest.mark.asyncio
async def test_plan_queries_disabled_returns_no_tasks() -> None:
    cfg = LLMConfig().model_copy(update={"planner_enabled": False})
    result = await query_planner.plan_queries("Sri Lestari", config=cfg)
    assert result.queries == []
    assert "nonaktif" in result.skipped_reason


@pytest.mark.asyncio
async def test_plan_queries_falls_back_on_invalid_json() -> None:
    client = _FakeClient(["bukan json sama sekali"])
    result = await query_planner.plan_queries("Sri Lestari", client=client)
    assert result.used_fallback is True
    assert result.queries


def test_seeded_queries_are_deterministic_and_context_aware() -> None:
    first = query_planner.seeded_queries("Sri Lestari", "Sleman, DIY", limit=4)
    second = query_planner.seeded_queries("Sri Lestari", "Sleman, DIY", limit=4)
    assert first == second
    assert first[0] == "Sri Lestari"
    assert any("Sleman" in q for q in first)
    assert query_planner.seeded_queries("", "Sleman") == []
