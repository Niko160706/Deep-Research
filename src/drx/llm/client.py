"""DeepSeek LLM client — the SINGLE gate for token metering.

Every cognitive call must flow through :meth:`complete`, which adds the
response's ``usage.total_tokens`` to the shared :class:`Budget` as a
side-effect. Without this, the token stop-condition would be decorative.
"""
from __future__ import annotations

import logging
from typing import Any

from drx.config import RoleConfig
from drx.memory.budget import Budget

log = logging.getLogger("drx.llm.client")


class DeepSeekClient:
    """Async wrapper over the OpenAI-compatible DeepSeek endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.deepseek.com",
        budget: Budget | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.budget = budget
        self._client = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI

            self._client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    async def complete(
        self,
        role: RoleConfig,
        messages: list[dict[str, str]],
        schema: type | None = None,
    ) -> Any:
        """Send a chat completion and meter tokens into the budget.

        Phase 2 wires the actual request; Phase 1 intentionally raises because
        the cognitive stages use mocks and never call this method.
        """
        raise NotImplementedError(
            "DeepSeekClient.complete lands in Phase 2; use mock cognitive stages "
            "for the Phase 1 muscle pipeline."
        )
