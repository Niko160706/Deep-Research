"""Runtime budget counters: tokens, URLs scraped, and wall-clock.

Pure Python, no I/O. The orchestrator checks these before doing any work so
that a runaway loop terminates on the cheapest possible condition first.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("drx.memory.budget")


class Budget:
    """Tracks consumption against hard limits.

    Token accounting is driven by :class:`drx.llm.client.DeepSeekClient` as a
    side-effect of every completion — without it the token stop-condition
    would be decorative.
    """

    def __init__(
        self,
        max_tokens: int = 150000,
        max_runtime_seconds: int = 600,
        max_urls: int = 200,
    ) -> None:
        self.max_tokens = max_tokens
        self.max_runtime_seconds = max_runtime_seconds
        self.max_urls = max_urls
        self._tokens = 0
        self._urls = 0
        self._start = time.monotonic()

    # ---- accounting ----
    def add_tokens(self, n: int) -> None:
        self._tokens += n

    def add_url(self, n: int = 1) -> None:
        self._urls += n

    # ---- read-only views ----
    @property
    def tokens(self) -> int:
        return self._tokens

    @property
    def urls(self) -> int:
        return self._urls

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._start

    # ---- stop predicates ----
    def tokens_exhausted(self) -> bool:
        return self._tokens >= self.max_tokens

    def clock_exhausted(self) -> bool:
        return self.elapsed >= self.max_runtime_seconds

    def urls_exhausted(self) -> bool:
        return self._urls >= self.max_urls

    def snapshot(self) -> dict[str, int | float]:
        return {
            "tokens": self._tokens,
            "urls": self._urls,
            "elapsed_seconds": round(self.elapsed, 2),
        }

    def status(self) -> dict[str, int | float]:
        """Cumulative usage + hard limits + remaining budget for the UI."""
        return {
            "tokens": self._tokens,
            "max_tokens": self.max_tokens,
            "tokens_remaining": max(0, self.max_tokens - self._tokens),
            "urls": self._urls,
            "max_urls": self.max_urls,
            "urls_remaining": max(0, self.max_urls - self._urls),
            "elapsed_seconds": round(self.elapsed, 2),
            "max_runtime_seconds": self.max_runtime_seconds,
            "seconds_remaining": round(
                max(0.0, self.max_runtime_seconds - self.elapsed), 2
            ),
        }
