"""Token-accurate batching guard + MMR diversity selection (0 LLM tokens).

Phase 3 production constraint: **no LLM request may carry more than
~3000-3500 input tokens**, because hosted free tiers reject large prompts
(Groq ``413 Request too large ... ITPM: Limit 7000``) and overburdened
endpoints (Gemini 503 high demand) fail more often on big payloads.

This module is the single place that answers:

* how many tokens a string costs — ``count_tokens`` (tiktoken, with a
  conservative character fallback when the BPE data is not available);
* how to cut a string without breaking it — ``truncate_tokens``;
* which evidence blocks to keep for diversity, not just relevance —
  ``mmr_indices`` (Maximal Marginal Relevance over token Jaccard);
* how to split N blocks into token-bounded batches — ``pack_batches``.

Everything here is deterministic pure Python; the only optional dependency
is ``tiktoken`` (fallback: ``len(text)/4`` like the legacy estimator).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

log = logging.getLogger("drx.brain.batching")

_encoder = None
_encoder_failed = False


def get_encoder():
    """Lazy ``tiktoken`` encoder; ``None`` when unavailable (offline)."""
    global _encoder, _encoder_failed
    if _encoder is not None or _encoder_failed:
        return _encoder
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:  # pragma: no cover - network/dependency guard
        log.warning(
            "tiktoken tidak tersedia (%s) -> fallback estimasi ~4 char/token", exc
        )
        _encoder_failed = True
        _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Exact-ish token count (tiktoken) with a safe character fallback."""
    raw = text or ""
    encoder = get_encoder()
    if encoder is None:
        return (len(raw) + 3) // 4
    try:
        return len(encoder.encode(raw, disallowed_special=()))
    except Exception:  # pragma: no cover - malformed input guard
        return (len(raw) + 3) // 4


def truncate_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to at most ``max_tokens`` tokens (character slice fallback)."""
    raw = text or ""
    if max_tokens <= 0:
        return ""
    encoder = get_encoder()
    if encoder is None:
        return raw[: max_tokens * 4]
    try:
        tokens = encoder.encode(raw, disallowed_special=())
        if len(tokens) <= max_tokens:
            return raw
        return encoder.decode(tokens[:max_tokens])
    except Exception:  # pragma: no cover
        return raw[: max_tokens * 4]


def _word_tokens(text: str) -> set[str]:
    return {t for t in (text or "").lower().replace("\n", " ").split() if t}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def mmr_indices(
    relevance: Sequence[float],
    texts: Sequence[str],
    *,
    max_items: int | None = None,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance selection order (greedy).

    ``score = lambda * relevance - (1 - lambda) * max_similarity_to_selected``.
    Similarity is token Jaccard — deterministic, embedding-free and cheap
    enough for thousands of chunks (< 50 MB RAM target).
    """
    n = len(texts)
    if n == 0:
        return []
    limit = n if max_items is None else max(0, min(max_items, n))
    if limit == 0:
        return []

    rel = list(relevance[:n])
    lo, hi = min(rel), max(rel)
    span = hi - lo
    normalized = [(r - lo) / span if span > 1e-12 else 1.0 for r in rel]
    token_sets = [_word_tokens(t) for t in texts]

    chosen: list[int] = []
    remaining = set(range(n))
    while remaining and len(chosen) < limit:
        best_idx = -1
        best_score = float("-inf")
        for idx in remaining:
            if chosen:
                max_sim = max(_jaccard(token_sets[idx], token_sets[j]) for j in chosen)
            else:
                max_sim = 0.0
            score = lambda_ * normalized[idx] - (1.0 - lambda_) * max_sim
            if score > best_score:
                best_score = score
                best_idx = idx
        chosen.append(best_idx)
        remaining.discard(best_idx)
    return chosen


def pack_batches(
    texts: Sequence[str],
    max_tokens: int,
    *,
    truncate_oversize: bool = True,
) -> list[list[str]]:
    """Greedy token-bounded batching: each batch ≤ ``max_tokens`` tokens.

    A single text larger than the budget is truncated (``truncate_oversize``)
    instead of being dropped, so no evidence silently disappears.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    batches: list[list[str]] = []
    current: list[str] = []
    used = 0
    for text in texts:
        if not (text or "").strip():
            continue
        cost = count_tokens(text)
        if cost > max_tokens:
            if not truncate_oversize:
                continue
            text = truncate_tokens(text, max_tokens)
            cost = count_tokens(text)
        if current and used + cost > max_tokens:
            batches.append(current)
            current = []
            used = 0
        current.append(text)
        used += cost
    if current:
        batches.append(current)
    return batches


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    *,
    price_input_per_1m: float = 0.0,
    price_output_per_1m: float = 0.0,
) -> float:
    """Cost estimate in USD from per-1M token prices (0 = free tier / unset)."""
    return (
        (input_tokens / 1_000_000.0) * float(price_input_per_1m)
        + (output_tokens / 1_000_000.0) * float(price_output_per_1m)
    )
