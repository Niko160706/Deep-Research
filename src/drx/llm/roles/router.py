"""Prompt template for the LLM Router role."""
from __future__ import annotations

ROUTER_SYSTEM = (
    "You are a research triage router. Classify the query as exactly one of: "
    "'entity' (who/what), 'heuristic' (why/how), or 'hybrid'. Reply with the "
    "single word only."
)


def build_messages(query: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": query},
    ]
