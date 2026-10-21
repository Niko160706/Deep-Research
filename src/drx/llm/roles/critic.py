"""Prompt template for the critic role."""
from __future__ import annotations


def build_critic_messages(graph_summary: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a rigorous critic. Identify contradictions, gaps and "
                "weak causal links, then propose follow-up sub-queries."
            ),
        },
        {"role": "user", "content": str(graph_summary)},
    ]
