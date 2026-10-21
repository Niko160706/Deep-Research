"""Prompt template for the synthesis role."""
from __future__ import annotations


def build_synthesis_messages(query: str, context: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an investigative reporter. Synthesize the provided "
                "evidence into a structured Markdown report with provenance."
            ),
        },
        {"role": "user", "content": f"Query: {query}\nContext: {context}"},
    ]
