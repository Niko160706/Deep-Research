"""Prompt templates for Def1 (hypothesis) and Def2 (decomposition)."""
from __future__ import annotations


def build_def1_messages(query: str, mode: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a research strategist. Given a query and a mode "
                f"('{mode}'), produce a hypothesis, a strategy, and seed queries."
            ),
        },
        {"role": "user", "content": query},
    ]


def build_def2_messages(plan: dict, survey_snippets: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a research decomposer. Revise the plan using real "
                "search snippets and emit concrete sub-queries."
            ),
        },
        {
            "role": "user",
            "content": f"Plan: {plan}\nSnippets:\n" + "\n".join(survey_snippets),
        },
    ]
