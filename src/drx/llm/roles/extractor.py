"""Prompt template for the fact extractor role."""
from __future__ import annotations


def build_extractor_messages(text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Extract facts as subject-predicate-object triples with a "
                "confidence score and provenance. Output strict JSON."
            ),
        },
        {"role": "user", "content": text},
    ]
