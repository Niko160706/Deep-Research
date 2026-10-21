"""Sliding-window text chunker with overlap (100% Python, 0 tokens)."""
from __future__ import annotations

from drx.models import Document


class Chunker:
    """Splits long text into fixed-size windows with controlled overlap."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start += step
        return chunks

    def chunk_document(self, doc: Document) -> list[str]:
        return self.chunk_text(doc.text)
