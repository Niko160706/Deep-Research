"""Deterministic bounded chunking with offsets into the extracted source text."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby

from gidr.infrastructure.parsing.html import detect_language


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    start: int
    end: int
    language: str


def chunk_text(
    text: str, *, target_chars: int = 4000, max_chunks: int = 256
) -> tuple[TextChunk, ...]:
    if not 128 <= target_chars <= 16_384 or not 1 <= max_chunks <= 4096:
        raise ValueError("CHUNK_LIMIT_INVALID")
    chunks: list[TextChunk] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target_chars, length)
        if end < length:
            boundary = max(
                text.rfind("\n", start + target_chars // 2, end),
                text.rfind(". ", start + target_chars // 2, end),
                text.rfind(" ", start + target_chars // 2, end),
            )
            if boundary > start:
                end = boundary + (2 if text[boundary : boundary + 2] == ". " else 1)
        piece = text[start:end]
        if piece:
            chunks.append(TextChunk(piece, start, end, detect_language(piece)))
            if len(chunks) > max_chunks:
                raise ValueError("CHUNK_COUNT_LIMIT")
        start = end
    return tuple(chunks)


def chunk_text_by_page(
    text: str, pages: tuple[int, ...], *, target_chars: int = 4000, max_chunks: int = 256
) -> tuple[TextChunk, ...]:
    """Keep each PDF chunk within one original page while retaining global offsets."""
    if len(pages) != len(text) or any(page < 1 for page in pages):
        raise ValueError("PAGE_LOCATOR_INVALID")
    if not text:
        return ()
    result: list[TextChunk] = []
    start = 0
    for page, values in groupby(pages):
        page_length = sum(1 for _ in values)
        end = start + page_length
        for chunk in chunk_text(text[start:end], target_chars=target_chars, max_chunks=max_chunks):
            result.append(
                TextChunk(
                    text=chunk.text,
                    start=start + chunk.start,
                    end=start + chunk.end,
                    language=chunk.language,
                )
            )
            if len(result) > max_chunks:
                raise ValueError("CHUNK_COUNT_LIMIT")
        start = end
    return tuple(result)
