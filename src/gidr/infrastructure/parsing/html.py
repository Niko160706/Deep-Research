"""Bounded, data-only HTML extraction with source character locators."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class ExtractedText:
    text: str
    quality: float
    parser_id: str
    parser_version: str
    language: str
    start: int
    end: int
    quality_flags: tuple[str, ...] = ()
    source_spans: tuple[tuple[int, int], ...] = ()
    source_pages: tuple[int, ...] = ()


class _TextCollector(HTMLParser):
    _DROP: ClassVar[set[str]] = {"script", "style", "noscript", "svg", "iframe", "template"}
    _BOILERPLATE: ClassVar[set[str]] = {"nav", "header", "footer", "aside", "form"}

    def __init__(self, source: str, *, max_chars: int, source_offset_base: int = 0):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.source_offset_base = source_offset_base
        self.line_offsets = [0]
        self.line_offsets.extend(i + 1 for i, char in enumerate(source) if char == "\n")
        self.max_chars = max_chars
        self.characters: list[str] = []
        self.spans: list[tuple[int, int] | None] = []
        self.suppressed: list[str] = []

    def _position(self) -> int:
        line, column = self.getpos()
        return (
            self.line_offsets[min(line - 1, len(self.line_offsets) - 1)]
            + column
            + self.source_offset_base
        )

    def _append(self, text: str, start: int, end: int | None = None) -> None:
        remaining = self.max_chars - len(self.characters)
        text = text[:remaining]
        self.characters.extend(text)
        if end is None:
            self.spans.extend((start + i, start + i + 1) for i in range(len(text)))
        else:
            self.spans.extend((start, end) for _ in text)

    def _break(self) -> None:
        position = self._position()
        self._append("\n", position, position)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.suppressed:
            if tag in self._DROP or tag in self._BOILERPLATE:
                self.suppressed.append(tag)
        elif tag in self._DROP or tag in self._BOILERPLATE:
            self.suppressed.append(tag)
        elif tag in {"p", "div", "article", "main", "section", "h1", "h2", "h3", "li", "tr", "br"}:
            self._break()

    def handle_endtag(self, tag: str) -> None:
        if self.suppressed:
            if tag in self.suppressed:
                index = len(self.suppressed) - 1 - self.suppressed[::-1].index(tag)
                del self.suppressed[index:]
        elif tag in {"p", "div", "article", "main", "section", "h1", "h2", "h3", "li", "tr"}:
            self._break()

    def handle_data(self, data: str) -> None:
        if not self.suppressed:
            self._append(data, self._position())

    def handle_entityref(self, name: str) -> None:
        if not self.suppressed:
            start = self._position()
            source_start = start - self.source_offset_base
            end = self.source.find(";", source_start, min(len(self.source), source_start + 40))
            if end < 0:
                self._append(f"&{name};", start, start + len(name) + 2)
                return
            self._append(
                unescape(self.source[source_start : end + 1]),
                start,
                end + 1 + self.source_offset_base,
            )

    def handle_charref(self, name: str) -> None:
        self.handle_entityref("#" + name)


def detect_language(text: str) -> str:
    """Conservatively identify common languages and flag materially mixed text."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) < 24:
        return "und"
    latin = sum("LATIN" in unicodedata.name(c, "") for c in letters)
    if latin / len(letters) < 0.8:
        return "und"
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    lexicons = {
        "en": {
            "the",
            "and",
            "of",
            "to",
            "in",
            "is",
            "are",
            "this",
            "that",
            "for",
            "with",
            "from",
            "was",
            "were",
            "research",
        },
        "id": {
            "yang",
            "dari",
            "dan",
            "untuk",
            "dengan",
            "ini",
            "adalah",
            "pada",
            "sebagai",
            "dalam",
            "penelitian",
            "sumber",
        },
        "es": {
            "el",
            "los",
            "las",
            "del",
            "y",
            "para",
            "con",
            "es",
            "por",
            "una",
            "este",
            "investigación",
        },
        "fr": {
            "le",
            "les",
            "des",
            "et",
            "pour",
            "avec",
            "est",
            "dans",
            "une",
            "cette",
            "recherche",
        },
        "de": {"der", "die", "das", "und", "für", "mit", "ist", "eine", "dieser", "forschung"},
        "pt": {"os", "as", "dos", "das", "e", "para", "com", "uma", "este", "pesquisa"},
    }
    scores = {
        language: sum(word in vocabulary for word in words)
        for language, vocabulary in lexicons.items()
    }
    ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
    first_language, first_score = ranked[0]
    second_score = ranked[1][1]
    if first_score < 2:
        return "und"
    if second_score >= 2 and second_score / first_score >= 0.35:
        return "mixed"
    if first_score - second_score < 1:
        return "und"
    return first_language


def extract_html(
    data: bytes, *, max_bytes: int = 2_000_000, max_chars: int = 500_000
) -> ExtractedText:
    if len(data) > max_bytes or max_chars < 1:
        raise ValueError("HTML_INPUT_LIMIT")
    try:
        source = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        source = data.decode("utf-8", errors="replace")
    offset_base = 1 if data.startswith(b"\xef\xbb\xbf") else 0
    parser = _TextCollector(source, max_chars=max_chars, source_offset_base=offset_base)
    parser.feed(source)
    parser.close()
    normalized: list[str] = []
    normalized_spans: list[tuple[int, int] | None] = []
    index = 0
    while index < len(parser.characters):
        char = parser.characters[index]
        if char in "\t\f\v ":
            end = index + 1
            while end < len(parser.characters) and parser.characters[end] in "\t\f\v ":
                end += 1
            values = [span for span in parser.spans[index:end] if span is not None]
            normalized.append(" ")
            normalized_spans.append((values[0][0], values[-1][1]) if values else None)
            index = end
            continue
        if char == "\n" and normalized and normalized[-1] == "\n":
            index += 1
            continue
        normalized.append(char)
        normalized_spans.append(parser.spans[index])
        index += 1
    left = 0
    right = len(normalized)
    while left < right and normalized[left].isspace():
        left += 1
    while right > left and normalized[right - 1].isspace():
        right -= 1
    text = "".join(normalized[left:right])
    source_spans = tuple(
        span if span is not None else (0, 0) for span in normalized_spans[left:right]
    )
    flags: list[str] = []
    if "\ufffd" in text:
        flags.append("DECODE_REPLACEMENT")
        source_spans = ()
    if len(text) >= max_chars:
        flags.append("TEXT_TRUNCATED")
    if len(text) < 80:
        flags.append("LOW_TEXT_YIELD")
    quality = min(1.0, len(text) / 1000) * (0.7 if flags else 1.0)
    return ExtractedText(
        text,
        quality,
        "stdlib-html",
        "1",
        detect_language(text),
        0,
        len(source),
        tuple(flags),
        source_spans,
    )


def extract_plain_text(data: bytes, *, max_bytes: int = 2_000_000) -> ExtractedText:
    if len(data) > max_bytes:
        raise ValueError("TEXT_INPUT_LIMIT")
    text = data.decode("utf-8", errors="replace")
    flags = ("DECODE_REPLACEMENT",) if "\ufffd" in text else ()
    source_spans = (
        tuple((index, index + 1) for index in range(len(text))) if "\ufffd" not in text else ()
    )
    return ExtractedText(
        text,
        0.5 if flags else 0.8,
        "stdlib-text",
        "1",
        detect_language(text),
        0,
        len(text),
        flags,
        source_spans,
    )
