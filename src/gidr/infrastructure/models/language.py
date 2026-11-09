"""Conservative local language observations. Ambiguity remains UNKNOWN."""

from __future__ import annotations

import unicodedata
from typing import Protocol

from gidr.contracts.common import Language


class LanguageDetector(Protocol):
    def detect(self, text: str) -> Language: ...


class ConservativeLanguageDetector:
    """Detect only scripts with a strong single-language signal; never infer Latin text."""

    _japanese = (range(0x3040, 0x30A0), range(0x30A0, 0x3100))
    _korean = (range(0xAC00, 0xD7AF), range(0x1100, 0x1200))

    def detect(self, text: str) -> Language:
        if not text.strip():
            return Language(code="unknown", status="UNKNOWN")
        scripts: set[str] = set()
        for char in text:
            codepoint = ord(char)
            if any(codepoint in block for block in self._japanese):
                scripts.add("ja")
            elif any(codepoint in block for block in self._korean):
                scripts.add("ko")
            elif "LATIN" in unicodedata.name(char, ""):
                scripts.add("latin")
            elif char.isalpha():
                scripts.add("other")
        if scripts == {"ja"}:
            return Language(code="ja", status="DETECTED")
        if scripts == {"ko"}:
            return Language(code="ko", status="DETECTED")
        if scripts <= {"ja", "other"} and "ja" in scripts:
            return Language(code="ja", status="DETECTED")
        if scripts <= {"ko", "other"} and "ko" in scripts:
            return Language(code="ko", status="DETECTED")
        if len(scripts) > 1:
            return Language(code="mixed", status="MIXED")
        return Language(code="unknown", status="UNKNOWN")
