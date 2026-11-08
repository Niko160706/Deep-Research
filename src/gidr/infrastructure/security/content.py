"""Keep source text in a data-only envelope and flag instruction-like material."""

from __future__ import annotations

import base64
import hashlib
import html
import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UntrustedSourceContent:
    source_id: str
    content_sha256: str
    text: str
    trust_label: str = "UNTRUSTED_SOURCE_CONTENT"


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    reason_code: str
    matched: bool


class SourceContentGuard:
    """Detection is advisory; the data-only type grants no tools or permissions."""

    _patterns = (
        re.compile(
            r"\b(ignore|disregard|override)\b.{0,80}\b(instruction|system prompt|policy)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\b(read|send|upload|exfiltrate)\b.{0,80}\b(secret|credential|\.env|api.?key)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(abaikan|lewati).{0,60}(instruksi|sistem|kunci|rahasia)", re.IGNORECASE),
        re.compile(r"<\s*(script|iframe)\b|javascript\s*:", re.IGNORECASE),
    )

    def wrap(
        self, source_id: str, text: str
    ) -> tuple[UntrustedSourceContent, tuple[InjectionFinding, ...]]:
        if not source_id or len(text.encode("utf-8")) > 2_000_000:
            raise ValueError("SOURCE_CONTENT_LIMIT")
        decoded: list[str] = []
        for token in re.findall(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{40,}={0,2}", text):
            try:
                candidate = base64.b64decode(token, validate=True).decode("utf-8")
            except (ValueError, UnicodeError):
                continue
            decoded.append(candidate[:8192])
        haystack = text + "\n" + "\n".join(decoded)
        findings = tuple(
            InjectionFinding(f"PROMPT_INJECTION_PATTERN_{index:02d}", True)
            for index, pattern in enumerate(self._patterns, start=1)
            if pattern.search(haystack)
        )
        wrapped = UntrustedSourceContent(source_id, hashlib.sha256(text.encode()).hexdigest(), text)
        return wrapped, findings

    @staticmethod
    def terminal_text(text: str) -> str:
        """Strip terminal control sequences before showing source text in a TTY."""
        return "".join(
            char for char in text if char in "\n\r\t" or ord(char) >= 32 and ord(char) != 127
        )

    @staticmethod
    def html_text(text: str) -> str:
        return html.escape(text, quote=True)
