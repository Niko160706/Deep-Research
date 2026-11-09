"""Page-aware PDF text extraction through the restricted parser sandbox."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from gidr.infrastructure.parsing.html import ExtractedText, detect_language
from gidr.infrastructure.parsing.sandbox import ParserSandbox, SandboxError


class PdfParserUnavailable(RuntimeError):
    def __init__(self, reason: str = "PDF_PARSER_DEPENDENCY_UNAVAILABLE"):
        super().__init__(reason)
        self.reason = reason


def available(executable: str = "/usr/bin/pdftotext") -> bool:
    return Path(executable).is_file() and shutil.which(executable) is not None


def extract_pdf(
    data: bytes,
    *,
    sandbox: ParserSandbox | None = None,
    input_path: Path | None = None,
    executable: str = "/usr/bin/pdftotext",
    max_output_chars: int = 1_500_000,
) -> ExtractedText:
    """Extract layout-preserving text and page locators without in-process parsing.

    The caller must pass the successful quarantine record path and an explicitly
    configured ParserSandbox that approves this exact executable.
    """
    if not data.startswith(b"%PDF-"):
        raise ValueError("PDF_SIGNATURE_INVALID")
    if len(data) > 10_000_000 or max_output_chars < 1:
        raise ValueError("PDF_INPUT_LIMIT")
    if sandbox is None or input_path is None:
        raise PdfParserUnavailable("PDF_SANDBOX_REQUIRED")
    if not available(executable):
        raise PdfParserUnavailable("PDF_PARSER_DEPENDENCY_UNAVAILABLE")
    try:
        result = sandbox.run(
            [executable, "-layout", "-enc", "UTF-8", "/input/document", "-"],
            input_path=input_path,
            timeout_seconds=20,
        )
    except SandboxError as exc:
        raise PdfParserUnavailable(exc.reason) from exc
    if result.returncode != 0:
        raise PdfParserUnavailable("PDF_PARSE_FAILED")
    try:
        page_outputs = result.stdout.decode("utf-8", errors="strict").split("\f")
    except UnicodeDecodeError as exc:
        raise PdfParserUnavailable("PDF_TEXT_ENCODING_INVALID") from exc

    pieces: list[str] = []
    pages: list[int] = []
    cursor = 0
    for page_number, page_output in enumerate(page_outputs, start=1):
        page_text = "\n".join(line.rstrip() for line in page_output.splitlines()).strip()
        if not page_text:
            continue
        if pieces:
            pieces.append("\n\n")
            pages.extend((page_number,) * 2)
            cursor += 2
        pieces.append(page_text)
        pages.extend((page_number,) * len(page_text))
        cursor += len(page_text)
        if cursor > max_output_chars:
            raise PdfParserUnavailable("PDF_TEXT_OUTPUT_LIMIT")
    text = "".join(pieces)
    if not text:
        raise PdfParserUnavailable("PDF_NO_EXTRACTABLE_TEXT")
    layout_suspected = bool(re.search(r"\S {3,}\S", text))
    flags = ("PDF_COLUMN_OR_TABLE_LAYOUT_REVIEW",) if layout_suspected else ()
    quality = min(1.0, len(text) / 1500) * (0.65 if flags else 0.85)
    return ExtractedText(
        text=text,
        quality=quality,
        parser_id="pdftotext-layout-sandbox",
        parser_version="system",
        language=detect_language(text),
        start=0,
        end=len(text),
        quality_flags=flags,
        source_spans=tuple((index, index + 1) for index in range(len(text))),
        source_pages=tuple(pages),
    )
