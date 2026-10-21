"""PDF text extraction for official documents (SK, LHKPN, berita acara).

Both ``pdfplumber`` (layout-aware, primary) and ``pypdf`` (robust fallback) are
optional at runtime: when a library is missing or fails on a malformed file the
module returns ``""`` instead of crashing the scrape. The scraper calls
:func:`extract_pdf_text` for every URL that looks like a PDF.
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger("drx.muscle.pdf")


def extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """Layout-aware extraction via pdfplumber (tables/multi-column friendly)."""
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover - dependency guard
        log.debug("pdfplumber tidak terpasang")
        return ""
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        log.debug("pdfplumber parse error: %s", exc)
        return ""


def extract_with_pypdf(pdf_bytes: bytes) -> str:
    """Fallback extraction via pypdf (tolerates more malformed files)."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - dependency guard
        log.debug("pypdf tidak terpasang")
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as exc:
        log.debug("pypdf parse error: %s", exc)
        return ""


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Best-effort text extraction (pdfplumber -> pypdf); never raises."""
    if not pdf_bytes:
        return ""
    text = extract_with_pdfplumber(pdf_bytes)
    if text:
        return text
    return extract_with_pypdf(pdf_bytes)
