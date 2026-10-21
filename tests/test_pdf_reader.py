"""Tests for the PDF reader used by official-document scraping (Phase 3).

A tiny real PDF is built in memory with pypdf, so both extractors are exercised
without network or fixture files.
"""
from __future__ import annotations

from drx.muscle import pdf_reader


def _sample_pdf(text: str = "Hello PDF Sri Lestari") -> bytes:
    """Build a tiny text-bearing PDF in memory (same helper as scraper tests)."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    fonts = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): writer._add_object(fonts)}
    )
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 14 Tf 40 200 Td ({text}) Tj ET\n".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)
    import io

    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def test_extract_pdf_text_reads_official_document() -> None:
    pdf = _sample_pdf("SK Bupati Sleman tentang Sri Lestari")
    text = pdf_reader.extract_pdf_text(pdf)
    assert "Sri Lestari" in text
    assert "Bupati Sleman" in text


def test_extract_pdf_text_falls_back_to_pypdf(monkeypatch) -> None:
    monkeypatch.setattr(pdf_reader, "extract_with_pdfplumber", lambda _pdf: "")
    text = pdf_reader.extract_pdf_text(_sample_pdf("LHKPN Sri Lestari 2024"))
    assert "LHKPN" in text


def test_extract_pdf_text_returns_empty_on_garbage(monkeypatch) -> None:
    assert pdf_reader.extract_pdf_text(b"") == ""
    assert pdf_reader.extract_with_pdfplumber(b"not a pdf") == ""
    assert pdf_reader.extract_with_pypdf(b"not a pdf") == ""
    monkeypatch.setattr(pdf_reader, "extract_with_pdfplumber", lambda _pdf: "")
    monkeypatch.setattr(pdf_reader, "extract_with_pypdf", lambda _pdf: "")
    assert pdf_reader.extract_pdf_text(b"anything") == ""
