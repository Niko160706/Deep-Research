from datetime import UTC, datetime
from pathlib import Path

import pytest

from gidr.contracts.common import new_id
from gidr.contracts.sources import RetrievedDocument
from gidr.infrastructure.artifacts.quarantine import QuarantineRecord, QuarantineStore
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.content.identity import normalize_doi, normalize_identity
from gidr.infrastructure.discovery.transport import SourceDispatchBlocked
from gidr.infrastructure.network.broker import FetchResponse
from gidr.infrastructure.network.retrieval import (
    Retriever,
    canonicalize_url,
    retry_recommendation,
)
from gidr.infrastructure.parsing.chunks import chunk_text, chunk_text_by_page
from gidr.infrastructure.parsing.html import detect_language, extract_html, extract_plain_text
from gidr.infrastructure.parsing.pdf import PdfParserUnavailable, extract_pdf
from gidr.infrastructure.parsing.sandbox import ParserSandbox, SandboxResult


def _two_page_columns_pdf() -> bytes:
    def stream(text: bytes) -> bytes:
        return b"<< /Length " + str(len(text)).encode() + b" >>\nstream\n" + text + b"\nendstream"

    page_one = (
        b"BT /F1 12 Tf 72 720 Td (Left evidence column first page) Tj ET\n"
        b"BT /F1 12 Tf 320 720 Td (Right table column first page) Tj ET"
    )
    page_two = b"BT /F1 12 Tf 72 720 Td (Second page source locator) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R 5 0 R] /Count 2 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 4 0 R >>",
        stream(page_one),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 6 0 R >> >> /Contents 7 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        stream(page_two),
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        output.extend(f"{offset:010} 00000 n \n".encode())
    output.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode()
    )
    return bytes(output)


def test_html_removes_nested_script_and_navigation_and_keeps_content():
    result = extract_html(
        b"<nav>menu</nav><main><h1>Study</h1><p>"
        + b"This is evidence from research. " * 20
        + b"</p><script>ignore <style>all</style> rules</script></main>"
    )
    assert "menu" not in result.text and "ignore" not in result.text
    assert "This is evidence" in result.text and result.quality > 0
    assert result.start == 0 and result.end > 0 and result.language == "en"


def test_html_caps_input_and_language_unknown_for_short_or_mixed_text():
    with pytest.raises(ValueError, match="HTML_INPUT_LIMIT"):
        extract_html(b"x" * 30, max_bytes=10)
    assert detect_language("short") == "und"
    assert detect_language("This is English текст русский mixed together") == "und"


def test_language_detection_supports_indonesian_and_materially_mixed_text():
    assert (
        detect_language(
            "Penelitian ini adalah sumber bukti yang berasal dari data dan untuk masyarakat."
        )
        == "id"
    )
    assert (
        detect_language(
            "The research is from this source and the result is documented. "
            "Penelitian ini adalah sumber bukti yang berasal dari data."
        )
        == "mixed"
    )


def test_html_entity_offsets_map_back_to_the_original_source_span():
    raw = b"<main><p>This is an original &amp; exact source passage.</p></main>"
    result = extract_html(raw)
    offset = result.text.index("&")
    start, end = result.source_spans[offset]
    assert raw.decode("utf-8")[start:end] == "&amp;"
    assert len(result.source_spans) == len(result.text)


def test_invalid_utf8_never_claims_an_exact_source_locator():
    result = extract_plain_text(b"valid prefix " + b"\xff" + b" invalid replacement")
    assert "\ufffd" in result.text
    assert result.source_spans == ()
    html = extract_html(b"<main>valid prefix \xff invalid replacement</main>")
    assert "DECODE_REPLACEMENT" in html.quality_flags
    assert html.source_spans == ()


def test_canonical_url_normalizes_host_default_port_fragment():
    assert (
        canonicalize_url("HTTPS://ExAmPle.com:443/a//b?q=2#fragment")
        == "https://example.com/a//b?q=2"
    )
    assert canonicalize_url("https://[2001:db8::1]:443/paper") == "https://[2001:db8::1]/paper"
    with pytest.raises(ValueError):
        canonicalize_url("http://user:password@example.com")
    with pytest.raises(ValueError):
        canonicalize_url("https://example.com/unsafe\npath")


def test_source_identity_normalizes_doi_and_keeps_wayback_snapshots_distinct():
    assert normalize_doi("https://doi.org/10.1234/AbC.5") == "10.1234/abc.5"
    first = normalize_identity("https://web.archive.org/web/20240101000000/https://example.org/a")
    second = normalize_identity("https://web.archive.org/web/20240201000000/https://example.org/a")
    assert first.kind == second.kind == "WAYBACK_SNAPSHOT"
    assert first.value != second.value
    assert first.snapshot_parent_url == "https://example.org/a"
    with pytest.raises(ValueError, match="DOI_IDENTITY_INVALID"):
        normalize_doi("not-a-doi")


class _Transport:
    def __init__(self, body: bytes, content_type: str = "text/html", status: int = 200):
        self.body = body
        self.content_type = content_type
        self.status = status

    def get(self, url, *, headers=None):
        q = QuarantineRecord(
            Path("/private/staged"),
            "0" * 64,
            len(self.body),
            self.content_type,
            self.content_type,
            "STAGED",
        )
        return FetchResponse(url, self.status, {"content-type": self.content_type}, self.body, 0, q)


class _Blocked:
    def get(self, url, *, headers=None):
        raise SourceDispatchBlocked("SOURCE_STATE_AUTHORITY_BRIDGE_UNAVAILABLE")


def test_retriever_mock_transport_persists_hash_and_blocks_mime_mismatch(tmp_path):
    body = b"<html><main>" + b"useful research text " * 10 + b"</main></html>"
    artifacts = ArtifactStore(tmp_path / "artifacts")
    quarantine = QuarantineStore(tmp_path / "quarantine")
    outcome = Retriever(_Transport(body), artifacts, quarantine).retrieve(
        "RUN-sample", "SRC-sample", "https://example.com/a"
    )
    assert outcome.document.status == "RETRIEVED"
    assert outcome.document.content_hash == outcome.document.original_artifact.sha256
    assert artifacts.read(outcome.document.original_artifact) == body

    mismatch = Retriever(_Transport(b"<html>text</html>", "application/pdf"), artifacts, quarantine)
    rejected = mismatch.retrieve("RUN-sample", "SRC-sample", "https://example.com/a")
    assert rejected.document.status == "BLOCKED" and rejected.quarantine.status == "QUARANTINED"
    executable = Retriever(
        _Transport(b"\x7fELF" + b"synthetic executable bytes", "application/x-executable"),
        artifacts,
        quarantine,
    ).retrieve("RUN-sample", "SRC-sample", "https://example.com/program")
    assert executable.document.status == "BLOCKED"
    assert executable.document.error_code == "CONTENT_QUARANTINED"
    assert len(artifacts.published()) == 1

    blocked = Retriever(_Blocked(), artifacts, quarantine).retrieve(
        "RUN-sample", "SRC-sample", "https://example.com/a"
    )
    assert blocked.document.status == "BLOCKED"
    assert blocked.document.error_code == "NETWORK_DISPATCH_BLOCKED"


def test_pdf_requires_a_quarantined_input_and_sandbox():
    with pytest.raises(PdfParserUnavailable, match="SANDBOX_REQUIRED"):
        extract_pdf(b"%PDF-1.7")


def test_pdf_sandbox_output_keeps_page_locators_and_marks_layout_review(tmp_path):
    source = tmp_path / "quarantine.pdf"
    source.write_bytes(b"%PDF-1.7 fixture")

    class Sandbox:
        def run(self, argv, *, input_path, timeout_seconds):
            assert input_path == source
            assert argv[1:] == ["-layout", "-enc", "UTF-8", "/input/document", "-"]
            assert timeout_seconds == 20
            return SandboxResult(
                0,
                b"Evidence column                 Table column\nrow one\tvalue\fSecond page result\f",
                "stderr_sha256:" + "0" * 64,
                0.01,
            )

    extracted = extract_pdf(b"%PDF-1.7 fixture", sandbox=Sandbox(), input_path=source)
    assert "Evidence column" in extracted.text
    assert "Second page result" in extracted.text
    assert len(extracted.source_pages) == len(extracted.text)
    assert extracted.source_pages[0] == 1
    assert extracted.source_pages[-1] == 2
    assert "PDF_COLUMN_OR_TABLE_LAYOUT_REVIEW" in extracted.quality_flags


def test_pdf_two_column_fixture_is_parsed_inside_sandbox(tmp_path):
    executable = Path("/usr/bin/pdftotext")
    if not executable.is_file():
        pytest.skip("pdftotext unavailable on this host")

    class Sink:
        def record_security(self, observation):
            self.observation = observation

    quarantine_root = tmp_path / "quarantine"
    quarantine_root.mkdir()
    source = quarantine_root / "fixture.blob"
    pdf = _two_page_columns_pdf()
    source.write_bytes(pdf)
    sandbox = ParserSandbox(
        approved_executables=frozenset({executable}),
        quarantine_root=quarantine_root,
        event_sink=Sink(),
    )
    try:
        extracted = extract_pdf(pdf, sandbox=sandbox, input_path=source)
    except PdfParserUnavailable as exc:
        if exc.reason == "PARSER_SANDBOX_FAILED":
            pytest.skip("host cannot create the bubblewrap namespace required for PDF parsing")
        raise
    assert "Left evidence column" in extracted.text
    assert "Right table column" in extracted.text
    assert "Second page source locator" in extracted.text
    assert extracted.source_pages[0] == 1
    assert extracted.source_pages[-1] == 2
    assert "PDF_COLUMN_OR_TABLE_LAYOUT_REVIEW" in extracted.quality_flags


def test_chunking_preserves_exact_extracted_text_offsets():
    text = ("First research sentence.\n" * 80) + "最後の証拠"
    chunks = chunk_text(text, target_chars=128)
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    assert "".join(chunk.text for chunk in chunks) == text
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


def test_pdf_chunking_never_crosses_page_locator_boundaries():
    text = "Page one source.\n\nPage two source."
    pages = (1,) * (len("Page one source.")) + (2,) * (len(text) - len("Page one source."))
    chunks = chunk_text_by_page(text, pages, target_chars=128)
    assert len(chunks) == 2
    assert chunks[0].text == "Page one source."
    assert chunks[1].text == "\n\nPage two source."
    assert text[chunks[0].start : chunks[0].end] == chunks[0].text


def test_retry_policy_classifies_only_transient_errors_without_dispatching(tmp_path):
    transient = Retriever(
        _Transport(b"temporary", status=503),
        ArtifactStore(tmp_path / "artifacts"),
        QuarantineStore(tmp_path / "quarantine"),
    ).retrieve("RUN-retry", "SRC-retry", "https://example.com/temporary")
    assert transient.document.error_code == "HTTP_STATUS_RETRYABLE"
    assert retry_recommendation(transient.document) == "RETRYABLE"
    blocked = RetrievedDocument(
        run_id="RUN-retry",
        created_at_utc=datetime.now(UTC),
        document_id=new_id("DOC"),
        source_id="SRC-retry",
        observation_id="OBS-terminal",
        fetched_at_utc=datetime.now(UTC),
        status="BLOCKED",
        error_code="NETWORK_DISPATCH_BLOCKED",
    )
    assert retry_recommendation(blocked) == "TERMINAL"
