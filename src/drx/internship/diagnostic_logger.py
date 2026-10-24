"""Accessibility diagnostics for every internship source hit.

The internship engine talks to several websites that have very different
failure modes: Cloudflare challenges, login walls, dead selectors, empty
result pages and plain network timeouts.  When one of those happens we do not
want a vague "search failed" log line -- we want a machine-readable record per
URL plus a terminal table that explains exactly what happened.

This module intentionally has no dependency on the rest of the project.  It is
safe to import from tests, scripts and future internship sources.

Output file::

    runs/<run_id>/internship_source_diagnostics.json

The JSON document contains a ``summary`` block and a ``sources`` list.  Every
source entry has the required fields:

``source_name``, ``target_url``, ``access_status``, ``http_status_code``,
``latency_ms``, ``raw_chars_scraped``, ``items_extracted_count`` and
``failure_reason``.

Allowed access statuses are exactly: ``SUCCESS``, ``EMPTY_RESULT``,
``BLOCKED`` and ``TIMEOUT``.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

#: Canonical access statuses (the only values the engine ever writes).
SUCCESS = "SUCCESS"
EMPTY_RESULT = "EMPTY_RESULT"
BLOCKED = "BLOCKED"
TIMEOUT = "TIMEOUT"

ALL_STATUSES = (SUCCESS, EMPTY_RESULT, BLOCKED, TIMEOUT)


@dataclass
class SourceDiagnostic:
    """One concrete fetch attempt against one concrete URL."""

    source_name: str
    target_url: str
    access_status: str = BLOCKED
    http_status_code: int = 0
    failure_reason: str = ""
    latency_ms: int = 0
    items_extracted_count: int = 0
    # Extra context is deliberately included: it makes future debugging much
    # easier without having to re-run the whole investigation.
    query: str = ""
    raw_chars_scraped: int = 0
    content_type: str = ""
    response_bytes: int = 0
    fetch_mode: str = "aiohttp"
    notes: str = ""
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Return the stable JSON representation (required fields first)."""
        data = asdict(self)
        return {
            "source_name": data["source_name"],
            "target_url": data["target_url"],
            "access_status": data["access_status"],
            "http_status_code": int(data["http_status_code"] or 0),
            "failure_reason": data["failure_reason"],
            "latency_ms": int(data["latency_ms"] or 0),
            "raw_chars_scraped": int(data["raw_chars_scraped"] or 0),
            "items_extracted_count": int(data["items_extracted_count"] or 0),
            "query": data["query"],
            "content_type": data["content_type"],
            "response_bytes": int(data["response_bytes"] or 0),
            "fetch_mode": data["fetch_mode"],
            "notes": data["notes"],
            "checked_at": data["checked_at"],
        }


class DiagnosticLogger:
    """Collect source diagnostics and render a transparent terminal table."""

    def __init__(self, run_id: str = "", *, clock=datetime.now) -> None:
        self.run_id = run_id
        self._clock = clock
        self.records: list[SourceDiagnostic] = []

    # ---- collection -----------------------------------------------------

    def begin(
        self,
        source_name: str,
        target_url: str,
        *,
        query: str = "",
        fetch_mode: str = "aiohttp",
        content_type: str = "",
        response_bytes: int = 0,
    ) -> SourceDiagnostic:
        """Create and register a diagnostic before a fetch starts."""
        record = SourceDiagnostic(
            source_name=source_name,
            target_url=target_url,
            query=query,
            fetch_mode=fetch_mode,
            content_type=content_type,
            response_bytes=response_bytes,
        )
        self.records.append(record)
        return record

    def complete(
        self,
        record: SourceDiagnostic,
        *,
        status: str,
        http_status_code: int | None = None,
        failure_reason: str = "",
        latency_ms: int | None = None,
        items_extracted_count: int | None = None,
        raw_chars_scraped: int | None = None,
        content_type: str | None = None,
        response_bytes: int | None = None,
        notes: str | None = None,
        fetch_mode: str | None = None,
    ) -> SourceDiagnostic:
        """Finalise one record (safe to call from exception handlers)."""
        if status not in ALL_STATUSES:
            # Unknown/legacy statuses degrade to BLOCKED -- NEVER to another
            # value outside the allowed set.
            status = BLOCKED
        record.access_status = status
        if http_status_code is not None:
            record.http_status_code = int(http_status_code or 0)
        record.failure_reason = (failure_reason or "").strip()
        if latency_ms is not None:
            record.latency_ms = max(0, int(latency_ms))
        if items_extracted_count is not None:
            record.items_extracted_count = max(0, int(items_extracted_count))
        if raw_chars_scraped is not None:
            record.raw_chars_scraped = max(0, int(raw_chars_scraped))
        if content_type is not None:
            record.content_type = content_type
        if response_bytes is not None:
            record.response_bytes = max(0, int(response_bytes))
        if notes is not None:
            record.notes = notes
        if fetch_mode is not None:
            record.fetch_mode = fetch_mode
        return record

    def fail(
        self,
        record: SourceDiagnostic,
        *,
        status: str = BLOCKED,
        http_status_code: int = 0,
        failure_reason: str = "",
        latency_ms: int | None = None,
    ) -> SourceDiagnostic:
        return self.complete(
            record,
            status=status,
            http_status_code=http_status_code,
            failure_reason=failure_reason,
            latency_ms=latency_ms,
            items_extracted_count=0,
        )

    # ---- summaries ------------------------------------------------------

    def summary(self) -> dict:
        """Counts by status plus a small set of useful totals."""
        counts = {status: 0 for status in ALL_STATUSES}
        for record in self.records:
            if record.access_status not in counts:
                counts[record.access_status] = 0
            counts[record.access_status] += 1
        return {
            "sources": len(self.records),
            "success": counts[SUCCESS],
            "blocked": counts[BLOCKED],
            "timeout": counts[TIMEOUT],
            "empty_result": counts[EMPTY_RESULT],
            "items_extracted": sum(r.items_extracted_count for r in self.records),
            "raw_chars_scraped_total": sum(r.raw_chars_scraped for r in self.records),
            "latency_ms_total": sum(r.latency_ms for r in self.records),
        }

    def write(self, run_dir: str | Path) -> Path:
        """Write ``internship_source_diagnostics.json`` and return its path."""
        path = Path(run_dir) / "internship_source_diagnostics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "run_id": self.run_id,
            "generated_at": self._clock().astimezone(timezone.utc).isoformat(),
            "summary": self.summary(),
            "sources": [record.to_dict() for record in self.records],
        }
        path.write_text(
            json.dumps(document, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    # ---- terminal rendering --------------------------------------------

    @staticmethod
    def _shorten(value: object, width: int) -> str:
        text = " ".join(str(value or "").split())
        if width <= 0 or len(text) <= width:
            return text
        return text[: max(1, width - 1)].rstrip() + "…"

    @staticmethod
    def _short_chars(value: object) -> str:
        count = int(value or 0)
        if count >= 1000:
            return f"{count / 1000:.1f}k"
        return str(count)

    def summary_table(self, *, max_reason: int = 48) -> str:
        """Return a compact ASCII table suitable for the terminal/report.

        The table is intentionally hand-rendered (no ``tabulate`` dependency)
        and preserves the full reasons in the JSON file.  A detail block is
        appended for rows whose reason was truncated so the user can still see
        the complete explanation.
        """
        headers = ("SUMBER", "STATUS", "HTTP", "LATENSI", "RAW", "ITEM", "ALASAN")
        rows: list[tuple[str, ...]] = []
        for record in self.records:
            rows.append(
                (
                    self._shorten(record.source_name, 28),
                    record.access_status,
                    str(record.http_status_code or "-"),
                    f"{record.latency_ms} ms",
                    self._short_chars(record.raw_chars_scraped),
                    str(record.items_extracted_count),
                    self._shorten(record.failure_reason or "-", max_reason),
                )
            )
        if not rows:
            rows.append(("-", "-", "-", "-", "-", "-", "-"))

        widths = [len(h) for h in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def line(char: str = "-") -> str:
            return "+" + "+".join(char * (width + 2) for width in widths) + "+"

        def fmt(row: tuple[str, ...]) -> str:
            cells = [
                f" {cell.ljust(widths[index])} "
                for index, cell in enumerate(row)
            ]
            return "|" + "|".join(cells) + "|"

        out = [line("-"), fmt(headers), line("=")]
        out.extend(fmt(row) for row in rows)
        out.append(line("-"))

        truncated = [
            r for r in self.records if len(r.failure_reason or "") > max_reason
        ]
        if truncated:
            out.append("Detail alasan kegagalan:")
            for record in truncated:
                out.append(
                    f"  - {record.source_name} [{record.access_status}] "
                    f"{record.failure_reason}"
                )

        return "\n".join(out)

    def print_summary(self, *, file=None) -> None:
        """Print the diagnostics table followed by full URLs for failures."""
        import sys

        stream = file or sys.stdout
        print("\nTabel Ringkasan Diagnostik Sumber", file=stream)
        print(self.summary_table(), file=stream)
        for record in self.records:
            if record.access_status in {SUCCESS}:
                continue
            print(
                f"  · {record.source_name}: {record.target_url}",
                file=stream,
            )

    @staticmethod
    def render_records(records: Iterable[SourceDiagnostic]) -> str:
        """Convenience helper for callers that only have a list of records."""
        logger = DiagnosticLogger()
        logger.records.extend(records)
        return logger.summary_table()
