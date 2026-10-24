"""Markdown and terminal reports for Internship Hunter."""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .diagnostic_logger import DiagnosticLogger, SourceDiagnostic
from .models import InternshipJob, InternshipPlan


def _md(value: object) -> str:
    """Escape one Markdown table cell."""
    return " ".join(str(value or "").split()).replace("|", "\\|")


def jobs_table(jobs: Sequence[InternshipJob]) -> str:
    """The required *Tabel Ringkasan Magang Siap Apply*."""
    lines = [
        "| No | Posisi | Perusahaan | Lokasi | Mode | Deadline | Apply |",
        "|---:|---|---|---|---|---|---|",
    ]
    if not jobs:
        lines.append("| - | (tidak ada lowongan valid) | - | - | - | - | - |")
        return "\n".join(lines)
    for index, job in enumerate(jobs, start=1):
        apply_cell = (
            f"[Buka lowongan]({job.apply_url})"
            if job.apply_url.startswith(("http://", "https://"))
            else "-"
        )
        lines.append(
            "| {index} | {role} | {company} | {location} | {mode} | {deadline} | {apply} |".format(
                index=index,
                role=_md(job.role_title),
                company=_md(job.company_name),
                location=_md(job.location),
                mode=_md(job.work_mode),
                deadline=_md(job.deadline or "-"),
                apply=apply_cell,
            )
        )
    return "\n".join(lines)


def diagnostics_table(records: Sequence[SourceDiagnostic]) -> str:
    logger = DiagnosticLogger()
    logger.records.extend(records)
    return logger.summary_table(max_reason=80)


def build_markdown_report(
    *,
    run_id: str,
    plan: InternshipPlan,
    jobs: Sequence[InternshipJob],
    diagnostics: Sequence[SourceDiagnostic],
    generated_at: str | None = None,
) -> str:
    """Build the full report, with the ready-to-apply table at the very top."""
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    status_counts = Counter(record.access_status for record in diagnostics)
    lines: list[str] = [
        "# Internship Hunter Report",
        "",
        "## Tabel Ringkasan Magang Siap Apply",
        "",
        jobs_table(jobs),
        "",
        f"**Total lowongan valid:** {len(jobs)}  ",
        f"**Run ID:** `{run_id}`  ",
        f"**Dibuat:** {generated_at}",
        "",
        "## Strategi Pencarian",
        "",
        f"- **Pendekatan:** {plan.approach}",
        f"- **Input:** {plan.user_input or '-'}",
        f"- **Lokasi/moda:** {plan.location or '-'}"
        + (f" · {plan.work_mode}" if plan.work_mode else ""),
        f"- **Peran yang dipetakan:** {', '.join(plan.roles) or '-'}",
        "- **Kueri yang dieksekusi:**",
    ]
    lines.extend(f"  - `{query}`" for query in plan.queries)
    if plan.rationale:
        lines.extend(["", f"> {plan.rationale}"])
    if plan.used_fallback:
        lines.extend(
            [
                "",
                "> Catatan: planner memakai fallback deterministik"
                + (f" ({plan.error})" if plan.error else "")
                + ".",
            ]
        )
    elif plan.provider:
        lines.extend(["", f"Planner: `{plan.provider}` / `{plan.model}`."])

    lines.extend(
        [
            "",
            "## Diagnostik Akses Sumber",
            "",
            "Ringkasan status: "
            + ", ".join(
                f"{status}={status_counts.get(status, 0)}"
                for status in ("SUCCESS", "EMPTY_RESULT", "BLOCKED", "TIMEOUT")
            ),
            "",
            "```text",
            diagnostics_table(diagnostics),
            "```",
            "",
            "## Detail Lowongan",
            "",
        ]
    )
    if not jobs:
        lines.append(
            "Tidak ada lowongan yang lolos validasi. Periksa tabel diagnostik di "
            "atas untuk melihat sumber yang terblokir/kosong, lalu coba kueri lain."
        )
    else:
        for index, job in enumerate(jobs, start=1):
            lines.extend(
                [
                    f"### {index}. {job.role_title}",
                    "",
                    f"- **Perusahaan:** {job.company_name}",
                    f"- **Lokasi:** {job.location}",
                    f"- **Mode kerja:** {job.work_mode}",
                    f"- **Deadline:** {job.deadline or 'Tidak disebutkan'}",
                    f"- **Apply:** {job.apply_url}",
                ]
            )
            if job.requirements:
                lines.append("- **Requirements:**")
                lines.extend(f"  - {item}" for item in job.requirements)
            lines.append("")
    lines.extend(
        [
            "",
            "## Catatan Transparansi",
            "",
            (
                "Semua sumber diakses langsung ke URL target. Status, kode HTTP, "
                "latensi, raw_chars_scraped, jumlah item, dan alasan kegagalan "
                "lengkap tersimpan di `internship_source_diagnostics.json`. "
                "Mode kerja dinormalisasi ke WFH/WFO/Hybrid; hanya kandidat "
                "dengan `is_valid_internship=true` yang dimasukkan ke tabel."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_reports(
    run_dir: str | Path,
    markdown: str,
) -> tuple[Path, Path]:
    """Write both ``report.md`` and ``report_llm.md`` (same curated content)."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = root / "report.md"
    llm_report = root / "report_llm.md"
    report.write_text(markdown, encoding="utf-8")
    llm_report.write_text(markdown, encoding="utf-8")
    return report, llm_report
