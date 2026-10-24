"""Interactive sub-menu for Internship Hunter.

The top-level *mode* prompt lives in ``main.py`` so the existing CLI/Web App
selection continues to work unchanged.  This module owns the three internship
input strategies and can be called independently in tests.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .models import Approach


@dataclass
class InternshipWizardResult:
    approach: Approach
    role: str = ""
    skills: list[str] | None = None
    location: str = "Indonesia OR Remote"
    work_mode: str = ""

    def __post_init__(self) -> None:
        if self.skills is None:
            self.skills = []


def _clean(value: str) -> str:
    return " ".join(str(value or "").strip().split())


def _parse_skills(value: str) -> list[str]:
    out: list[str] = []
    for part in str(value or "").replace(";", ",").split(","):
        clean = _clean(part)
        if clean and clean.casefold() not in {item.casefold() for item in out}:
            out.append(clean)
    return out


def prompt_internship_setup(
    *,
    ask: Callable[[str], str] | None = None,
    default_location: str = "Indonesia OR Remote",
) -> InternshipWizardResult:
    """Prompt for A/B/C strategy and its required input.

    ``ask`` is injectable (``main._ask``) so EOF/Ctrl+C behaviour remains
    consistent with the General Investigation wizard.
    """
    ask = ask or input
    print()
    print("=" * 64)
    print("  INTERNSHIP HUNTER ENGINE")
    print("=" * 64)
    print("Pilih pendekatan pencarian magang tech:")
    print("  [A] Sub-Opsi A — Spesifik Role")
    print('       contoh: "AI Engineer" -> kueri role murni (ML/Data Science)')
    print("  [B] Sub-Opsi B — Open Discovery")
    print("       contoh: semua magang tech tanpa spesialisasi")
    print("  [C] Sub-Opsi C — Skill-Based Matching")
    print('       contoh: "Python, Go, SQL, Docker" -> 3 peran paling cocok')

    while True:
        choice = _clean(ask("Pendekatan [A/B/C, default B]: ")) or "b"
        if choice in {"1", "a", "role"}:
            approach: Approach = "role"
            break
        if choice in {"2", "b", "open", "discovery"}:
            approach = "discovery"
            break
        if choice in {"3", "c", "skill", "skills"}:
            approach = "skill"
            break
        print("  ! Pilihan tidak dikenal — masukkan A, B, atau C.")

    role = ""
    skills: list[str] = []
    if approach == "role":
        while not role:
            role = _clean(ask("Peran spesifik (contoh: AI Engineer): "))
            if not role:
                print("  ! Peran tidak boleh kosong.")
    elif approach == "skill":
        while not skills:
            skills = _parse_skills(
                ask("Daftar skill dipisah koma (contoh: Python, Go, SQL, Docker): ")
            )
            if not skills:
                print("  ! Minimal satu skill diperlukan.")

    location = _clean(
        ask(
            "Lokasi/moda pencarian "
            f'(contoh: "Surabaya OR Remote") [default {default_location}]: '
        )
    ) or default_location
    work_mode = _clean(
        ask("Mode kerja (Remote/Hybrid/On-site; kosong = semua): ")
    )
    result = InternshipWizardResult(
        approach=approach,
        role=role,
        skills=skills,
        location=location,
        work_mode=work_mode,
    )
    print()
    print("Konfigurasi Internship Hunter:")
    print(f"  Pendekatan : {result.approach}")
    if result.role:
        print(f"  Role       : {result.role}")
    if result.skills:
        print(f"  Skills     : {', '.join(result.skills)}")
    print(f"  Lokasi     : {result.location}")
    print(f"  Mode kerja : {result.work_mode or '(semua)'}")
    print("-" * 64)
    return result
