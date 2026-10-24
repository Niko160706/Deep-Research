"""Typed data containers for the Internship Hunter engine.

Only the internship feature imports these models.  Mode 1's schemas in
``drx.models`` are deliberately left untouched.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Approach = Literal["role", "discovery", "skill"]


@dataclass
class InternshipPlan:
    """The search strategy produced by the LLM planner or deterministic fallback."""

    approach: Approach
    user_input: str = ""
    roles: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    location: str = "Indonesia OR Remote"
    work_mode: str = ""
    rationale: str = ""
    provider: str = ""
    model: str = ""
    used_fallback: bool = False
    error: str = ""

    @property
    def display_title(self) -> str:
        if self.approach in {"role", "skill"}:
            role = self.roles[0] if self.roles else "Tech"
            # Keep the root distinct from any child label and always in the
            # ``Magang [Role]`` format required by the PRD.
            return f"Magang {role}"
        return "Magang Tech"

    @property
    def search_context(self) -> str:
        bits = [self.location] if self.location else []
        if self.work_mode:
            bits.append(self.work_mode)
        if self.approach == "skill" and self.user_input:
            bits.append(f"skills: {self.user_input}")
        return " · ".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RawJob:
    """A candidate extracted from HTML/JSON/Markdown before LLM curation."""

    role_title: str
    company_name: str = ""
    location: str = ""
    work_mode: str = ""
    requirements: list[str] = field(default_factory=list)
    deadline: str = ""
    apply_url: str = ""
    description: str = ""
    source_name: str = ""
    source_url: str = ""
    is_internship_hint: bool | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InternshipJob:
    """The exact normalised schema requested by the internship PRD."""

    role_title: str
    company_name: str
    location: str
    work_mode: str
    requirements: list[str]
    deadline: str
    apply_url: str
    is_valid_internship: bool

    def to_dict(self) -> dict:
        """Serialise only the eight contract fields."""
        return {
            "role_title": self.role_title,
            "company_name": self.company_name,
            "location": self.location,
            "work_mode": self.work_mode,
            "requirements": list(self.requirements),
            "deadline": self.deadline,
            "apply_url": self.apply_url,
            "is_valid_internship": bool(self.is_valid_internship),
        }

    @classmethod
    def from_mapping(cls, data: dict) -> InternshipJob:
        requirements = data.get("requirements") or []
        if isinstance(requirements, str):
            requirements = [
                part.strip()
                for part in requirements.replace(";", ",").split(",")
                if part.strip()
            ]
        elif not isinstance(requirements, (list, tuple)):
            requirements = []
        valid_raw = data.get("is_valid_internship")
        if isinstance(valid_raw, str):
            valid = valid_raw.strip().casefold() in {"1", "true", "yes", "ya"}
        else:
            valid = bool(valid_raw)
        return cls(
            role_title=str(data.get("role_title") or "").strip(),
            company_name=str(data.get("company_name") or "").strip(),
            location=str(data.get("location") or "").strip(),
            work_mode=str(data.get("work_mode") or "").strip(),
            requirements=[
                str(item).strip() for item in requirements if str(item).strip()
            ],
            deadline=str(data.get("deadline") or "").strip(),
            apply_url=str(data.get("apply_url") or "").strip(),
            is_valid_internship=valid,
        )


@dataclass
class CurationResult:
    """Outcome of normalising one source's raw candidates."""

    jobs: list[InternshipJob] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    used_fallback: bool = False
    error: str = ""
    input_chars: int = 0


@dataclass
class SourceContent:
    """Fetched source content plus parse/failure metadata.

    ``access_status`` starts as the fetch-level status.  The engine may change
    a successful-but-empty parse to ``EMPTY_RESULT`` after curation.
    """

    source_name: str
    target_url: str
    query: str = ""
    raw_jobs: list[RawJob] = field(default_factory=list)
    raw_text: str = ""
    http_status_code: int = 0
    access_status: str = "BLOCKED"
    failure_reason: str = ""
    content_type: str = ""
    response_bytes: int = 0
    latency_ms: int = 0
    fetch_mode: str = "aiohttp"
    raw_chars_scraped: int = 0
    notes: str = ""
