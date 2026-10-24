"""LLM curator + deterministic fallback for internship candidates.

Raw job pages are noisy: navigation links end up in the anchor fallback, some
boards put several fields in one title, and a table may contain full-time jobs.
The curator's job is to enforce the eight-field contract and discard anything
that is not an internship.

When a provider is available the raw candidates are sent as compact JSON and
the LLM is asked for the exact schema.  When no provider/API key exists (or the
call fails) :func:`normalize_raw_jobs` still produces a clean, deterministic
result so the feature never becomes unusable offline.
"""
from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Mapping, Sequence

from drx.brain.llm_synthesizer import (
    LLMClient,
    LLMConfig,
    extract_json_block,
    generate_with_failover,
    resolve_providers,
)

from .models import CurationResult, InternshipJob, RawJob

log = logging.getLogger("drx.internship.curator")

MAX_LLM_CANDIDATES = 25
MAX_RAW_TEXT_CHARS = 3000
_MAX_REQUIREMENTS = 10

_EXCLUDE_TITLE = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|manager|head of|director|"
    r"full[ -]?time|fulltime|permanent|experienced|new grad|"
    r"fresh graduate|entry[ -]?level)\b",
    re.IGNORECASE,
)
_INTERN_RE = re.compile(
    r"\b(intern|interns?|internship|magang|mahasiswa\s+magang)\b",
    re.IGNORECASE,
)
_TITLE_SENIOR_RE = re.compile(
    r"\b(senior|sr\.?|lead|principal|staff|manager|head of|director)\b",
    re.IGNORECASE,
)
_DEADLINE_RE = re.compile(
    r"(\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[a-z]*\s+\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b)",
    re.IGNORECASE,
)


def _text(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", " ").split())


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "ya", "valid"}


def _looks_like_internship(role: str, description: str = "") -> bool:
    return bool(_INTERN_RE.search(f"{role} {description}"))


def _reject_senior(role: str, description: str = "") -> bool:
    # A senior/manager title is not an internship even when the page also
    # mentions an internship programme somewhere in its description.
    if _TITLE_SENIOR_RE.search(role):
        return True
    combined = f"{role} {description}".casefold()
    if not _EXCLUDE_TITLE.search(combined):
        return False
    return not _looks_like_internship(role, description)


def _normalise_work_mode(value: str, combined: str = "") -> str:
    """Normalise to the required contract: WFH / WFO / Hybrid."""
    explicit = _text(value).casefold()
    if "hybrid" in explicit:
        return "Hybrid"
    if (
        "remote" in explicit
        or "wfh" in explicit
        or "work from home" in explicit
        or "telecommute" in explicit
    ):
        return "WFH"
    if "wfo" in explicit or "on-site" in explicit or "onsite" in explicit:
        return "WFO"
    if "office" in explicit or "kantor" in explicit:
        return "WFO"
    text = f"{value} {combined}".casefold()
    if "hybrid" in text:
        return "Hybrid"
    if (
        "remote" in text
        or "wfh" in text
        or "work from home" in text
        or "telecommute" in text
    ):
        return "WFH"
    if "wfo" in text or "on-site" in text or "onsite" in text:
        return "WFO"
    if "office" in text or "kantor" in text:
        return "WFO"
    return _text(value) or "Tidak disebutkan"


def _normalise_requirements(value: object) -> list[str]:
    if isinstance(value, str):
        parts = re.split(r"[,;\n|]", value)
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = []
    out: list[str] = []
    for part in parts:
        clean = _text(part)
        if clean and clean.casefold() not in {item.casefold() for item in out}:
            out.append(clean[:120])
    return out[:_MAX_REQUIREMENTS]


def _valid_url(value: str, fallback: str = "") -> str:
    candidate = _text(value)
    if candidate.startswith(("http://", "https://")):
        return candidate
    fallback_text = _text(fallback)
    if fallback_text.startswith(("http://", "https://")):
        return fallback_text
    return ""


def _deadline_from(value: str, description: str = "") -> str:
    clean = _text(value)
    if clean:
        return clean[:80]
    match = _DEADLINE_RE.search(description or "")
    return match.group(1) if match else ""


def normalise_job(
    data: Mapping[str, object],
    *,
    fallback_url: str = "",
    fallback_company: str = "",
) -> InternshipJob | None:
    """Validate one LLM/deterministic record against the internship contract."""
    role = _text(data.get("role_title"))
    company = _text(data.get("company_name")) or _text(fallback_company)
    location = _text(data.get("location")) or "Tidak disebutkan"
    description = _text(data.get("description"))
    combined = f"{role} {description}"
    has_flag = "is_valid_internship" in data
    valid = _bool(data.get("is_valid_internship"))
    if not role:
        return None
    # Explicit LLM false wins.  Explicit true is trusted for the internship
    # decision (the LLM may normalise a title without the word "intern"), but
    # obvious senior/full-time titles are still rejected.  When the flag is
    # absent, apply the deterministic internship filter.
    if has_flag and not valid:
        return None
    if not has_flag and not _looks_like_internship(role, description):
        return None
    if _reject_senior(role, description):
        return None
    requirements = _normalise_requirements(data.get("requirements"))
    apply_url = _valid_url(_text(data.get("apply_url")), fallback_url)
    if not apply_url:
        return None
    deadline = _deadline_from(_text(data.get("deadline")), description)
    return InternshipJob(
        role_title=role[:180],
        company_name=(company or "Tidak diketahui")[:160],
        location=location[:120],
        work_mode=_normalise_work_mode(_text(data.get("work_mode")), combined)[:60],
        requirements=requirements,
        deadline=deadline,
        apply_url=apply_url,
        is_valid_internship=True,
    )


def normalize_raw_jobs(raw_jobs: Sequence[RawJob]) -> list[InternshipJob]:
    """Deterministic fallback used when the LLM is unavailable."""
    jobs: list[InternshipJob] = []
    for raw in raw_jobs:
        role = _text(raw.role_title)
        description = _text(f"{raw.description} {raw.company_name} {raw.location}")
        if not role:
            continue
        if not (raw.is_internship_hint or _looks_like_internship(role, description)):
            continue
        if _reject_senior(role, description):
            continue
        requirements = _normalise_requirements(raw.requirements)
        apply_url = _valid_url(raw.apply_url, raw.source_url)
        if not apply_url:
            continue
        jobs.append(
            InternshipJob(
                role_title=role[:180],
                company_name=(_text(raw.company_name) or "Tidak diketahui")[:160],
                location=(_text(raw.location) or "Tidak disebutkan")[:120],
                work_mode=_normalise_work_mode(
                    raw.work_mode, f"{role} {description}"
                )[:60],
                requirements=requirements,
                deadline=_deadline_from(raw.deadline, description),
                apply_url=apply_url,
                is_valid_internship=True,
            )
        )
    return _dedupe_jobs(jobs)


def _dedupe_jobs(jobs: Sequence[InternshipJob]) -> list[InternshipJob]:
    out: list[InternshipJob] = []
    seen: set[str] = set()
    for job in jobs:
        key = (
            f"{job.role_title.casefold()}|{job.company_name.casefold()}|"
            f"{job.apply_url.casefold()}"
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(job)
    return out


class InternshipCurator:
    """Normalise raw candidates for one source at a time."""

    def __init__(
        self,
        config: LLMConfig | None = None,
        *,
        client: LLMClient | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.client = client
        self.env = os.environ if env is None else env

    @property
    def available(self) -> bool:
        if self.client is not None:
            return True
        return bool(self.config.enabled and resolve_providers(self.config, self.env))

    def _llm_config(self) -> LLMConfig:
        # Keep the curator call small; the report synthesizer has its own much
        # larger output budget elsewhere.
        return self.config.model_copy(
            update={
                "max_output_tokens": min(max(self.config.max_output_tokens, 1200), 2000),
                "temperature": 0.1,
            }
        )

    async def curate(
        self,
        raw_jobs: Sequence[RawJob],
        *,
        source_name: str,
        source_url: str,
        query: str = "",
        raw_text: str = "",
    ) -> CurationResult:
        """Return valid jobs for one source; never raises."""
        fallback = normalize_raw_jobs(raw_jobs)
        result = CurationResult(
            jobs=fallback,
            used_fallback=True,
            input_chars=len(raw_text or ""),
        )
        if not self.available:
            result.error = "LLM curator tidak tersedia; memakai normalisasi deterministik"
            return result
        if not raw_jobs and not (raw_text or "").strip():
            result.error = "Tidak ada kandidat/teks untuk dikurasi"
            return result

        # Large GitHub tables can contain hundreds of rows.  Keep one LLM
        # request to respect provider token/rate limits, then normalise the
        # overflow deterministically.  This still gives the LLM a chance to
        # curate the first page of the source instead of skipping it wholesale.
        overflow_jobs: list[InternshipJob] = []
        candidate_jobs = list(raw_jobs)
        if len(candidate_jobs) > MAX_LLM_CANDIDATES:
            candidate_jobs = candidate_jobs[:MAX_LLM_CANDIDATES]
            overflow_jobs = normalize_raw_jobs(
                list(raw_jobs)[MAX_LLM_CANDIDATES:]
            )
            result.error = (
                f"Kandidat {len(raw_jobs)} > {MAX_LLM_CANDIDATES}; "
                "batch pertama dinormalisasi LLM, sisanya deterministik"
            )

        candidates = [
            {
                "role_title": job.role_title,
                "company_name": job.company_name,
                "location": job.location,
                "work_mode": job.work_mode,
                "requirements": list(job.requirements),
                "deadline": job.deadline,
                "apply_url": job.apply_url,
                "description": job.description[:300],
            }
            for job in candidate_jobs
        ]
        system = (
            "Anda adalah Internship Curator. Validasi dan normalisasi hasil "
            "mentah menjadi JSON. Buang lowongan full-time, senior, manager, "
            "atau yang bukan magang/internship. Normalisasi work_mode HANYA ke "
            "'WFH' (remote), 'WFO' (on-site) atau 'Hybrid'; pakai 'Tidak "
            "disebutkan' bila halaman tidak menyebutnya. Jangan mengarang data "
            "yang tidak ada. Jawab HANYA JSON valid."
        )
        user = f"""
Sumber: {source_name}
URL sumber: {source_url}
Kueri: {query}

Schema setiap item:
{{
  "role_title": "string",
  "company_name": "string",
  "location": "string",
  "work_mode": "string (WFH/WFO/Hybrid/Tidak disebutkan)",
  "requirements": ["string"],
  "deadline": "string",
  "apply_url": "https://...",
  "is_valid_internship": true
}}

Kandidat mentah (JSON):
{json.dumps(candidates, ensure_ascii=False)[:9000]}

Teks halaman (potongan, hanya bila kandidat tidak terstruktur):
{(raw_text or "")[:MAX_RAW_TEXT_CHARS]}

Balas dengan {{"jobs": [...]}} dan sertakan maksimal 30 item valid.
""".strip()
        try:
            specs = resolve_providers(self.config, self.env)
            text, _meta, spec = await generate_with_failover(
                specs,
                self._llm_config(),
                system,
                user,
                json_mode=True,
                client=self.client,
            )
            payload = self._parse_payload(text)
            raw_items = payload.get("jobs") if isinstance(payload, dict) else None
            if not isinstance(raw_items, list):
                raise TypeError("JSON curator tidak memiliki list 'jobs'")
            curated: list[InternshipJob] = []
            for item in raw_items:
                if not isinstance(item, dict):
                    continue
                job = normalise_job(
                    item,
                    fallback_url=source_url,
                    fallback_company=next(
                        (raw.company_name for raw in raw_jobs if raw.company_name), ""
                    ),
                )
                if job is not None:
                    curated.append(job)
            # If the model returned only false/none, fall back to the
            # deterministic parse instead of silently discarding a source that
            # clearly had internship rows.
            if not curated:
                curated = fallback
            elif overflow_jobs:
                curated = [*curated, *overflow_jobs]
            if not curated:
                result.error = "LLM curator tidak menemukan lowongan valid"
                return result
            return CurationResult(
                jobs=_dedupe_jobs(curated),
                provider=spec.name,
                model=spec.model,
                used_fallback=bool(overflow_jobs),
                input_chars=len(user),
            )
        except Exception as exc:
            result.error = f"LLM curator gagal: {exc.__class__.__name__}: {exc}"
            log.warning("internship curator fallback (%s): %s", source_name, exc)
            return result

    @staticmethod
    def _parse_payload(text: str) -> dict:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return {"jobs": parsed}
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        parsed = extract_json_block(text)
        if isinstance(parsed, list):
            return {"jobs": parsed}
        return parsed if isinstance(parsed, dict) else {}
