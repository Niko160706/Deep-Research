"""LLM query planner for the three Internship Hunter input modes.

The general-investigation planner in ``drx.brain.query_planner`` is tuned for
OSINT entities and is intentionally not reused here.  This planner has three
job-specific strategies:

``role``
    Expand one role title into related internship roles/queries.
``discovery``
    Search broadly across tech internships, optionally constrained by location
    or work mode (stored as metadata; never merged into queries).
``skill``
    Map a skill list to the three most suitable internship roles and queries.

PENTING (required contract): every role and every query is a *pure role*
title like ``"AI Engineer"`` — never ``"AI Engineer Intern"``, never
``"AI Engineer Intern Indonesia"``.  Location and work-mode are kept as plan
metadata only; the URL builders in ``site_parsers.py`` already encode the
internship route plus percent-encode/hyphenate the query themselves.

Every LLM call is best-effort.  Missing keys, provider errors or malformed JSON
fall back to a deterministic plan so ``--no-llm`` and offline smoke runs still
produce useful searches.
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

from .models import Approach, InternshipPlan

log = logging.getLogger("drx.internship.planner")

MAX_QUERIES = 3

#: Deterministic pure-role expansions used when the LLM is unavailable.
ROLE_SYNONYMS: dict[str, list[str]] = {
    "ai": ["Machine Learning", "Data Science", "AI Research"],
    "machine learning": ["AI Engineer", "Data Science", "MLOps"],
    "data": ["Data Analyst", "Data Science", "Business Intelligence"],
    "backend": ["Software Engineer", "API Engineer", "Platform Engineer"],
    "frontend": ["Web Developer", "React Developer", "UI Engineer"],
    "software": ["Backend Engineer", "Frontend Engineer", "QA Engineer"],
    "devops": ["Cloud Engineer", "Site Reliability", "Platform Engineer"],
    "security": ["Cybersecurity", "Security Analyst", "Penetration Tester"],
    "mobile": ["Android Developer", "iOS Developer", "Mobile Engineer"],
    "product": ["Product Manager", "Product Analyst", "UX Research"],
}

SKILL_ROLE_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("python", "pytorch", "tensorflow", "ml", "machine learning", "ai"), "AI/ML Engineer"),
    (("sql", "tableau", "power bi", "analytics", "data"), "Data Analyst"),
    (
        ("go", "golang", "java", "node", "backend", "api", "django", "flask", "fastapi"),
        "Backend Engineer",
    ),
    (
        (
            "react",
            "vue",
            "angular",
            "javascript",
            "typescript",
            "frontend",
            "html",
            "css",
        ),
        "Frontend Engineer",
    ),
    (("docker", "kubernetes", "aws", "gcp", "azure", "devops"), "DevOps Engineer"),
    (("android", "ios", "flutter", "kotlin", "swift"), "Mobile Engineer"),
    (("security", "pentest", "cybersecurity"), "Cybersecurity"),
    (("figma", "ux", "ui", "design"), "Product Designer"),
)

GENERIC_ROLES = ["Software Engineer", "Data Analyst", "AI Engineer"]

DEFAULT_ROLES = list(GENERIC_ROLES)

#: Words that must never be appended to a planner role/query.
_EMBELLISHMENT_RE = re.compile(
    r"[\s\--/]*(?:\binternship\b|\bmagang\b|\bintern(?:s)?\b|\bjobs?\b)"
    r"\W*(?:\bin\b)?\W*(?:indonesia|remote|onsite|on-site|hybrid|wfh|wfo)?"
    r"\s*$",
    re.IGNORECASE,
)


def _clean_terms(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = re.split(r"[,;\n|]", value)
    else:
        parts = list(value)
    terms: list[str] = []
    for part in parts:
        clean = " ".join(str(part or "").strip().split())
        if clean and clean.casefold() not in {t.casefold() for t in terms}:
            terms.append(clean)
    return terms


def _pure_role(value: str) -> str:
    """Strip internship/location/work-mode embellishments from a role title."""
    cleaned = " ".join(str(value or "").strip().split())
    while True:
        stripped = _EMBELLISHMENT_RE.sub("", cleaned).strip(" -–—/\t")
        stripped = " ".join(stripped.split())
        if not stripped or stripped == cleaned:
            return cleaned
        cleaned = stripped


def _dedupe_queries(queries: Sequence[str], *, limit: int = MAX_QUERIES) -> list[str]:
    out: list[str] = []
    for query in queries:
        clean = _pure_role(query)
        if clean and clean.casefold() not in {q.casefold() for q in out}:
            out.append(clean)
    return out[:limit]


def fallback_plan(
    approach: Approach,
    *,
    role: str = "",
    skills: str | Sequence[str] | None = None,
    location: str = "Indonesia OR Remote",
    work_mode: str = "",
) -> InternshipPlan:
    """Build a useful deterministic plan without calling any provider.

    ``roles`` and ``queries`` stay pure; ``location``/``work_mode`` are plan
    metadata.  The URL builders add the intern filter themselves.
    """
    if approach == "role":
        base_role = _pure_role(role or "Software Engineer")
        synonyms: list[str] = []
        lowered = base_role.casefold()
        for key, names in ROLE_SYNONYMS.items():
            if key in lowered:
                synonyms.extend(names)
        if not synonyms:
            synonyms = []
        roles = _dedupe_queries([base_role] + synonyms)
        for generic in GENERIC_ROLES:
            if len(roles) >= MAX_QUERIES:
                break
            if generic.casefold() not in {r.casefold() for r in roles}:
                roles.append(generic)
        roles = roles[:MAX_QUERIES]
        queries = list(roles)[:MAX_QUERIES]
        return InternshipPlan(
            approach=approach,
            user_input=base_role,
            roles=roles,
            queries=queries,
            location=location,
            work_mode=work_mode,
            rationale="Ekspansi deterministik dari role dan sinonimnya (role murni).",
            used_fallback=True,
        )

    if approach == "skill":
        terms = _clean_terms(skills)
        matches: list[str] = []
        for keywords, mapped_role in SKILL_ROLE_MAP:
            if any(
                any(
                    re.search(
                        rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])",
                        term.casefold(),
                    )
                    for keyword in keywords
                )
                for term in terms
            ):
                if mapped_role not in matches:
                    matches.append(mapped_role)
        for role_name in DEFAULT_ROLES:
            if len(matches) >= MAX_QUERIES:
                break
            if role_name not in matches:
                matches.append(role_name)
        roles = _dedupe_queries(matches)[:MAX_QUERIES]
        for generic in DEFAULT_ROLES:
            if len(roles) >= MAX_QUERIES:
                break
            if generic.casefold() not in {r.casefold() for r in roles}:
                roles.append(generic)
        roles = roles[:MAX_QUERIES]
        queries = list(roles)[:MAX_QUERIES]
        return InternshipPlan(
            approach=approach,
            user_input=", ".join(terms),
            roles=roles,
            queries=queries,
            location=location,
            work_mode=work_mode,
            rationale="Pemetaan skill deterministik ke tiga peran magang.",
            used_fallback=True,
        )

    roles = _dedupe_queries(GENERIC_ROLES)[:MAX_QUERIES]
    return InternshipPlan(
        approach="discovery",
        user_input="",
        roles=roles,
        queries=list(roles),
        location=location,
        work_mode=work_mode,
        rationale="Pencarian luas seluruh magang tech pada lokasi/moda yang dipilih.",
        used_fallback=True,
    )


def _sanitize_plan(
    data: dict,
    base: InternshipPlan,
) -> InternshipPlan:
    """Merge LLM output into a fallback plan, keeping it small, safe and pure."""
    roles = data.get("roles") or []
    queries = data.get("queries") or data.get("search_queries") or []
    if isinstance(roles, str):
        roles = _clean_terms(roles)
    if isinstance(queries, str):
        queries = _clean_terms(queries)
    roles = _clean_terms(roles if isinstance(roles, list) else [])
    queries = _clean_terms(queries if isinstance(queries, list) else [])
    roles = [_pure_role(role)[:100] for role in roles][:MAX_QUERIES]
    queries = [_pure_role(query)[:100] for query in queries][:MAX_QUERIES]
    if not roles:
        roles = base.roles
    if not queries:
        queries = base.queries
    # Keep the three-role contract even if a provider returns only one or two
    # roles for skill-based matching.
    for candidate in base.roles:
        if len(roles) >= MAX_QUERIES:
            break
        if candidate.casefold() not in {role.casefold() for role in roles}:
            roles.append(candidate)
    for candidate in base.queries:
        if len(queries) >= MAX_QUERIES:
            break
        if candidate.casefold() not in {query.casefold() for query in queries}:
            queries.append(candidate)
    # Contract: queries are pure role names aligned with roles.
    queries = _dedupe_queries(queries)[:MAX_QUERIES]
    roles = _dedupe_queries(roles)[:MAX_QUERIES]
    return InternshipPlan(
        approach=base.approach,
        user_input=base.user_input,
        roles=roles,
        queries=queries,
        location=base.location,
        work_mode=base.work_mode,
        rationale=str(data.get("rationale") or "").strip()[:400] or base.rationale,
        used_fallback=False,
    )


def _prompt_for(
    approach: Approach,
    *,
    role: str,
    skills: Sequence[str],
    location: str,
    work_mode: str,
) -> tuple[str, str]:
    system = (
        "Anda adalah Internship Query Planner. Tugas Anda hanya merancang "
        "kueri pencarian magang tech yang spesifik dan dapat dieksekusi. "
        "Jangan mengarang lowongan, perusahaan, atau URL. Jawab HANYA JSON valid."
    )
    contract = (
        "KONTRAK WAJIB: setiap role dan setiap query harus berupa NAMA PERAN "
        "MURNI tanpa kata Intern/Magang/Internship dan TANPA lokasi/moda kerja. "
        "Contoh BAIK: 'AI Engineer'. Contoh BURUK: 'AI Engineer Intern Indonesia'."
    )
    if approach == "role":
        user = f"""
Input pengguna: peran spesifik "{role}".
{contract}
Lokasi/moda metadata: "{location}" {('(' + work_mode + ')') if work_mode else ''}.
Buat 3 peran magang paling relevan (termasuk sinonim/peran terkait). Kueri = peran.

Balas dengan skema:
{{"roles": ["..."], "queries": ["..."], "rationale": "satu kalimat"}}
""".strip()
    elif approach == "skill":
        user = f"""
Daftar skill pengguna: {json.dumps(list(skills), ensure_ascii=False)}.
{contract}
Lokasi/moda metadata: "{location}" {('(' + work_mode + ')') if work_mode else ''}.
Petakan skill tersebut ke 3 peran magang tech paling cocok. Kueri = peran. Jangan membuat lebih dari 3.

Balas dengan skema:
{{"roles": ["..."], "queries": ["..."], "rationale": "satu kalimat"}}
""".strip()
    else:
        user = f"""
Cari semua jenis magang tech tanpa batasan spesialisasi.
{contract}
Lokasi/moda metadata: "{location}" {('(' + work_mode + ')') if work_mode else ''}.
Buat 3 peran tech yang representatif untuk pencarian magang (peran murni). Kueri = peran.

Balas dengan skema:
{{"roles": ["..."], "queries": ["..."], "rationale": "satu kalimat"}}
""".strip()
    return system, user


async def plan_internship(
    approach: Approach,
    *,
    role: str = "",
    skills: str | Sequence[str] | None = None,
    location: str = "Indonesia OR Remote",
    work_mode: str = "",
    config: LLMConfig | None = None,
    client: LLMClient | None = None,
    env: Mapping[str, str] | None = None,
) -> InternshipPlan:
    """Plan queries with the LLM when available, otherwise deterministically."""
    skill_terms = _clean_terms(skills)
    fallback = fallback_plan(
        approach,
        role=role,
        skills=skill_terms,
        location=location,
        work_mode=work_mode,
    )
    config = config or LLMConfig()
    if not config.enabled and client is None:
        return fallback
    env_map = os.environ if env is None else env
    specs = resolve_providers(config, env_map)
    if not specs and client is None:
        fallback.error = "LLM planner dilewati: provider/API key tidak tersedia"
        return fallback

    system, user = _prompt_for(
        approach,
        role=role,
        skills=skill_terms,
        location=location,
        work_mode=work_mode,
    )
    try:
        _text, _meta, spec = await generate_with_failover(
            specs,
            config,
            system,
            user,
            json_mode=True,
            client=client,
        )
        payload = extract_json_block(_text)
        if not isinstance(payload, dict):
            raise TypeError("LLM planner tidak mengembalikan objek JSON")
        plan = _sanitize_plan(payload, fallback)
        plan.provider = spec.name
        plan.model = spec.model
        if not plan.queries:
            raise ValueError("LLM planner mengembalikan queries kosong")
        return plan
    except Exception as exc:
        fallback.error = f"LLM planner gagal: {exc.__class__.__name__}: {exc}"
        log.warning("internship planner fallback: %s", exc)
        return fallback
