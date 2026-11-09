"""Stable source identity normalization; identity never merges evidence records."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import unquote, urlsplit

from gidr.infrastructure.network.retrieval import canonicalize_url

_DOI = re.compile(r"^10\.\d{4,9}/[-._;()/:A-Z0-9]+$", re.IGNORECASE)
_WAYBACK = re.compile(r"^/web/(\d{4,14})(?:[a-z_]{1,8})?/(https?://.+)$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class NormalizedIdentity:
    kind: str
    value: str
    snapshot_parent_url: str | None = None


def normalize_doi(value: str) -> str:
    candidate = value.strip()
    if candidate.casefold().startswith("https://doi.org/"):
        candidate = candidate[len("https://doi.org/") :]
    elif candidate.casefold().startswith("http://doi.org/"):
        candidate = candidate[len("http://doi.org/") :]
    elif candidate.casefold().startswith("doi:"):
        candidate = candidate[4:]
    candidate = unquote(candidate).strip().rstrip(".,; ").casefold()
    if len(candidate) > 512 or not _DOI.fullmatch(candidate):
        raise ValueError("DOI_IDENTITY_INVALID")
    return candidate


def normalize_identity(value: str) -> NormalizedIdentity:
    if len(value) > 8192 or any(ord(char) < 33 or ord(char) == 127 for char in value):
        raise ValueError("SOURCE_IDENTITY_INVALID")
    if value.casefold().startswith("doi:"):
        return NormalizedIdentity("DOI", normalize_doi(value))
    parts = urlsplit(value)
    host = (parts.hostname or "").casefold().rstrip(".")
    if host in {"doi.org", "www.doi.org"}:
        return NormalizedIdentity("DOI", normalize_doi(unquote(parts.path.lstrip("/"))))
    if host == "web.archive.org":
        match = _WAYBACK.fullmatch(parts.path)
        if match is None:
            raise ValueError("WAYBACK_IDENTITY_INVALID")
        timestamp, original = match.groups()
        parent_url = canonicalize_url(original)
        return NormalizedIdentity("WAYBACK_SNAPSHOT", f"{timestamp}:{parent_url}", parent_url)
    return NormalizedIdentity("URL", canonicalize_url(value))
