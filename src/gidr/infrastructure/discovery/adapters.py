"""Bounded metadata-first adapters for B06 P0 sources.

Adapters accept only a transport port. Production construction must use
BrokerSourceTransport; tests inject a deterministic transport without making
that transport eligible for a live-verification status.
"""

from __future__ import annotations

import base64
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Literal, cast
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from gidr.contracts.common import ArtifactRef, Language, ObservedValue, ValueState, new_id
from gidr.contracts.sources import (
    ArxivMetadata,
    DiscoveryProvenance,
    GdeltMetadata,
    GithubMetadata,
    SourceCandidate,
    WaybackMetadata,
)
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.discovery.compiler import compile_query
from gidr.infrastructure.discovery.contracts import (
    CanonicalQuery,
    FilterDecision,
    ProviderCapabilities,
    SourceCoverageObservation,
)
from gidr.infrastructure.discovery.registry import default_registry
from gidr.infrastructure.discovery.transport import SourceDispatchBlocked, SourceTransport


class DiscoveryError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class DiscoveryBatch:
    provider_id: str
    candidates: tuple[SourceCandidate, ...]
    next_cursor: str | None
    coverage: SourceCoverageObservation
    warnings: tuple[str, ...] = ()
    filter_decisions: tuple[FilterDecision, ...] = ()


@dataclass(frozen=True, slots=True)
class DiscoveryContext:
    run_id: str
    transport: SourceTransport
    artifacts: ArtifactStore
    secret_policy: SecretPolicy
    now: datetime
    max_pages: int = 2
    max_page_size: int = 50

    def __post_init__(self) -> None:
        if self.now.tzinfo is None or not 1 <= self.max_pages <= 5:
            raise ValueError("DISCOVERY_CONTEXT_LIMIT_INVALID")
        if not 1 <= self.max_page_size <= 100:
            raise ValueError("DISCOVERY_PAGE_SIZE_INVALID")


@dataclass(frozen=True, slots=True)
class _Item:
    title: str | None
    url: str
    raw: Mapping[str, Any]
    language: str = "unknown"
    external_ids: Mapping[str, str] | None = None
    extension: Any = None
    access_status: Literal["UNKNOWN", "PUBLIC", "RESTRICTED", "BLOCKED", "FAILED"] = "UNKNOWN"


class SourceAdapter:
    provider_id: str

    @property
    def capabilities(self) -> ProviderCapabilities:
        return default_registry().get(self.provider_id)

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        raise NotImplementedError

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        raise NotImplementedError

    async def discover(self, query: CanonicalQuery, context: DiscoveryContext) -> DiscoveryBatch:
        query = CanonicalQuery.model_validate(query)
        compiled = compile_query(query, self.capabilities)
        decisions = tuple(
            FilterDecision(
                provider_id=compiled.provider_id,
                filter_name=item.name,
                disposition=item.disposition,
                mandatory=item.mandatory,
                reason=item.reason,
            )
            for item in compiled.filters
        )
        if compiled.blocked_reason:
            return self._blocked(query, context, "MANDATORY_FILTER_UNSUPPORTED", decisions)
        candidates: list[SourceCandidate] = []
        seen: set[str] = set()
        provider_result_count = 0
        matched_candidate_count = 0
        filtered_out_count = 0
        next_cursor: str | None = None
        request_count = 0
        try:
            cursor: str | None = None
            for _ in range(context.max_pages):
                urls = self.request_urls(query, cursor=cursor, context=context)
                if not urls:
                    return self._blocked(query, context, "QUERY_CANNOT_BE_EXPRESSED", decisions)
                url = urls[0]
                request_count += 1
                # FetchBroker is synchronous today; B09's bounded worker pool must
                # invoke this adapter off the event-loop thread before enabling live I/O.
                response = context.transport.get(url, headers=self.headers())
                if response.status == 429:
                    return self._failed(
                        query, context, candidates, "RATE_LIMITED", request_count, decisions
                    )
                if response.status in {401, 403}:
                    return self._failed(
                        query,
                        context,
                        candidates,
                        "AUTH_OR_ACCESS_DENIED",
                        request_count,
                        decisions,
                    )
                if response.status < 200 or response.status >= 300:
                    return self._failed(
                        query, context, candidates, "PROVIDER_HTTP_ERROR", request_count, decisions
                    )
                items, page_cursor = self.parse(response.body, response.final_url)
                page_items = items[: min(query.max_results, context.max_page_size)]
                provider_result_count += len(page_items)
                next_cursor = page_cursor
                for item in page_items:
                    if not self._matches_local(query, item):
                        filtered_out_count += 1
                        continue
                    matched_candidate_count += 1
                    identity = _item_identity(item)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    candidates.append(self._candidate(query, context, item))
                    if len(candidates) >= query.max_results:
                        next_cursor = None
                        break
                if not page_cursor or len(candidates) >= query.max_results:
                    break
                cursor = page_cursor
            warnings = (
                ("LOCAL_POSTFILTER_ELIMINATED_RESULTS",)
                if provider_result_count and not matched_candidate_count and filtered_out_count
                else ()
            )
            return DiscoveryBatch(
                provider_id=self.provider_id,
                candidates=tuple(candidates),
                next_cursor=next_cursor,
                coverage=self._coverage(
                    query,
                    context,
                    "RESULTS" if candidates else "ZERO_RESULTS",
                    matched_candidate_count,
                    warnings[0] if warnings else None,
                    query_count=request_count,
                    provider_result_count=provider_result_count,
                    filtered_out_count=filtered_out_count,
                    unique_candidate_count=len(candidates),
                ),
                warnings=warnings,
                filter_decisions=decisions,
            )
        except SourceDispatchBlocked as exc:
            return self._failed(query, context, candidates, exc.reason, request_count, decisions)
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError, ET.ParseError):
            return self._failed(
                query, context, candidates, "PROVIDER_SCHEMA_INVALID", request_count, decisions
            )
        except DiscoveryError as exc:
            return self._failed(query, context, candidates, exc.reason, request_count, decisions)

    def headers(self) -> Mapping[str, str]:
        return {"Accept": "application/json"}

    def _matches_local(self, query: CanonicalQuery, item: _Item) -> bool:
        if query.required_terms:
            haystack = f"{item.title or ''} {json.dumps(item.raw, ensure_ascii=False)}".casefold()
            if not all(term.casefold() in haystack for term in query.required_terms):
                return False
        if query.domains:
            hostname = (urlsplit(item.url).hostname or "").casefold()
            if not any(
                hostname == d.casefold() or hostname.endswith("." + d.casefold())
                for d in query.domains
            ):
                return False
        if query.languages and item.language != "unknown" and item.language not in query.languages:
            return False
        if query.languages and item.language == "unknown":
            return False
        if query.date_from or query.date_to:
            observed_date = _item_date(item)
            if observed_date is None:
                return False
            if query.date_from and observed_date < query.date_from:
                return False
            if query.date_to and observed_date > query.date_to:
                return False
        return True

    def _candidate(
        self, query: CanonicalQuery, context: DiscoveryContext, item: _Item
    ) -> SourceCandidate:
        raw = json.dumps(
            item.raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        try:
            context.secret_policy.validate_json(raw.decode("utf-8"))
        except ValueError as exc:
            raise DiscoveryError("SECRET_ARTIFACT_BLOCKED") from exc
        artifact: ArtifactRef = context.artifacts.publish(raw)
        language = Language(
            code=item.language, status="UNKNOWN" if item.language == "unknown" else "DECLARED"
        )
        return SourceCandidate(
            run_id=context.run_id,
            created_at_utc=context.now,
            source_id=new_id("SRC"),
            source_type=self.capabilities.category,
            provider_id=self.provider_id,
            canonical_url=item.url,
            external_ids=dict(item.external_ids or {}),
            title=item.title,
            language=language,
            discovery=DiscoveryProvenance(
                question_id=query.question_id,
                query_id=query.query_id,
                method="METADATA_DISCOVERY",
            ),
            raw_record_ref=artifact,
            access_status=item.access_status,
            extension=item.extension,
        )

    def _coverage(
        self,
        query: CanonicalQuery,
        context: DiscoveryContext,
        status: Literal[
            "NOT_SELECTED", "NOT_IMPLEMENTED", "BLOCKED", "FAILED", "ZERO_RESULTS", "RESULTS"
        ],
        count: int,
        reason: str | None = None,
        query_count: int = 1,
        provider_result_count: int = 0,
        filtered_out_count: int = 0,
        unique_candidate_count: int | None = None,
    ) -> SourceCoverageObservation:
        return SourceCoverageObservation(
            run_id=context.run_id,
            question_id=query.question_id,
            provider_id=self.provider_id,
            category=self.capabilities.category,
            selected=True,
            attempted=True,
            status=status,
            query_count=query_count,
            provider_result_count=provider_result_count,
            filtered_out_count=filtered_out_count,
            candidate_count=count,
            unique_candidate_count=(
                count if unique_candidate_count is None else unique_candidate_count
            ),
            gap_codes=(reason,) if reason else (),
        )

    def _blocked(
        self,
        query: CanonicalQuery,
        context: DiscoveryContext,
        reason: str,
        decisions: tuple[FilterDecision, ...] = (),
    ) -> DiscoveryBatch:
        return DiscoveryBatch(
            self.provider_id,
            (),
            None,
            self._coverage(query, context, "BLOCKED", 0, reason, query_count=0),
            (reason,),
            decisions,
        )

    def _failed(
        self,
        query: CanonicalQuery,
        context: DiscoveryContext,
        candidates: list[SourceCandidate],
        reason: str,
        query_count: int = 1,
        decisions: tuple[FilterDecision, ...] = (),
    ) -> DiscoveryBatch:
        return DiscoveryBatch(
            self.provider_id,
            tuple(candidates),
            None,
            self._coverage(
                query, context, "FAILED", len(candidates), reason, query_count=query_count
            ),
            (reason,),
            decisions,
        )


class _JSONAdapter(SourceAdapter):
    def _json(self, body: bytes) -> Any:
        if len(body) > 2_000_000:
            raise DiscoveryError("RESPONSE_SIZE_LIMIT")
        return json.loads(body.decode("utf-8"))


class SearXNGAdapter(_JSONAdapter):
    provider_id = "searxng"

    def __init__(self, base_url: str):
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"https", "http"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("SEARXNG_BASE_URL_INVALID")
        if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("SEARXNG_HTTP_REQUIRES_LOOPBACK")
        self.base_url = base_url.rstrip("/")

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        params = {"q": query.text, "format": "json", "pageno": cursor or "1", "safesearch": "1"}
        return (self.base_url + "/search?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise TypeError("SEARXNG_SCHEMA")
        items = [
            _Item(r.get("title"), r["url"], r, _language_code(r.get("language")))
            for r in data["results"]
            if isinstance(r, dict) and isinstance(r.get("url"), str)
        ]
        return items, "2" if len(items) >= 20 else None


class GDELTAdapter(_JSONAdapter):
    provider_id = "gdelt"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        params: dict[str, str] = {
            "query": query.text,
            "mode": "ArtList",
            "format": "json",
            "maxrecords": str(min(query.max_results, context.max_page_size, 100)),
            "sort": "DateDesc",
            "timespan": "30d",
        }
        if query.date_from:
            params["startdatetime"] = query.date_from.strftime("%Y%m%d000000")
            params.pop("timespan", None)
        if query.date_to:
            params["enddatetime"] = query.date_to.strftime("%Y%m%d235959")
            params.pop("timespan", None)
        return ("https://api.gdeltproject.org/api/v2/doc/doc?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        if not isinstance(data, dict) or not isinstance(data.get("articles", []), list):
            raise TypeError("GDELT_SCHEMA")
        items = []
        for row in data.get("articles", []):
            if not isinstance(row, dict) or not isinstance(row.get("url"), str):
                continue
            seen = _parse_compact_time(row.get("seendate"))
            country = row.get("sourceCountry")
            extension = GdeltMetadata(
                seen_at_utc=seen,
                source_country=ObservedValue(state=ValueState.KNOWN, value=country)
                if country
                else ObservedValue(state=ValueState.UNKNOWN),
            )
            items.append(
                _Item(
                    row.get("title"),
                    row["url"],
                    row,
                    _language_code(row.get("language")),
                    extension=extension,
                    access_status="UNKNOWN",
                )
            )
        return items, None


class WaybackCDXAdapter(_JSONAdapter):
    provider_id = "wayback_cdx"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        target = query.text.strip()
        parsed = urlsplit(target)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            return ()
        params = {
            "url": target,
            "output": "json",
            "filter": "statuscode:200",
            "fl": "timestamp,original,statuscode,mimetype,digest",
            "collapse": "timestamp:6",
            "limit": str(min(query.max_results, context.max_page_size, 100)),
        }
        if query.date_from:
            params["from"] = query.date_from.strftime("%Y%m%d")
        if query.date_to:
            params["to"] = query.date_to.strftime("%Y%m%d")
        return ("https://web.archive.org/cdx/search/cdx?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        if not isinstance(data, list) or not data or not isinstance(data[0], list):
            raise ValueError("CDX_SCHEMA")
        header, *rows = data
        if header[:2] != ["timestamp", "original"]:
            raise ValueError("CDX_COLUMNS_UNEXPECTED")
        items = []
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) < 2
                or not re.fullmatch(r"\d{14}", str(row[0]))
            ):
                continue
            stamp, original = str(row[0]), str(row[1])
            captured = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            url = f"https://web.archive.org/web/{stamp}/{quote(original, safe=':/?&=%')}"
            raw = dict(zip(header, row, strict=False))
            extension = WaybackMetadata(
                original_url=original, capture_at_utc=captured, capture_url=url
            )
            items.append(_Item(None, url, raw, extension=extension, access_status="UNKNOWN"))
        return items, None


class ArxivAdapter(SourceAdapter):
    provider_id = "arxiv"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        params = {
            "search_query": f"all:{query.text}",
            "start": cursor or "0",
            "max_results": str(min(query.max_results, context.max_page_size)),
        }
        return ("https://export.arxiv.org/api/query?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        if b"<!DOCTYPE" in body[:4096].upper() or len(body) > 2_000_000:
            raise ValueError("ARXIV_XML_UNSAFE_OR_OVERSIZED")
        root = ET.fromstring(body)
        namespace = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
        items = []
        for node in root.findall("a:entry", namespace):
            ident = node.findtext("a:id", default="", namespaces=namespace).rsplit("/", 1)[-1]
            if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z.-]*/\d{7})(?:v\d+)?", ident):
                continue
            url = f"https://arxiv.org/abs/{ident}"
            category = node.find("a:primary_category", namespace)
            authors = tuple(
                n.text.strip() for n in node.findall("a:author/a:name", namespace) if n.text
            )
            extension = ArxivMetadata(
                arxiv_id=ident,
                version=None,
                doi=ObservedValue(state=ValueState.UNKNOWN),
                authors=authors,
                primary_category=category.attrib.get("term") if category is not None else None,
            )
            raw = {
                "id": ident,
                "title": node.findtext("a:title", default="", namespaces=namespace).strip(),
                "published": node.findtext("a:published", default="", namespaces=namespace),
                "authors": authors,
                "category": extension.primary_category,
            }
            title = raw["title"] if isinstance(raw["title"], str) else None
            items.append(_Item(title, url, raw, extension=extension, external_ids={"arxiv": ident}))
        start = (
            int(urlsplit(final_url).query.split("start=")[-1].split("&")[0])
            if "start=" in final_url
            else 0
        )
        request_values = dict(
            part.split("=", 1) for part in urlsplit(final_url).query.split("&") if "=" in part
        )
        per_page = int(request_values.get("max_results", "10"))
        total = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
        next_cursor = (
            str(start + per_page)
            if total is not None and start + per_page < int(total.text or "0")
            else None
        )
        return items, next_cursor


class OpenAlexAdapter(_JSONAdapter):
    provider_id = "openalex"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        params = {
            "search": query.text,
            "per-page": str(min(query.max_results, context.max_page_size)),
            "cursor": cursor or "*",
        }
        filters = []
        if query.date_from:
            filters.append("from_publication_date:" + query.date_from.isoformat())
        if query.date_to:
            filters.append("to_publication_date:" + query.date_to.isoformat())
        if filters:
            params["filter"] = ",".join(filters)
        return ("https://api.openalex.org/works?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise TypeError("OPENALEX_SCHEMA")
        items = []
        for row in data["results"]:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                continue
            title = row.get("display_name")
            doi = row.get("doi")
            language = (
                row.get("language")
                if isinstance(row.get("language"), str)
                and re.fullmatch(r"[a-z]{2,3}", row["language"])
                else "unknown"
            )
            items.append(
                _Item(
                    title,
                    row["id"],
                    row,
                    cast(str, language),
                    {"doi": doi} if doi else {"openalex": row["id"].rsplit("/", 1)[-1]},
                )
            )
        cursor = (
            data.get("meta", {}).get("next_cursor") if isinstance(data.get("meta"), dict) else None
        )
        return items, cursor if isinstance(cursor, str) and cursor else None


class CrossrefAdapter(_JSONAdapter):
    provider_id = "crossref"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        params = {
            "query.bibliographic": query.text,
            "rows": str(min(query.max_results, context.max_page_size)),
            "offset": cursor or "0",
        }
        date_filters = []
        if query.date_from:
            date_filters.append("from-pub-date:" + query.date_from.isoformat())
        if query.date_to:
            date_filters.append("until-pub-date:" + query.date_to.isoformat())
        if date_filters:
            params["filter"] = ",".join(date_filters)
        return ("https://api.crossref.org/works?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        rows = (
            data.get("message", {}).get("items")
            if isinstance(data, dict) and isinstance(data.get("message"), dict)
            else None
        )
        if not isinstance(rows, list):
            raise TypeError("CROSSREF_SCHEMA")
        items = []
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("DOI"), str):
                continue
            doi = row["DOI"]
            title = row.get("title", [None])[0] if isinstance(row.get("title"), list) else None
            url = row.get("URL", "https://doi.org/" + quote(doi, safe="/()"))
            language = row.get("language")
            language_code = (
                language.lower()
                if isinstance(language, str) and re.fullmatch(r"[A-Za-z]{2,3}", language)
                else "unknown"
            )
            items.append(_Item(title, url, row, language_code, external_ids={"doi": doi}))
        query = dict(
            part.split("=", 1) for part in urlsplit(final_url).query.split("&") if "=" in part
        )
        offset = int(query.get("offset", "0"))
        total = data.get("message", {}).get("total-results", 0)
        next_cursor = str(offset + len(rows)) if rows and offset + len(rows) < total else None
        return items, next_cursor


class ZenodoAdapter(_JSONAdapter):
    provider_id = "zenodo"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        params = {
            "q": query.text,
            "size": str(min(query.max_results, context.max_page_size)),
            "page": cursor or "1",
            "sort": "mostrecent",
        }
        return ("https://zenodo.org/api/records?" + urlencode(params),)

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        hits = (
            data.get("hits", {}).get("hits")
            if isinstance(data, dict) and isinstance(data.get("hits"), dict)
            else None
        )
        if not isinstance(hits, list):
            raise TypeError("ZENODO_SCHEMA")
        items = []
        for row in hits:
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata", {})
            url = row.get("links", {}).get("html") or row.get("doi")
            if not isinstance(url, str):
                continue
            doi = row.get("doi")
            raw = dict(row)
            raw["publication_date"] = metadata.get("publication_date")
            languages = metadata.get("languages")
            language_code = (
                languages[0].lower()
                if isinstance(languages, list)
                and languages
                and isinstance(languages[0], str)
                and re.fullmatch(r"[A-Za-z]{2,3}", languages[0])
                else "unknown"
            )
            items.append(
                _Item(
                    metadata.get("title"),
                    url,
                    raw,
                    language_code,
                    external_ids={"doi": doi} if doi else {"zenodo": str(row.get("id"))},
                )
            )
        links = data.get("links", {})
        next_page = links.get("next") if isinstance(links, dict) else None
        return items, str(
            int(urlsplit(next_page).query.split("page=")[-1].split("&")[0])
        ) if isinstance(next_page, str) and "page=" in next_page else None


class GitHubAdapter(_JSONAdapter):
    provider_id = "github"

    def request_urls(
        self, query: CanonicalQuery, *, cursor: str | None, context: DiscoveryContext
    ) -> tuple[str, ...]:
        page = cursor or "1"
        params = {
            "q": query.text,
            "per_page": str(min(query.max_results, context.max_page_size)),
            "page": page,
            "sort": "updated",
            "order": "desc",
        }
        if query.date_from or query.date_to:
            lo = query.date_from.isoformat() if query.date_from else "*"
            hi = query.date_to.isoformat() if query.date_to else "*"
            params["q"] += f" pushed:{lo}..{hi}"
        return ("https://api.github.com/search/repositories?" + urlencode(params),)

    def headers(self) -> Mapping[str, str]:
        return {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}

    def parse(self, body: bytes, final_url: str) -> tuple[list[_Item], str | None]:
        data = self._json(body)
        rows = data.get("items") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise TypeError("GITHUB_SEARCH_SCHEMA")
        items = []
        for row in rows:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("full_name"), str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["full_name"])
                or not isinstance(row.get("html_url"), str)
            ):
                continue
            repo = row["full_name"]
            raw = {
                "full_name": repo,
                "html_url": row["html_url"],
                "description": row.get("description"),
                "default_branch": row.get("default_branch"),
                "pushed_at": row.get("pushed_at"),
                "id": row.get("id"),
            }
            ext = GithubMetadata(
                repository=repo, ref=row.get("default_branch"), inspection="REFERENCE_ONLY"
            )
            items.append(
                _Item(
                    repo,
                    f"https://github.com/{repo}",
                    raw,
                    extension=ext,
                    external_ids={"github_repo": str(row.get("id", repo))},
                )
            )
        page = (
            int(urlsplit(final_url).query.split("page=")[-1].split("&")[0])
            if "page=" in final_url
            else 1
        )
        total = data.get("total_count", 0)
        per_page = int(
            dict(
                part.split("=", 1) for part in urlsplit(final_url).query.split("&") if "=" in part
            ).get("per_page", "30")
        )
        return items, str(page + 1) if rows and page * per_page < total else None

    async def discover(self, query: CanonicalQuery, context: DiscoveryContext) -> DiscoveryBatch:
        batch = await super().discover(query, context)
        if not batch.candidates:
            return batch
        inspected: list[SourceCandidate] = []
        warnings = list(batch.warnings)
        readme_requests = 0
        for candidate in batch.candidates:
            repository = (
                candidate.extension.repository
                if isinstance(candidate.extension, GithubMetadata)
                else ""
            )
            readme_url = f"https://api.github.com/repos/{repository}/readme"
            try:
                readme_requests += 1
                response = context.transport.get(readme_url, headers=self.headers())
                if response.status == 404:
                    warnings.append("README_NOT_FOUND")
                    inspected.append(candidate)
                    continue
                if response.status in {401, 403, 429} or not 200 <= response.status < 300:
                    warnings.append("README_ACCESS_UNAVAILABLE")
                    inspected.append(candidate)
                    continue
                doc = self._json(response.body)
                if (
                    not isinstance(doc, dict)
                    or doc.get("encoding") != "base64"
                    or not isinstance(doc.get("content"), str)
                ):
                    warnings.append("README_SCHEMA_INVALID")
                    inspected.append(candidate)
                    continue
                readme = base64.b64decode(doc["content"], validate=True)
                if len(readme) > 256_000:
                    warnings.append("README_SIZE_LIMIT")
                    inspected.append(candidate)
                    continue
                original = context.artifacts.read(candidate.raw_record_ref)
                combined = json.loads(original)
                combined["readme_text"] = readme.decode("utf-8", errors="replace")
                combined["readme_url"] = doc.get("html_url")
                combined_raw = json.dumps(combined, ensure_ascii=False, sort_keys=True).encode()
                context.secret_policy.validate_json(combined_raw.decode("utf-8"))
                artifact = context.artifacts.publish(combined_raw)
                assert isinstance(candidate.extension, GithubMetadata)
                updated = GithubMetadata(
                    repository=candidate.extension.repository,
                    ref=candidate.extension.ref,
                    commit_sha=None,
                    inspection="README",
                )
                inspected.append(
                    candidate.model_copy(update={"extension": updated, "raw_record_ref": artifact})
                )
            except SourceDispatchBlocked as exc:
                warnings.append(exc.reason)
                inspected.append(candidate)
            except (ValueError, UnicodeError, json.JSONDecodeError):
                warnings.append("README_INVALID")
                inspected.append(candidate)
        coverage = batch.coverage.model_copy(
            update={
                "query_count": batch.coverage.query_count + readme_requests,
                "gap_codes": tuple(dict.fromkeys((*batch.coverage.gap_codes, *warnings))),
            }
        )
        return DiscoveryBatch(
            batch.provider_id,
            tuple(inspected),
            batch.next_cursor,
            coverage,
            tuple(dict.fromkeys(warnings)),
            batch.filter_decisions,
        )


def _parse_compact_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{14}", value):
        return None
    return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=UTC)


def _item_date(item: _Item) -> date | None:
    candidates: list[object] = [
        item.raw.get("published"),
        item.raw.get("publication_date"),
        item.raw.get("published_at"),
        item.raw.get("pushed_at"),
        item.raw.get("created_at"),
        item.raw.get("datePublished"),
    ]
    published = item.raw.get("published-print") or item.raw.get("published-online")
    if isinstance(published, dict):
        parts = published.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list):
            try:
                parts_list = [int(part) for part in parts[0]]
                if len(parts_list) >= 3:
                    return date(*parts_list[:3])
                if len(parts_list) < 3:
                    return None
            except (TypeError, ValueError):
                return None
    if item.extension is not None and isinstance(item.extension, WaybackMetadata):
        return cast(date, item.extension.capture_at_utc.date())
    if item.extension is not None and isinstance(item.extension, GdeltMetadata):
        return item.extension.seen_at_utc.date() if item.extension.seen_at_utc else None
    for value in candidates:
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                continue
    metadata = item.raw.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("publication_date"), str):
        try:
            return date.fromisoformat(metadata["publication_date"][:10])
        except ValueError:
            pass
    return None


def _language_code(value: object) -> str:
    if (
        isinstance(value, str)
        and re.fullmatch(r"[A-Za-z]{2,3}", value)
        and value.casefold() != "all"
    ):
        return value.lower()
    return "unknown"


def _item_identity(item: _Item) -> str:
    for key, value in (item.external_ids or {}).items():
        if key.casefold() == "doi" and value:
            normalized = value.casefold().removeprefix("https://doi.org/")
            return "doi:" + normalized
    parsed = urlsplit(item.url)
    hostname = (parsed.hostname or "").casefold().rstrip(".")
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    port = parsed.port
    if port is not None and not (
        (parsed.scheme == "https" and port == 443) or (parsed.scheme == "http" and port == 80)
    ):
        hostname = f"{hostname}:{port}"
    return urlunsplit((parsed.scheme.casefold(), hostname, parsed.path or "/", parsed.query, ""))


P0_ADAPTERS: Mapping[str, type[SourceAdapter]] = {
    "searxng": SearXNGAdapter,
    "gdelt": GDELTAdapter,
    "wayback_cdx": WaybackCDXAdapter,
    "arxiv": ArxivAdapter,
    "openalex": OpenAlexAdapter,
    "crossref": CrossrefAdapter,
    "zenodo": ZenodoAdapter,
    "github": GitHubAdapter,
}
