import asyncio
import base64
import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from gidr.contracts.sources import ArxivMetadata, GithubMetadata, WaybackMetadata
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.audit.policy import SecretPolicy
from gidr.infrastructure.discovery.adapters import (
    ArxivAdapter,
    CrossrefAdapter,
    DiscoveryContext,
    GDELTAdapter,
    GitHubAdapter,
    OpenAlexAdapter,
    SearXNGAdapter,
    WaybackCDXAdapter,
    ZenodoAdapter,
)
from gidr.infrastructure.discovery.contracts import CanonicalQuery
from gidr.infrastructure.discovery.transport import BrokerSourceTransport, SourceDispatchBlocked


class MockTransport:
    def __init__(self, *responses: tuple[int, bytes, str]):
        self.responses = list(responses)
        self.urls: list[str] = []

    def get(self, url: str, *, headers=None):
        self.urls.append(url)
        status, body, final_url = self.responses.pop(0)
        return SimpleNamespace(status=status, body=body, final_url=final_url, headers={})


def make_query(text: str = "water research", **overrides: object) -> CanonicalQuery:
    fields: dict[str, object] = {
        "query_id": "QUERY-test",
        "question_id": "RQ-test",
        "text": text,
        "max_results": 5,
    }
    fields.update(overrides)
    return CanonicalQuery.model_validate(fields)


def context(tmp_path: Path, transport: MockTransport, *, max_pages: int = 2) -> DiscoveryContext:
    return DiscoveryContext(
        run_id="RUN-test",
        transport=transport,
        artifacts=ArtifactStore(tmp_path / "artifacts", max_bytes=1_000_000),
        secret_policy=SecretPolicy(),
        now=datetime(2026, 1, 1, tzinfo=UTC),
        max_pages=max_pages,
        max_page_size=10,
    )


def response(body: object, url: str, status: int = 200) -> tuple[int, bytes, str]:
    payload = body if isinstance(body, bytes) else json.dumps(body).encode()
    return status, payload, url


def discover(adapter, query: CanonicalQuery, ctx: DiscoveryContext):
    return asyncio.run(adapter.discover(query, ctx))


def test_gdelt_normalizes_provenance_and_candidate_artifact(tmp_path: Path) -> None:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    mock = MockTransport(
        response(
            {
                "articles": [
                    {
                        "url": "https://news.example/a",
                        "title": "Water study",
                        "seendate": "20250102030405",
                        "sourceCountry": "ID",
                    },
                    {
                        "url": "https://news.example/a",
                        "title": "Water study duplicate",
                        "seendate": "20250102030405",
                        "sourceCountry": "ID",
                    },
                ]
            },
            url,
        )
    )
    batch = discover(GDELTAdapter(), make_query(), context(tmp_path, mock))
    assert batch.coverage.status == "RESULTS"
    assert batch.coverage.run_id == "RUN-test"
    assert batch.coverage.provider_result_count == 2
    assert batch.coverage.candidate_count == 2
    assert batch.coverage.unique_candidate_count == 1
    assert len(batch.candidates) == 1
    candidate = batch.candidates[0]
    assert candidate.provider_id == "gdelt"
    assert candidate.discovery.query_id == "QUERY-test"
    assert candidate.extension.kind == "gdelt"
    assert candidate.raw_record_ref.sha256


def test_unsupported_mandatory_filter_blocks_before_transport(tmp_path: Path) -> None:
    mock = MockTransport()
    q = make_query(mandatory_filters=("language",), languages=("en",))

    class UnsupportedLanguageGDELT(GDELTAdapter):
        @property
        def capabilities(self):
            return GDELTAdapter().capabilities.model_copy(update={"local_filters": ()})

    batch = discover(UnsupportedLanguageGDELT(), q, context(tmp_path, mock))
    assert batch.coverage.status == "BLOCKED"
    assert batch.warnings == ("MANDATORY_FILTER_UNSUPPORTED",)
    assert batch.filter_decisions[0].disposition == "NATIVE"
    assert batch.filter_decisions[-1].disposition == "UNSUPPORTED"
    assert mock.urls == []


@pytest.mark.parametrize("status,reason", [(429, "RATE_LIMITED"), (403, "AUTH_OR_ACCESS_DENIED")])
def test_rate_limit_and_auth_are_explicit(tmp_path: Path, status: int, reason: str) -> None:
    mock = MockTransport((status, b"{}", "https://api.gdeltproject.org/api/v2/doc/doc"))
    batch = discover(GDELTAdapter(), make_query(), context(tmp_path, mock))
    assert batch.coverage.status == "FAILED"
    assert batch.warnings == (reason,)


def test_schema_drift_and_empty_results_are_distinct(tmp_path: Path) -> None:
    endpoint = "https://api.openalex.org/works"
    malformed = MockTransport(response({"unexpected": []}, endpoint))
    assert discover(OpenAlexAdapter(), make_query(), context(tmp_path, malformed)).warnings == (
        "PROVIDER_SCHEMA_INVALID",
    )
    empty = MockTransport(response({"results": [], "meta": {}}, endpoint))
    result = discover(OpenAlexAdapter(), make_query(), context(tmp_path, empty))
    assert result.coverage.status == "ZERO_RESULTS"


def test_openalex_cursor_pagination_is_bounded(tmp_path: Path) -> None:
    endpoint = "https://api.openalex.org/works"
    first = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "display_name": "water",
                "publication_date": "2024-01-01",
            }
        ],
        "meta": {"next_cursor": "cursor-2"},
    }
    second = {
        "results": [
            {
                "id": "https://openalex.org/W2",
                "display_name": "water",
                "publication_date": "2024-01-02",
            }
        ],
        "meta": {"next_cursor": "cursor-3"},
    }
    mock = MockTransport(response(first, endpoint), response(second, endpoint))
    batch = discover(OpenAlexAdapter(), make_query(), context(tmp_path, mock, max_pages=2))
    assert len(batch.candidates) == 2
    assert batch.next_cursor == "cursor-3"
    assert "cursor=cursor-2" in mock.urls[1]


def test_arxiv_atom_schema_and_typed_extension(tmp_path: Path) -> None:
    xml = b"""<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom" xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"><opensearch:totalResults>1</opensearch:totalResults><entry><id>http://arxiv.org/abs/2401.12345v2</id><title>Water research</title><published>2024-01-02T00:00:00Z</published><author><name>Example Author</name></author><primary_category term="cs.AI"/></entry></feed>"""
    endpoint = "https://export.arxiv.org/api/query?start=0"
    batch = discover(
        ArxivAdapter(), make_query(), context(tmp_path, MockTransport(response(xml, endpoint)))
    )
    assert len(batch.candidates) == 1
    assert isinstance(batch.candidates[0].extension, ArxivMetadata)
    assert batch.candidates[0].extension.arxiv_id == "2401.12345v2"


def test_arxiv_rejects_doctype(tmp_path: Path) -> None:
    endpoint = "https://export.arxiv.org/api/query"
    body = b'<!DOCTYPE feed [<!ENTITY x "bad">]><feed xmlns="http://www.w3.org/2005/Atom" />'
    batch = discover(
        ArxivAdapter(), make_query(), context(tmp_path, MockTransport(response(body, endpoint)))
    )
    assert batch.warnings == ("PROVIDER_SCHEMA_INVALID",)


def test_cdx_preserves_exact_original_url_and_timestamp(tmp_path: Path) -> None:
    endpoint = "https://web.archive.org/cdx/search/cdx"
    rows = [
        ["timestamp", "original", "statuscode", "mimetype", "digest"],
        ["20240203040506", "https://example.org/a", "200", "text/html", "abc"],
    ]
    batch = discover(
        WaybackCDXAdapter(),
        make_query("https://example.org/a"),
        context(tmp_path, MockTransport(response(rows, endpoint))),
    )
    candidate = batch.candidates[0]
    assert isinstance(candidate.extension, WaybackMetadata)
    assert candidate.extension.original_url == "https://example.org/a"
    assert (
        candidate.canonical_url
        == "https://web.archive.org/web/20240203040506/https://example.org/a"
    )


def test_cdx_rejects_non_url_query_without_network(tmp_path: Path) -> None:
    mock = MockTransport()
    batch = discover(WaybackCDXAdapter(), make_query("file:///etc/passwd"), context(tmp_path, mock))
    assert batch.coverage.status == "BLOCKED"
    assert batch.coverage.query_count == 0
    assert mock.urls == []


def test_github_reads_readme_without_clone_and_records_ref(tmp_path: Path) -> None:
    search_url = "https://api.github.com/search/repositories"
    readme_url = "https://api.github.com/repos/octo/water/readme"
    repo = {
        "total_count": 1,
        "items": [
            {
                "id": 123,
                "full_name": "octo/water",
                "html_url": "https://github.com/octo/water",
                "default_branch": "main",
                "description": "water research",
            }
        ],
        "total-results": 1,
    }
    readme = {
        "encoding": "base64",
        "content": base64.b64encode(b"# Water\nResearch repository").decode(),
        "sha": "a" * 40,
        "html_url": "https://github.com/octo/water/blob/main/README.md",
    }
    mock = MockTransport(response(repo, search_url), response(readme, readme_url))
    ctx = context(tmp_path, mock)
    batch = discover(GitHubAdapter(), make_query(), ctx)
    candidate = batch.candidates[0]
    assert isinstance(candidate.extension, GithubMetadata)
    assert candidate.extension.inspection == "README"
    assert candidate.extension.ref == "main"
    assert "readme_text" in json.loads(ctx.artifacts.read(candidate.raw_record_ref))
    assert len(mock.urls) == 2
    assert batch.coverage.query_count == 2
    assert all("/git/" not in url and not url.endswith(".git") for url in mock.urls)


def test_github_readme_is_checked_for_secret_fields(tmp_path: Path) -> None:
    search_url = "https://api.github.com/search/repositories"
    readme_url = "https://api.github.com/repos/octo/water/readme"
    repo = {
        "total_count": 1,
        "items": [
            {
                "id": 1,
                "full_name": "octo/water",
                "html_url": "https://github.com/octo/water",
                "default_branch": "main",
            }
        ],
    }
    unsafe_readme = {
        "encoding": "base64",
        "content": base64.b64encode(b"ghp_" + b"A" * 36).decode(),
    }
    batch = discover(
        GitHubAdapter(),
        make_query(),
        context(
            tmp_path, MockTransport(response(repo, search_url), response(unsafe_readme, readme_url))
        ),
    )
    assert batch.candidates[0].extension.inspection == "REFERENCE_ONLY"
    assert "README_INVALID" in batch.warnings


def test_searxng_requires_https_or_loopback_http() -> None:
    with pytest.raises(ValueError, match="SEARXNG_HTTP_REQUIRES_LOOPBACK"):
        SearXNGAdapter("http://public.example")
    with pytest.raises(ValueError, match="SEARXNG_BASE_URL_INVALID"):
        SearXNGAdapter("https://user:pass@search.example")


def test_other_json_adapters_normalize_metadata(tmp_path: Path) -> None:
    cases = [
        (
            CrossrefAdapter(),
            "https://api.crossref.org/works",
            {
                "message": {
                    "total-results": 1,
                    "items": [
                        {
                            "DOI": "10.1234/example",
                            "title": ["Water"],
                            "URL": "https://doi.org/10.1234/example",
                        }
                    ],
                }
            },
        ),
        (
            ZenodoAdapter(),
            "https://zenodo.org/api/records",
            {
                "hits": {
                    "hits": [
                        {
                            "id": 9,
                            "metadata": {"title": "Water", "publication_date": "2024-04-01"},
                            "links": {"html": "https://zenodo.org/records/9"},
                        }
                    ]
                },
                "links": {},
            },
        ),
        (
            SearXNGAdapter("https://search.example"),
            "https://search.example/search",
            {"results": [{"title": "Water", "url": "https://example.org/water"}]},
        ),
    ]
    for adapter, endpoint, payload in cases:
        batch = discover(
            adapter,
            make_query(),
            context(tmp_path / adapter.provider_id, MockTransport(response(payload, endpoint))),
        )
        assert batch.coverage.status == "RESULTS"
        assert batch.candidates[0].provider_id == adapter.provider_id


def test_date_postfilter_rejects_unknown_and_out_of_window(tmp_path: Path) -> None:
    endpoint = "https://api.openalex.org/works"
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "publication_date": "2020-01-01",
                "display_name": "water",
            },
            {
                "id": "https://openalex.org/W2",
                "publication_date": "2024-03-01",
                "display_name": "water",
            },
            {"id": "https://openalex.org/W3", "display_name": "water"},
        ],
        "meta": {},
    }
    q = make_query(date_from=date(2024, 1, 1))
    batch = discover(
        OpenAlexAdapter(), q, context(tmp_path, MockTransport(response(payload, endpoint)))
    )
    assert [candidate.external_ids.get("openalex") for candidate in batch.candidates] == ["W2"]
    assert batch.coverage.provider_result_count == 3
    assert batch.coverage.filtered_out_count == 2


def test_local_postfilter_zero_is_not_confused_with_provider_zero(tmp_path: Path) -> None:
    endpoint = "https://api.openalex.org/works"
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W1",
                "publication_date": "2020-01-01",
                "display_name": "water",
            }
        ],
        "meta": {},
    }
    batch = discover(
        OpenAlexAdapter(),
        make_query(date_from=date(2024, 1, 1)),
        context(tmp_path, MockTransport(response(payload, endpoint))),
    )
    assert batch.coverage.status == "ZERO_RESULTS"
    assert batch.coverage.provider_result_count == 1
    assert batch.coverage.filtered_out_count == 1
    assert batch.warnings == ("LOCAL_POSTFILTER_ELIMINATED_RESULTS",)
    assert batch.coverage.gap_codes == ("LOCAL_POSTFILTER_ELIMINATED_RESULTS",)


def test_source_transport_fails_closed_without_state_bridge() -> None:
    transport = BrokerSourceTransport(object(), authorize=None)  # type: ignore[arg-type]
    with pytest.raises(SourceDispatchBlocked, match="SOURCE_STATE_AUTHORITY_BRIDGE_UNAVAILABLE"):
        transport.get("https://api.example.org/")


def test_source_transport_binds_capability_and_source_to_broker_call() -> None:
    calls: list[tuple[str, object, str | None]] = []

    class Broker:
        def fetch(self, url, *, headers, authorization, source_id):
            calls.append((url, authorization, source_id))
            return "response"

    capability = object()
    transport = BrokerSourceTransport(
        Broker(),
        authorize=lambda url: capability,
        source_id="SRC-bound",  # type: ignore[arg-type]
    )
    assert transport.get("https://api.example.org/") == "response"
    assert calls == [("https://api.example.org/", capability, "SRC-bound")]
