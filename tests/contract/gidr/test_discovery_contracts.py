from datetime import date

import pytest
from pydantic import ValidationError

from gidr.infrastructure.discovery.adapters import P0_ADAPTERS
from gidr.infrastructure.discovery.compiler import compile_query
from gidr.infrastructure.discovery.contracts import (
    CanonicalQuery,
    ProviderCapabilities,
    SourceCoverageObservation,
)
from gidr.infrastructure.discovery.registry import default_registry


def query(**changes: object) -> CanonicalQuery:
    data: dict[str, object] = {
        "query_id": "QUERY-test",
        "question_id": "RQ-test",
        "text": "climate adaptation",
        "max_results": 10,
    }
    data.update(changes)
    return CanonicalQuery.model_validate(data)


def test_query_rejects_inverted_dates_and_unknown_mandatory_filters() -> None:
    with pytest.raises(ValidationError):
        query(date_from=date(2024, 2, 1), date_to=date(2024, 1, 1))
    with pytest.raises(ValidationError):
        query(mandatory_filters=("organization",))
    with pytest.raises(ValidationError, match="MANDATORY_FILTER_VALUE_REQUIRED"):
        query(mandatory_filters=("language",))


def test_provider_compiler_reports_native_and_local_filters() -> None:
    compiled = compile_query(
        query(languages=("en",), domains=("example.org",), required_terms=("study",)),
        default_registry().get("crossref"),
    )
    assert compiled.blocked_reason is None
    assert {f.name: f.disposition for f in compiled.filters} == {
        "text": "NATIVE",
        "language": "LOCAL_POSTFILTER",
        "domain": "LOCAL_POSTFILTER",
        "required_terms": "LOCAL_POSTFILTER",
    }


def test_mandatory_unsupported_filter_blocks_compilation() -> None:
    caps = ProviderCapabilities(
        provider_id="isolated",
        category="web",
        status="MOCK_TESTED",
        native_filters=("text",),
        pagination="NONE",
        max_page_size=1,
        base_hosts=(),
    )
    compiled = compile_query(query(languages=("en",), mandatory_filters=("language",)), caps)
    assert compiled.blocked_reason == "MANDATORY_FILTER_UNSUPPORTED"
    assert compiled.filters[-1].disposition == "UNSUPPORTED"


def test_registry_distinguishes_conditional_provider_from_unselected() -> None:
    registry = default_registry()
    assert registry.get("core").status == "NOT_IMPLEMENTED"
    assert "semantic_scholar" in {item.provider_id for item in registry.all()}
    assert all(not item.default_clone for item in registry.all())
    assert set(P0_ADAPTERS) == {item.provider_id for item in registry.implemented()}
    academic = registry.ordered_for(("academic",))
    assert academic[0].provider_id == "openalex"
    assert all(item.status != "NOT_IMPLEMENTED" for item in academic)


def test_duplicate_registry_ids_are_rejected() -> None:
    registry = default_registry()
    with pytest.raises(ValueError, match="DUPLICATE_PROVIDER_ID"):
        registry.register(registry.get("github"))


def test_coverage_counts_and_attempt_state_are_consistent() -> None:
    with pytest.raises(ValidationError):
        SourceCoverageObservation(
            run_id="RUN-test",
            question_id="RQ-test",
            provider_id="core",
            category="academic",
            selected=False,
            attempted=False,
            status="NOT_SELECTED",
            query_count=1,
        )
    with pytest.raises(ValidationError):
        SourceCoverageObservation(
            run_id="RUN-test",
            question_id="RQ-test",
            provider_id="core",
            category="academic",
            selected=True,
            attempted=True,
            status="RESULTS",
            candidate_count=1,
            unique_candidate_count=2,
        )
    with pytest.raises(ValidationError):
        SourceCoverageObservation(
            run_id="RUN-test",
            question_id="RQ-test",
            provider_id="core",
            category="academic",
            selected=True,
            attempted=True,
            status="ZERO_RESULTS",
            provider_result_count=1,
            filtered_out_count=2,
        )
