"""Identity/port validation is local; no real provider is implemented here."""

import hashlib
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from gidr.contracts.common import parse_contract
from gidr.contracts.identity import Project, ResearchRun, Session
from gidr.domain.states import RunState
from gidr.interfaces.provider_ports import ProviderHealth


def test_distinct_project_session_and_run_identity_roundtrip():
    when = datetime(2020, 1, 1, tzinfo=UTC)
    project = Project(
        project_id="PRJ-fixture", title="Synthetic", owner="test", created_at_utc=when
    )
    session = Session(session_id="SES-fixture", project_id=project.project_id, created_at_utc=when)
    run = ResearchRun(
        run_id="RUN-fixture",
        project_id=project.project_id,
        initiating_session_id=session.session_id,
        created_at_utc=when,
        operation="INITIAL",
    )
    assert run.state == RunState.CREATED and session.opened_at_utc is None
    for value in (project, session, run):
        assert parse_contract(type(value), value.model_dump_json()) == value
    with pytest.raises(ValidationError):
        ResearchRun.model_validate({**run.model_dump(), "parent_run_ids": (run.run_id,)})


@pytest.mark.parametrize(
    "data",
    [
        {"status": "LIVE_VERIFIED"},
        {"status": "LIVE_VERIFIED", "checked_at_utc": datetime(2020, 1, 1, tzinfo=UTC)},
    ],
)
def test_live_status_requires_actual_proof_reference(data):
    with pytest.raises(ValidationError):
        ProviderHealth(**data)


def test_mock_health_never_upgrades_itself():
    mock = ProviderHealth(status="MOCK_TESTED")
    assert parse_contract(ProviderHealth, mock.model_dump_json()).status == "MOCK_TESTED"


def test_discovery_batch_rejects_foreign_run_sources():
    from gidr.contracts.sources import SourceCandidate
    from gidr.interfaces.provider_ports import DiscoveryBatch

    # Reuse an explicit serialized synthetic record, never a provider response.
    raw = {
        "schema_version": "1.0",
        "run_id": "RUN-other",
        "created_at_utc": "2020-01-01T00:00:00Z",
        "source_id": "SRC-fixture",
        "source_type": "FIXTURE",
        "provider_id": "synthetic",
        "canonical_url": "https://example.org/fixture",
        "title": None,
        "language": {"code": "unknown", "status": "UNKNOWN"},
        "discovery": {
            "question_id": "RQ-fixture",
            "query_id": "QUERY-fixture",
            "method": "FIXTURE",
        },
        "raw_record_ref": {
            "artifact_id": "ART-fixture",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "size_bytes": 0,
        },
        "access_status": "UNKNOWN",
    }
    source = parse_contract(SourceCandidate, json.dumps(raw))
    with pytest.raises(ValidationError):
        DiscoveryBatch(
            run_id="RUN-fixture",
            created_at_utc=datetime(2020, 1, 1, tzinfo=UTC),
            provider_id="synthetic",
            candidates=(source,),
        )
