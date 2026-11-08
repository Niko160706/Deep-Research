from __future__ import annotations

import json
from pathlib import Path

import pytest

from gidr.infrastructure.audit.projection import AuditExporter
from gidr.infrastructure.db.authority import PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.db.migrate import migrate
from gidr.infrastructure.security.audit_sink import SQLiteSecurityEventSink
from gidr.infrastructure.security.events import SecurityObservation


def test_security_observation_commits_canonical_event_and_redacted_projection(
    tmp_path: Path,
) -> None:
    database = Database(tmp_path / "state.sqlite3")
    migrate(database)
    authority = PersistenceAuthority(database)
    sink = SQLiteSecurityEventSink(authority)
    observation = SecurityObservation(
        category="UNAUTHORIZED_ACTION",
        reason_code="PROVIDER_OUT_OF_SCOPE",
        request_id="ACT-fixture",
        source_id="SRC-fixture",
        url_sha256="a" * 64,
    )
    sink.record_security(observation)
    with database.reader() as connection:
        row = connection.execute(
            "SELECT component,event_type,event_payload_json FROM audit_events"
        ).fetchone()
        count = connection.execute("SELECT count(*) FROM processed_commands").fetchone()[0]
    assert row["component"] == "security"
    assert row["event_type"] == "UNAUTHORIZED_ACTION"
    payload = json.loads(row["event_payload_json"])
    assert payload["security"]["url_sha256"] == "a" * 64
    assert "url" not in payload
    assert count == 1

    projection = AuditExporter(authority).export(tmp_path / "logs")
    security_log = projection / "security.jsonl"
    assert security_log.exists()
    projected = json.loads(security_log.read_text().splitlines()[0])
    assert projected["reason_code"] == "PROVIDER_OUT_OF_SCOPE"
    assert "query" not in security_log.read_text()


def test_security_sink_failure_is_sanitized_and_does_not_write_partial_event(
    tmp_path: Path,
) -> None:
    from gidr.infrastructure.security.audit_sink import SecurityAuditFailure
    from gidr.infrastructure.security.events import SecurityObservation

    class BrokenAuthority:
        def execute(self, *args, **kwargs):
            raise RuntimeError("synthetic failure detail")

    sink = SQLiteSecurityEventSink(BrokenAuthority())  # type: ignore[arg-type]
    with pytest.raises(SecurityAuditFailure, match="PERSISTENCE_FAILED"):
        sink.record_security(
            SecurityObservation(
                category="PARSER_SANDBOX_FAILURE",
                reason_code="PARSER_TIMEOUT",
                request_id="PARSE-fixture",
                content_sha256="b" * 64,
            )
        )
