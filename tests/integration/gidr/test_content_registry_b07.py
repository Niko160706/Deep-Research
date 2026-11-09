"""Local SQLite/artifact integration for atomic B07 content publication."""

import json
import shutil
import tempfile
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import pytest

from gidr.contracts.common import new_id
from gidr.infrastructure.artifacts.quarantine import QuarantineRecord, QuarantineStore
from gidr.infrastructure.artifacts.repository import ArtifactRepository
from gidr.infrastructure.artifacts.store import ArtifactStore
from gidr.infrastructure.audit.events import event_for
from gidr.infrastructure.content.ingest import ContentIngestor
from gidr.infrastructure.content.registry import ChunkPublication, ContentRegistry, PublishContent
from gidr.infrastructure.db.authority import CommitError, PersistenceAuthority
from gidr.infrastructure.db.database import Database
from gidr.infrastructure.db.index import LexicalIndex
from gidr.infrastructure.db.migrate import migrate
from gidr.infrastructure.discovery.transport import SourceDispatchBlocked
from gidr.infrastructure.network.broker import FetchResponse
from gidr.infrastructure.network.retrieval import Retriever
from gidr.infrastructure.parsing.sandbox import SandboxResult


def test_content_and_fts_publish_and_remove_atomically(tmp_path, request):
    db = Database(tmp_path / "db/state.sqlite3")
    migrate(db)
    authority = PersistenceAuthority(db)
    when = datetime(2024, 1, 1, tzinfo=UTC)
    project = "PRJ-test"
    run = "RUN-test"
    source = "SRC-test"

    class Setup(__import__("gidr.contracts.common", fromlist=["Contract"]).Contract):
        label: str = "fixture"

    def setup(context):
        context.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            (project, "fixture", "test", when.isoformat()),
        )
        context.execute(
            "INSERT INTO sessions(session_id,project_id,created_at_utc) VALUES(?,?,?)",
            ("SES-test", project, when.isoformat()),
        )
        context.execute(
            "INSERT INTO runs(run_id,project_id,session_id,state,created_at_utc,updated_at_utc) VALUES(?,?,?,'CREATED',?,?)",
            (run, project, "SES-test", when.isoformat(), when.isoformat()),
        )
        context.execute(
            "INSERT INTO source_candidates(source_id,run_id,provider_id,canonical_url,title,discovered_at_utc) VALUES(?,?,?,?,?,?)",
            (source, run, "fixture", "https://example.org/original", "Evidence", when.isoformat()),
        )
        context.emit(
            event_for(
                context,
                component="test",
                event_type="fixture.created",
                aggregate_id=source,
                run_id=run,
            )
        )
        return Setup()

    authority.execute("CMD-fixture", "TRACE-fixture", Setup(), Setup, setup)
    shared_memory = Path("/dev/shm")
    if shared_memory.is_dir() and shared_memory.stat().st_dev != tmp_path.stat().st_dev:
        artifact_root = Path(tempfile.mkdtemp(prefix="gidr-b07-artifacts-", dir=shared_memory))
        request.addfinalizer(partial(shutil.rmtree, artifact_root, ignore_errors=True))
    else:
        artifact_root = tmp_path / "artifacts"
    store = ArtifactStore(artifact_root)
    repo = ArtifactRepository(store, authority)
    cross_filesystem = store.root.stat().st_dev != db.path.parent.stat().st_dev
    extracted = "A traceable research result with evidence."
    source_text = f"<html><p>{extracted}</p></html>"
    original = store.publish(source_text.encode())
    text_ref = store.publish(extracted.encode())
    chunk_id = new_id("CHK")
    publish_request = PublishContent(
        run_id=run,
        source_id=source,
        document_id=new_id("DOC"),
        observation_id="OBS-b07-fixture",
        canonical_url="https://example.org/canonical",
        original_artifact=original,
        mime_type="text/html",
        fetched_at_utc=when,
        chunks=(
            ChunkPublication(
                chunk_id=chunk_id,
                artifact=text_ref,
                text=extracted,
                locator_json=(
                    f'{{"unit":"CHAR","start":{source_text.index(extracted)},'
                    f'"end":{source_text.index(extracted) + len(extracted)}}}'
                ),
                language="en",
                parser_id="fixture-parser",
                parser_version="1",
                quality=0.9,
            ),
        ),
    )
    registry = ContentRegistry(authority, repo)

    def fail_before_commit(stage):
        if stage == "before_commit":
            raise RuntimeError("simulated transaction interruption")

    with pytest.raises(CommitError):
        registry.publish("CMD-content", "TRACE-content", publish_request, fault=fail_before_commit)
    if cross_filesystem:
        assert repo.inventory()["orphans"]
    with db.reader() as connection:
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM source_index").fetchone()[0] == 0
    receipt = registry.publish("CMD-content", "TRACE-content", publish_request)
    if cross_filesystem:
        assert repo.inventory()["orphans"] == ()
    assert registry.publish("CMD-content", "TRACE-content", publish_request) == receipt
    assert receipt.chunk_ids == (chunk_id,) and receipt.alias_added
    index = LexicalIndex(authority)
    assert index.search('"traceable research"')
    assert index.search("missingterm", expansions=("traceable research",))
    with pytest.raises(ValueError, match="SEARCH_BOUNDS"):
        index.search("traceable", expansions=tuple(f"term-{i}" for i in range(9)))
    with db.reader() as connection:
        assert (
            connection.execute(
                "SELECT language FROM documents WHERE document_id=?", (receipt.document_id,)
            ).fetchone()[0]
            == "und"
        )
        assert (
            connection.execute(
                "SELECT quality_flags_json FROM chunks WHERE chunk_id=?", (chunk_id,)
            ).fetchone()[0]
            == "[]"
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM documents WHERE document_id=?", (receipt.document_id,)
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM source_observations WHERE observation_id=?",
                (receipt.observation_id,),
            ).fetchone()[0]
            == 1
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM artifact_refs WHERE owner_id IN (?,?)",
                (receipt.document_id, chunk_id),
            ).fetchone()[0]
            == 2
        )
    repeated_chunk_id = new_id("CHK")
    repeated = publish_request.model_copy(
        update={
            "document_id": new_id("DOC"),
            "observation_id": "OBS-b07-repeat",
            "chunks": (
                publish_request.chunks[0].model_copy(update={"chunk_id": repeated_chunk_id}),
            ),
        }
    )
    repeated_receipt = registry.publish("CMD-content-repeat", "TRACE-content-repeat", repeated)
    assert not repeated_receipt.alias_added
    assert repeated_receipt.document_id != receipt.document_id
    assert repeated.chunks[0].artifact == publish_request.chunks[0].artifact
    registry.remove_index("CMD-index-remove", "TRACE-index-remove", source)
    assert index.search('"traceable research"') == ()
    with db.reader() as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM chunks WHERE chunk_id IN (?,?)",
                (chunk_id, repeated_chunk_id),
            ).fetchone()[0]
            == 2
        )
        assert connection.execute("SELECT count(*) FROM documents").fetchone()[0] == 2
    with db.reader() as connection:
        db.check(connection)


def test_ingestor_connects_mock_retrieval_parser_chunker_and_registry(tmp_path):
    db = Database(tmp_path / "db/state.sqlite3")
    migrate(db)
    authority = PersistenceAuthority(db)
    when = datetime(2024, 1, 1, tzinfo=UTC)

    class Setup(__import__("gidr.contracts.common", fromlist=["Contract"]).Contract):
        label: str = "fixture"

    def setup(context):
        context.execute(
            "INSERT INTO projects(project_id,title,owner,created_at_utc) VALUES(?,?,?,?)",
            ("PRJ-pipe", "fixture", "test", when.isoformat()),
        )
        context.execute(
            "INSERT INTO sessions(session_id,project_id,created_at_utc) VALUES(?,?,?)",
            ("SES-pipe", "PRJ-pipe", when.isoformat()),
        )
        context.execute(
            "INSERT INTO runs(run_id,project_id,session_id,state,created_at_utc,updated_at_utc) VALUES(?,?,?,'CREATED',?,?)",
            ("RUN-pipe", "PRJ-pipe", "SES-pipe", when.isoformat(), when.isoformat()),
        )
        context.execute(
            "INSERT INTO source_candidates(source_id,run_id,provider_id,canonical_url,title,discovered_at_utc) VALUES(?,?,?,?,?,?)",
            ("SRC-pipe", "RUN-pipe", "fixture", "https://example.org", "Fixture", when.isoformat()),
        )
        context.emit(
            event_for(
                context,
                component="test",
                event_type="fixture.created",
                aggregate_id="SRC-pipe",
                run_id="RUN-pipe",
            )
        )
        return Setup()

    authority.execute("CMD-fixture-pipe", "TRACE-fixture-pipe", Setup(), Setup, setup)
    store = ArtifactStore(tmp_path / "artifacts")
    registry = ContentRegistry(authority, ArtifactRepository(store, authority))
    body = (
        b"<html><nav>menu</nav><main><p>"
        + b"Evidence from this mocked source remains source data. " * 8
        + b"</p></main></html>"
    )

    class Transport:
        def get(self, url, *, headers=None):
            quarantine = QuarantineRecord(
                Path("/tmp/mock-staged"), "0" * 64, len(body), "text/html", "text/html", "STAGED"
            )
            return FetchResponse(url, 200, {"content-type": "text/html"}, body, 0, quarantine)

    ingestor = ContentIngestor(
        Retriever(Transport(), store, QuarantineStore(tmp_path / "quarantine")),
        store,
        registry,
    )
    result = ingestor.ingest("RUN-pipe", "SRC-pipe", "https://example.org/#citation")
    assert result.status == "INDEXED" and result.content is not None
    assert result.retrieval.document.original_artifact is not None
    assert LexicalIndex(authority).search('"mocked source"')
    with db.reader() as connection:
        document = connection.execute(
            "SELECT retrieval_status,language FROM documents WHERE document_id=?",
            (result.content.document_id,),
        ).fetchone()
        assert document["retrieval_status"] == "RETRIEVED"
        assert document["language"] == "en"
        locator = json.loads(
            connection.execute(
                "SELECT locator_json FROM chunks WHERE document_id=? ORDER BY chunk_id LIMIT 1",
                (result.content.document_id,),
            ).fetchone()[0]
        )
        raw_html = body.decode("utf-8")
        assert locator["unit"] == "CHAR"
        assert "Evidence from this mocked source" in raw_html[locator["start"] : locator["end"]]
        assert (
            connection.execute(
                "SELECT count(*) FROM chunks WHERE document_id=?", (result.content.document_id,)
            ).fetchone()[0]
            >= 1
        )
        db.check(connection)

    class BlockedTransport:
        def get(self, url, *, headers=None):
            raise SourceDispatchBlocked("SOURCE_STATE_AUTHORITY_BRIDGE_UNAVAILABLE")

    blocked = ContentIngestor(
        Retriever(BlockedTransport(), store, QuarantineStore(tmp_path / "blocked-quarantine")),
        store,
        registry,
    ).ingest("RUN-pipe", "SRC-pipe", "https://example.org/blocked")
    assert blocked.status == "RETRIEVAL_FAILURE_RECORDED"
    with db.reader() as connection:
        row = connection.execute(
            "SELECT d.retrieval_status,o.access_status FROM documents d JOIN source_observations o USING(observation_id) WHERE d.document_id=?",
            (blocked.retrieval.document.document_id,),
        ).fetchone()
        assert tuple(row) == ("BLOCKED", "BLOCKED")
        assert (
            connection.execute(
                "SELECT count(*) FROM audit_events WHERE aggregate_id=?",
                (blocked.retrieval.document.document_id,),
            ).fetchone()[0]
            == 1
        )

    pdf_body = b"%PDF-1.7 synthetic document bytes"

    class PdfTransport:
        def get(self, url, *, headers=None):
            quarantine = QuarantineRecord(
                Path("/tmp/mock-pdf"),
                "0" * 64,
                len(pdf_body),
                "application/pdf",
                "application/pdf",
                "STAGED",
            )
            return FetchResponse(
                url, 200, {"content-type": "application/pdf"}, pdf_body, 0, quarantine
            )

    class PdfSandbox:
        def run(self, argv, *, input_path, timeout_seconds):
            assert input_path.is_file() and input_path.read_bytes() == pdf_body
            assert argv[1:] == ["-layout", "-enc", "UTF-8", "/input/document", "-"]
            return SandboxResult(
                0,
                b"Page one source text with sufficient evidence.\fPage two continuation.\f",
                "stderr_sha256:" + "0" * 64,
                0.01,
            )

    pdf_result = ContentIngestor(
        Retriever(PdfTransport(), store, QuarantineStore(tmp_path / "pdf-quarantine")),
        store,
        registry,
        pdf_sandbox=PdfSandbox(),
    ).ingest("RUN-pipe", "SRC-pipe", "https://example.org/paper.pdf")
    assert pdf_result.status == "INDEXED" and pdf_result.content is not None
    assert LexicalIndex(authority).search('"Page one source"')
    with db.reader() as connection:
        pdf_locators = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT locator_json FROM chunks WHERE document_id=?",
                (pdf_result.content.document_id,),
            ).fetchall()
        ]
        assert {locator["page"] for locator in pdf_locators} == {1, 2}
        assert len(pdf_locators) == 2

    mirror_id = "SRC-doi-mirror"
    archive_id = "SRC-wayback-snapshot"

    class MoreSources(__import__("gidr.contracts.common", fromlist=["Contract"]).Contract):
        count: int = 2

    def add_identity_sources(context):
        context.execute(
            "INSERT INTO source_candidates(source_id,run_id,provider_id,canonical_url,title,discovered_at_utc) VALUES(?,?,?,?,?,?)",
            (
                mirror_id,
                "RUN-pipe",
                "fixture",
                "https://mirror.example/paper",
                "Mirror",
                when.isoformat(),
            ),
        )
        context.execute(
            "INSERT INTO source_candidates(source_id,run_id,provider_id,canonical_url,title,discovered_at_utc) VALUES(?,?,?,?,?,?)",
            (
                archive_id,
                "RUN-pipe",
                "wayback_cdx",
                "https://web.archive.org/web/20240101000000/https://example.org/",
                "Snapshot",
                when.isoformat(),
            ),
        )
        context.emit(
            event_for(
                context,
                component="test",
                event_type="fixture.identity_sources_created",
                aggregate_id=mirror_id,
                run_id="RUN-pipe",
            )
        )
        return MoreSources()

    authority.execute(
        "CMD-identity-sources",
        "TRACE-identity-sources",
        MoreSources(),
        MoreSources,
        add_identity_sources,
    )
    first_doi = registry.record_identity(
        "CMD-identity-doi-original",
        "TRACE-identity-doi-original",
        "SRC-pipe",
        "https://doi.org/10.1234/AbC.5",
    )
    mirror_doi = registry.record_identity(
        "CMD-identity-doi-mirror",
        "TRACE-identity-doi-mirror",
        mirror_id,
        "doi:10.1234/abc.5",
    )
    snapshot = registry.record_identity(
        "CMD-identity-wayback",
        "TRACE-identity-wayback",
        archive_id,
        "https://web.archive.org/web/20240101000000/https://example.org/",
    )
    assert first_doi.related_source_ids == ()
    assert mirror_doi.related_source_ids == ("SRC-pipe",)
    assert snapshot.snapshot_parent_id == "SRC-pipe"
    with db.reader() as connection:
        dependencies = {
            row[0]
            for row in connection.execute(
                "SELECT kind FROM source_dependencies WHERE from_source_id IN (?,?)",
                (mirror_id, archive_id),
            )
        }
        assert dependencies == {"SAME_IDENTITY", "SNAPSHOT_OF"}


def test_b07_migration_adds_document_language_and_parser_flags(tmp_path):
    db = Database(tmp_path / "db/migration.sqlite3")
    assert migrate(db, target_version=4) == 4
    assert migrate(db) == 6
    with db.reader() as connection:
        document_columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        chunk_columns = {row[1] for row in connection.execute("PRAGMA table_info(chunks)")}
        versions = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert "language" in document_columns
        assert "quality_flags_json" in chunk_columns
        assert [row[0] for row in versions] == [1, 2, 3, 4, 5, 6]
        db.check(connection)
