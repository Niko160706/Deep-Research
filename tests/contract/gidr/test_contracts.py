"""Synthetic offline fixtures only. No fixture is evidence of a live adapter."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from gidr.contracts.common import (
    ArtifactRef,
    ContractInputError,
    Language,
    ObservedValue,
    ValueState,
    new_id,
    parse_contract,
)
from gidr.contracts.control import (
    AuditEvent,
    Question,
    ResearchPlanProposal,
    ReservationPayload,
    TaskCommand,
    TaskSpecification,
)
from gidr.contracts.evidence import Claim, EvidenceRecord
from gidr.contracts.models import (
    ChoiceDecision,
    DecisionConfidence,
    DecisionRecord,
    FinancialCharge,
    ModelExecutionResult,
    ScoreDecision,
    TokenUsage,
    TranslationArtifact,
)
from gidr.contracts.sources import (
    ArxivMetadata,
    ContentChunk,
    DiscoveryProvenance,
    GdeltMetadata,
    GithubMetadata,
    RetrievedDocument,
    SourceCandidate,
    TextLocator,
    WaybackMetadata,
)
from gidr.domain.links import validate_artifact_links
from gidr.domain.states import Verdict

WHEN = datetime(2020, 1, 1, tzinfo=UTC)
BASE = {"run_id": "RUN-fixture", "created_at_utc": WHEN}
TEXT = "匿名の噂は未検証です。 This is synthetic fixture content, never a live source."


TEXT_BYTES = TEXT.encode()


def artifact(name="text", data=TEXT_BYTES):
    return ArtifactRef(
        artifact_id="ART-" + name, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data)
    )


def candidate(provider="arxiv"):
    extensions = {
        "arxiv": ArxivMetadata(arxiv_id="2401.00001", doi=ObservedValue(state=ValueState.UNKNOWN)),
        "gdelt": GdeltMetadata(
            seen_at_utc=WHEN, source_country=ObservedValue(state=ValueState.UNKNOWN)
        ),
        "github": GithubMetadata(repository="synthetic/fixture"),
        "wayback": WaybackMetadata(
            original_url="https://example.org/fixture",
            capture_at_utc=WHEN,
            capture_url="https://web.archive.org/web/20200101000000/https://example.org/fixture",
        ),
    }
    return SourceCandidate(
        **BASE,
        source_id="SRC-fixture",
        provider_id=provider,
        source_type="SYNTHETIC_FIXTURE",
        canonical_url="https://example.org/synthetic-fixture",
        external_ids={"fixture": "offline-only"},
        title=None,
        language=Language(code="unknown", status="UNKNOWN"),
        discovery=DiscoveryProvenance(
            question_id="RQ-fixture", query_id="QUERY-fixture", method="FIXTURE"
        ),
        raw_record_ref=artifact("metadata", b'{"synthetic":true}'),
        access_status="UNKNOWN",
        extension=extensions[provider],
    )


def bundle():
    source = candidate()
    doc = RetrievedDocument(
        **BASE,
        document_id="DOC-fixture",
        source_id=source.source_id,
        observation_id="OBS-fixture",
        fetched_at_utc=WHEN,
        status="RETRIEVED",
        mime_type="text/plain",
        content_hash=artifact().sha256,
        original_artifact=artifact(),
    )
    locator = TextLocator(unit="CHAR", start=0, end=len(TEXT), page=1)
    chunk = ContentChunk(
        **BASE,
        chunk_id="CHK-fixture",
        document_id=doc.document_id,
        original_locator=locator,
        language=Language(code="mixed", status="MIXED"),
        extracted_text_ref=artifact(),
        parser_id="fixture",
        parser_version="fixture-only",
    )
    claim = Claim(
        **BASE,
        claim_id="CLM-fixture",
        proposition="A rumor remains unverified",
        claim_types=("ANONYMOUS_TESTIMONY",),
        scope="Synthetic test only",
        valid_time=None,
        definition=None,
        original_formulation="匿名の噂は未検証です。",
        derivation_ref=artifact(),
    )
    evidence = EvidenceRecord(
        **BASE,
        evidence_id="EVD-fixture",
        claim_id=claim.claim_id,
        chunk_id=chunk.chunk_id,
        verbatim_excerpt=TEXT,
        original_locator=locator,
        relation="INSUFFICIENT",
        context_ref=artifact(),
    )
    return {
        "sources": (source,),
        "documents": (doc,),
        "chunks": (chunk,),
        "claims": (claim,),
        "evidence": (evidence,),
    }


@pytest.mark.parametrize("provider", ["arxiv", "gdelt", "github", "wayback"])
def test_provider_roundtrip_preserves_raw_identity_and_unknown_metadata(provider):
    value = candidate(provider)
    restored = parse_contract(SourceCandidate, value.model_dump_json())
    assert restored == value
    assert restored.source_id == value.source_id
    assert restored.title is None
    assert restored.language.code == "unknown"
    assert restored.raw_record_ref.sha256 == value.raw_record_ref.sha256
    assert restored.content_status == "NOT_FETCHED"


@pytest.mark.parametrize(
    "data",
    [
        {"source_id": "DOC-wrong-family"},
        {"source_id": "SRC-../escape"},
        {"unexpected": "field"},
        {"created_at_utc": WHEN.replace(tzinfo=None)},
        {"canonical_url": "file:///etc/passwd"},
        {"canonical_url": "https://user:password@example.org"},
        {"canonical_url": "https://example.org:invalid/"},
        {"canonical_url": "https://example.org/\n"},
        {"external_ids": {"fixture": 42}},
        {"extension": {"kind": "unsupported", "extra": "x"}},
        {"extension": GithubMetadata(repository="synthetic/fixture")},
    ],
)
def test_invalid_candidates_rejected(data):
    payload = candidate().model_dump()
    payload.update(data)
    with pytest.raises(ValidationError):
        SourceCandidate.model_validate(payload)


@pytest.mark.parametrize(
    "state", [ValueState.UNKNOWN, ValueState.NOT_APPLICABLE, ValueState.FAILED]
)
def test_unknown_inapplicable_and_failed_remain_distinct(state):
    value = ObservedValue[str](
        state=state, reason_code="LOOKUP_FAILED" if state == ValueState.FAILED else None
    )
    assert value.value is None
    assert value.state == state
    with pytest.raises(ValidationError):
        ObservedValue[str](state=state, value="invented")


def test_missing_doi_never_fabricated():
    value = candidate().extension
    assert isinstance(value, ArxivMetadata)
    assert value.doi.state == ValueState.UNKNOWN and value.doi.value is None
    with pytest.raises(ValidationError):
        ArxivMetadata(
            arxiv_id="2401.00001", doi=ObservedValue(state=ValueState.KNOWN, value="fake-doi")
        )


def test_missing_media_date_remains_unknown_without_dropping_candidate():
    original = candidate("gdelt")
    extension = GdeltMetadata(source_country=ObservedValue(state=ValueState.UNKNOWN))
    restored = SourceCandidate.model_validate({**original.model_dump(), "extension": extension})
    assert parse_contract(SourceCandidate, restored.model_dump_json()).extension.seen_at_utc is None


@pytest.mark.parametrize("version", [None, "0.9", "2.0"])
def test_schema_evolution_requires_explicit_supported_version(version):
    raw = json.loads(candidate().model_dump_json())
    if version is None:
        raw.pop("schema_version")
    else:
        raw["schema_version"] = version
    with pytest.raises(ContractInputError):
        parse_contract(SourceCandidate, json.dumps(raw))


def test_validation_errors_never_echo_raw_inputs():
    canary = "SYNTHETIC-CREDENTIAL-CANARY-NEVER-A-REAL-KEY"
    raw = json.loads(candidate().model_dump_json())
    raw["api_key"] = canary
    with pytest.raises(ContractInputError) as caught:
        parse_contract(SourceCandidate, json.dumps(raw))
    assert canary not in str(caught.value) and canary not in repr(caught.value)
    assert caught.value.__cause__ is None


def test_stable_ids_and_timezone_normalization():
    ids = {new_id("SRC") for _ in range(1000)}
    assert len(ids) == 1000
    raw = candidate().model_dump()
    raw["created_at_utc"] = datetime(2020, 1, 1, 7, tzinfo=timezone(timedelta(hours=7)))
    assert SourceCandidate.model_validate(raw).created_at_utc == WHEN


def test_frozen_record_and_schema_generation():
    with pytest.raises(ValidationError):
        candidate().title = "changed"
    with pytest.raises(TypeError):
        candidate().external_ids["new"] = "changed"
    models = [
        SourceCandidate,
        RetrievedDocument,
        ContentChunk,
        Claim,
        EvidenceRecord,
        TranslationArtifact,
        DecisionRecord,
        ModelExecutionResult,
        AuditEvent,
        TaskCommand[ReservationPayload],
        ResearchPlanProposal,
    ]
    for model in models:
        schema = model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert "schema_version" in schema["properties"]


def test_reference_snapshot_and_unassessed_claim():
    data = bundle()
    validate_artifact_links(**data)
    assert data["claims"][0].status == Verdict.UNASSESSED
    assert data["evidence"][0].locator_verification == "UNCHECKED"


@pytest.mark.parametrize(
    "kind", ["source", "document", "claim", "span", "page", "duplicate", "run"]
)
def test_orphan_and_misaligned_evidence_rejected(kind):
    data = bundle()
    if kind == "source":
        data["sources"] = ()
    elif kind == "document":
        data["documents"] = ()
    elif kind == "claim":
        data["claims"] = ()
    elif kind == "duplicate":
        data["sources"] *= 2
    elif kind == "run":
        data["claims"] = (
            Claim.model_validate({**data["claims"][0].model_dump(), "run_id": "RUN-other"}),
        )
    else:
        locator = TextLocator(
            unit="CHAR",
            start=0,
            end=len(TEXT) + 1 if kind == "span" else len(TEXT),
            page=2 if kind == "page" else 1,
        )
        data["evidence"] = (
            EvidenceRecord.model_validate(
                {**data["evidence"][0].model_dump(), "original_locator": locator}
            ),
        )
    with pytest.raises(ValueError):
        validate_artifact_links(**data)


@pytest.mark.parametrize("status", ["FAILED", "BLOCKED"])
def test_failed_retrieval_is_not_success(status):
    doc = bundle()["documents"][0]
    with pytest.raises(ValidationError):
        RetrievedDocument.model_validate(
            {**doc.model_dump(), "status": status, "error_code": "DENIED"}
        )


def test_snapshot_versions_and_hash_consistency():
    doc = bundle()["documents"][0]
    new = RetrievedDocument.model_validate(
        {
            **doc.model_dump(),
            "document_id": "DOC-second",
            "observation_id": "OBS-second",
            "fetched_at_utc": WHEN + timedelta(days=1),
        }
    )
    assert new.source_id == doc.source_id and new.document_id != doc.document_id
    with pytest.raises(ValidationError):
        RetrievedDocument.model_validate({**doc.model_dump(), "content_hash": "a" * 64})


def test_label_fallback_does_not_invent_probability():
    answer = ChoiceDecision(choices=("KEEP", "DEFER", "UNKNOWN"), selected="UNKNOWN")
    assert answer.confidence.value is None
    assert answer.confidence.semantics == "UNAVAILABLE"
    assert answer.model_dump()["confidence"]["value"] is None
    with pytest.raises(ValidationError):
        DecisionConfidence(value=0.9)
    with pytest.raises(ValidationError):
        ChoiceDecision(choices=("KEEP", "DEFER"), selected="DELETE")
    with pytest.raises(ValidationError):
        ScoreDecision(score=float("nan"), rubric_version="test-v1")


def test_unknown_tokens_and_charge_are_not_zero():
    assert TokenUsage().input_tokens is None and FinancialCharge().usd is None
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=0, output_tokens=0)
    with pytest.raises(ValidationError):
        FinancialCharge(usd=Decimal(0))
    assert FinancialCharge(state="KNOWN", usd=Decimal(0)).usd == 0


def test_model_failure_cannot_contain_successful_output():
    raw = dict(
        BASE,
        model_execution_id="MOD-fixture",
        task_id="TASK-fixture",
        attempt_ids=("ATT-fixture",),
        provider_id="synthetic",
        model_id=None,
        model_version=None,
        prompt_version="fixture-v1",
        status="FAILED",
        error_code="AUTH",
    )
    result = ModelExecutionResult(**raw)
    assert result.charge.usd is None and result.output_ref is None
    with pytest.raises(ValidationError):
        ModelExecutionResult(**{**raw, "output_ref": artifact()})


def test_translation_preserves_original_and_is_not_a_witness():
    data = bundle()
    chunk = data["chunks"][0]
    translated = TranslationArtifact(
        **BASE,
        translation_id="TRN-fixture",
        chunk_id=chunk.chunk_id,
        original_text_ref=chunk.extracted_text_ref,
        source_language=chunk.language,
        target_language=Language(code="id", status="DECLARED"),
        translator_provider="synthetic",
        translator_model="fixture-only",
        prompt_version="fixture-v1",
        translated_artifact=artifact("translation", b"Synthetic translation"),
        risk_flags=("MODALITY",),
    )
    validate_artifact_links(**data, translations=(translated,))
    with pytest.raises(ValidationError):
        TranslationArtifact.model_validate(
            {**translated.model_dump(), "independent_evidence": True}
        )
    damaged = TranslationArtifact.model_validate(
        {**translated.model_dump(), "original_text_ref": artifact("wrong", b"wrong")}
    )
    with pytest.raises(ValueError):
        validate_artifact_links(**data, translations=(damaged,))


def test_question_dependencies_and_proposal_are_not_approval():
    q = Question(
        question_id="RQ-a", text="What is known?", evidence_requirements=("Original evidence",)
    )
    raw = dict(
        BASE,
        proposal_id="PROP-fixture",
        project_id="PRJ-fixture",
        session_id="SES-fixture",
        expected_plan_version=0,
        objective="Synthetic objective",
        scope_in=("Fixture",),
        questions=(q,),
        source_categories=("SCHOLARLY",),
        budget_policy_ref=artifact(),
    )
    assert ResearchPlanProposal(**raw).status == "PROPOSED"
    cycle = Question(
        question_id="RQ-a",
        text=q.text,
        evidence_requirements=q.evidence_requirements,
        dependencies=("RQ-a",),
    )
    with pytest.raises(ValidationError):
        ResearchPlanProposal(**{**raw, "questions": (cycle,)})
    with pytest.raises(ValidationError):
        ResearchPlanProposal(**{**raw, "status": "USER_APPROVED"})
    with pytest.raises(ValidationError):
        TaskSpecification(
            task_id="TASK-fixture",
            question_id="RQ-fixture",
            task_type="FETCH",
            reason_code="TEST",
            dependencies=("TASK-fixture",),
        )


def test_typed_command_roundtrip_and_revision_guard_shape():
    payload = ReservationPayload(
        task_id="TASK-fixture",
        attempt_id="ATT-fixture",
        lease_ttl_seconds=30,
        budget_policy_ref=artifact(),
    )
    command = TaskCommand[ReservationPayload](
        command_id="CMD-fixture",
        trace_id="TRACE-fixture",
        run_id="RUN-fixture",
        actor="SCHEDULER",
        command_type="RESERVE_TASK",
        aggregate_id="TASK-fixture",
        expected_revision=2,
        plan_version=1,
        grant_id="GRANT-fixture",
        idempotency_key="fixture-reserve-1",
        payload=payload,
        created_at_utc=WHEN,
    )
    assert parse_contract(TaskCommand[ReservationPayload], command.model_dump_json()) == command
    with pytest.raises(ValidationError):
        TaskCommand[ReservationPayload].model_validate(
            {**command.model_dump(), "expected_revision": -1}
        )
    with pytest.raises(ValidationError):
        TaskCommand[ReservationPayload].model_validate({**command.model_dump(), "actor": "LLM"})


def test_audit_event_retains_identity_refs_and_unknown_commit_sequence():
    event = AuditEvent(
        **BASE,
        event_id="EVT-fixture",
        trace_id="TRACE-fixture",
        command_id="CMD-fixture",
        session_id="SES-fixture",
        task_id="TASK-fixture",
        attempt_id="ATT-fixture",
        plan_version=1,
        grant_id="GRANT-fixture",
        aggregate_type="TASK",
        aggregate_id="TASK-fixture",
        revision_before=1,
        revision_after=2,
        old_state="READY",
        new_state="RESERVED",
        component="scheduler",
        event_type="task.reserved",
        cause="RESERVE_TASK",
        policy_version="fixture-v1",
        reason_code="AUTHORIZED",
        occurred_at_utc=WHEN,
    )
    assert event.seq is None and event.committed_at_utc is None
    assert parse_contract(AuditEvent, event.model_dump_json()) == event
    with pytest.raises(ValidationError):
        AuditEvent.model_validate({**event.model_dump(), "revision_after": 0})


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"0.9","schema_version":"1.0"}',
        '{"schema_version":"1.0","nested":{"a":1,"a":2}}',
    ],
)
def test_ambiguous_duplicate_json_keys_rejected(payload):
    with pytest.raises(ContractInputError, match="Invalid contract JSON"):
        parse_contract(SourceCandidate, payload)


def test_constructed_or_copied_invalid_models_are_revalidated():
    bad = candidate().model_copy(update={"source_id": "DOC-wrong-family"})
    with pytest.raises(ValidationError):
        SourceCandidate.model_validate(bad)
