from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import pytest

from gidr.contracts.common import ArtifactRef, Language, new_id
from gidr.contracts.models import ModelExecutionResult, ModelTask, TranslationArtifact
from gidr.infrastructure.models.ports import ExecutableModelCall
from gidr.infrastructure.models.translation import (
    AlignedTranslation,
    SelectiveTranslator,
    TranslationPublisher,
)


class TranslationGatewayStub:
    def __init__(self, output_ref: ArtifactRef) -> None:
        self.output_ref = output_ref
        self.calls: list[ExecutableModelCall] = []

    async def execute(self, task: ModelTask, call: ExecutableModelCall) -> ModelExecutionResult:
        self.calls.append(call)
        return ModelExecutionResult(
            model_execution_id=new_id("MOD"),
            run_id=task.run_id,
            created_at_utc=datetime.now(UTC),
            task_id=task.task_id,
            attempt_ids=(new_id("ATT"),),
            provider_id="gemini",
            model_id="configured-gemini-model",
            model_version=None,
            prompt_version=call.contract.prompt_version,
            status="SUCCESS",
            output_ref=self.output_ref,
        )


class TranslationReaderStub:
    def __init__(self, content: bytes) -> None:
        self.content = content

    async def read(self, artifact: ArtifactRef) -> bytes:
        return self.content


class TranslationPublisherStub(TranslationPublisher):
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    async def publish_translation(self, **record) -> ArtifactRef:
        self.records.append(record)
        translated = str(record["translated_text"]).encode("utf-8")
        return ArtifactRef(
            artifact_id=new_id("ART"),
            sha256=hashlib.sha256(translated).hexdigest(),
            size_bytes=len(translated),
        )


def task(source_ref: ArtifactRef) -> ModelTask:
    return ModelTask(
        run_id=new_id("RUN"),
        created_at_utc=datetime(2026, 11, 8, tzinfo=UTC),
        task_id=new_id("TASK"),
        plan_version=1,
        authorization_ref="GRANT-TRANSLATION",
        task_type="translation",
        required_capabilities=("translation",),
        input_refs=(source_ref,),
        input_language=Language(code="en", status="DECLARED"),
        risk_class="MEDIUM",
        reservation_id="ATT-TRANSLATION",
        attempt_limit=2,
        preferred_provider="gemini",
        allowed_fallbacks=(),
        prompt_version="translate-v2",
        privacy_scope="PUBLIC_SOURCE",
    )


def output_json(source: str, source_ref: ArtifactRef, translated: str) -> bytes:
    return json.dumps(
        {
            "source_sha256": source_ref.sha256,
            "target_language": {"code": "id", "status": "DECLARED"},
            "segments": [
                {
                    "source_start": 0,
                    "source_end": len(source),
                    "original_text": source,
                    "translated_text": translated,
                }
            ],
        }
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_selective_translation_preserves_original_reference_alignment_and_risk_flags() -> (
    None
):
    source = "The report did not confirm 12 kg on 2024-05-06."
    source_ref = ArtifactRef(
        artifact_id=new_id("ART"),
        sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        size_bytes=len(source.encode("utf-8")),
    )
    translated_output = output_json(
        source, source_ref, "Laporan tidak mengonfirmasi 12 kg pada 2024-05-06."
    )
    output_ref = ArtifactRef(
        artifact_id=new_id("ART"), sha256="1" * 64, size_bytes=len(translated_output)
    )
    gateway = TranslationGatewayStub(output_ref)
    publisher = TranslationPublisherStub()
    service = SelectiveTranslator(
        gateway,
        output_reader=TranslationReaderStub(translated_output),
        publisher=publisher,
    )
    task_value = task(source_ref)
    chunk_id = new_id("CHK")

    result = await service.translate(
        task=task_value,
        source_ref=source_ref,
        chunk_id=chunk_id,
        source_text=source,
        source_language=Language(code="en", status="DECLARED"),
        target_language=Language(code="id", status="DECLARED"),
        selected_for_translation=True,
        selection_reason="High-value source requires human-readable comparison.",
        prompt_version="translate-v2",
    )

    assert isinstance(result, TranslationArtifact)
    assert result.original_text_ref == source_ref
    assert result.chunk_id == chunk_id
    assert (
        result.translated_artifact.sha256
        == hashlib.sha256(b"Laporan tidak mengonfirmasi 12 kg pada 2024-05-06.").hexdigest()
    )
    assert result.independent_evidence is False
    assert {"NEGATION", "NUMERALS", "UNITS", "TENSE"} <= set(result.risk_flags)
    assert gateway.calls[0].contract.operation == "TRANSLATION"
    assert gateway.calls[0].contract.response_language == Language(code="id", status="DECLARED")
    assert publisher.records[0]["source_ref"] == source_ref
    assert publisher.records[0]["chunk_id"] == chunk_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selected", "reason"),
    [(False, "language unknown"), (True, None), (True, "")],
)
async def test_translation_needs_explicit_selective_reason_before_provider_attempt(
    selected: bool, reason: str | None
) -> None:
    source = "Original passage."
    source_ref = ArtifactRef(
        artifact_id=new_id("ART"),
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        size_bytes=len(source),
    )
    output_ref = ArtifactRef(artifact_id=new_id("ART"), sha256="2" * 64, size_bytes=1)
    gateway = TranslationGatewayStub(output_ref)
    service = SelectiveTranslator(
        gateway,
        output_reader=TranslationReaderStub(b"{}"),
        publisher=TranslationPublisherStub(),
    )
    with pytest.raises(ValueError, match="TRANSLATION_REQUIRES_SELECTIVE_ROUTING_REASON"):
        await service.translate(
            task=task(source_ref),
            source_ref=source_ref,
            chunk_id=new_id("CHK"),
            source_text=source,
            source_language=Language(code="unknown", status="UNKNOWN"),
            target_language=Language(code="id", status="DECLARED"),
            selected_for_translation=selected,
            selection_reason=reason,
            prompt_version="translate-v2",
        )
    assert gateway.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("defect", ["wrong-hash", "wrong-quote", "incomplete-coverage"])
async def test_translation_rejects_provenance_or_quote_mismatch(defect: str) -> None:
    source = "Original passage with a date 2025."
    source_ref = ArtifactRef(
        artifact_id=new_id("ART"),
        sha256=hashlib.sha256(source.encode()).hexdigest(),
        size_bytes=len(source),
    )
    document = {
        "source_sha256": source_ref.sha256,
        "target_language": {"code": "id", "status": "DECLARED"},
        "segments": [
            {
                "source_start": 0,
                "source_end": len(source),
                "original_text": source,
                "translated_text": "Terjemahan.",
            }
        ],
    }
    if defect == "wrong-hash":
        document["source_sha256"] = "a" * 64
    elif defect == "wrong-quote":
        document["segments"][0]["original_text"] = "changed passage"
    else:
        document["segments"][0]["source_start"] = 1
        document["segments"][0]["source_end"] = len(source)
        document["segments"][0]["original_text"] = source[1:]
    raw = json.dumps(document).encode()
    gateway = TranslationGatewayStub(
        ArtifactRef(artifact_id=new_id("ART"), sha256="3" * 64, size_bytes=len(raw))
    )
    service = SelectiveTranslator(
        gateway,
        output_reader=TranslationReaderStub(raw),
        publisher=TranslationPublisherStub(),
    )

    with pytest.raises(ValueError):
        await service.translate(
            task=task(source_ref),
            source_ref=source_ref,
            chunk_id=new_id("CHK"),
            source_text=source,
            source_language=Language(code="en", status="DECLARED"),
            target_language=Language(code="id", status="DECLARED"),
            selected_for_translation=True,
            selection_reason="User-approved, targeted translation.",
            prompt_version="translate-v2",
        )


def test_aligned_translation_rejects_gaps_and_overlapping_spans() -> None:
    with pytest.raises(ValueError):
        AlignedTranslation(
            source_sha256="a" * 64,
            target_language=Language(code="id", status="DECLARED"),
            segments=(
                {
                    "source_start": 0,
                    "source_end": 2,
                    "original_text": "ab",
                    "translated_text": "xy",
                },
                {
                    "source_start": 3,
                    "source_end": 5,
                    "original_text": "de",
                    "translated_text": "zw",
                },
            ),
        )
