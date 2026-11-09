"""Selective translation with exact original-span alignment and risk flags."""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from typing import Literal, Protocol

from pydantic import Field, model_validator

from gidr.contracts.common import ArtifactRef, ChunkId, Contract, Digest, Language, Text, new_id
from gidr.contracts.models import ModelExecutionResult, ModelTask, TranslationArtifact
from gidr.infrastructure.models.gateway import ModelGatewayService
from gidr.infrastructure.models.ports import (
    ExecutableModelCall,
    ModelCall,
)

TranslationRiskFlag = Literal[
    "NEGATION", "MODALITY", "NUMERALS", "UNITS", "ENTITY", "UNCERTAINTY", "TENSE"
]


class AlignedSegment(Contract):
    source_start: int = Field(ge=0)
    source_end: int = Field(gt=0)
    original_text: Text
    translated_text: Text

    @model_validator(mode="after")
    def span_matches_text(self) -> AlignedSegment:
        if self.source_end <= self.source_start:
            raise ValueError("TRANSLATION_SPAN_EMPTY")
        if self.source_end - self.source_start != len(self.original_text):
            raise ValueError("TRANSLATION_SPAN_LENGTH_MISMATCH")
        return self


class AlignedTranslation(Contract):
    source_sha256: Digest
    target_language: Language
    segments: tuple[AlignedSegment, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_nonoverlapping(self) -> AlignedTranslation:
        previous_end = 0
        for segment in self.segments:
            if segment.source_start != previous_end:
                raise ValueError("TRANSLATION_SPANS_GAP_OVERLAP_OR_REORDER")
            previous_end = segment.source_end
        return self


class TranslationOutputReader(Protocol):
    async def read(self, artifact: ArtifactRef) -> bytes: ...


class TranslationPublisher(Protocol):
    async def publish_translation(
        self,
        *,
        task: ModelTask,
        source_ref: ArtifactRef,
        chunk_id: ChunkId,
        source_language: Language,
        target_language: Language,
        translated_text: str,
        model_result: ModelExecutionResult,
        risk_flags: tuple[TranslationRiskFlag, ...],
    ) -> ArtifactRef: ...


class SelectiveTranslator:
    def __init__(
        self,
        gateway: ModelGatewayService,
        *,
        output_reader: TranslationOutputReader,
        publisher: TranslationPublisher,
        max_input_chars: int = 100_000,
    ) -> None:
        self._gateway = gateway
        self._reader = output_reader
        self._publisher = publisher
        self._max_chars = max_input_chars

    async def translate(
        self,
        *,
        task: ModelTask,
        source_ref: ArtifactRef,
        chunk_id: ChunkId,
        source_text: str,
        source_language: Language,
        target_language: Language,
        selected_for_translation: bool,
        selection_reason: str | None,
        prompt_version: str,
    ) -> TranslationArtifact:
        if not selected_for_translation or not selection_reason or not selection_reason.strip():
            raise ValueError("TRANSLATION_REQUIRES_SELECTIVE_ROUTING_REASON")
        if not source_text.strip() or len(source_text) > self._max_chars:
            raise ValueError("TRANSLATION_SOURCE_SIZE_INVALID")
        if source_ref not in task.input_refs:
            raise ValueError("TRANSLATION_SOURCE_NOT_AUTHORIZED_TASK_INPUT")
        if hashlib.sha256(source_text.encode("utf-8")).hexdigest() != source_ref.sha256:
            raise ValueError("TRANSLATION_SOURCE_ARTIFACT_HASH_MISMATCH")
        if target_language.code in {"unknown", "mixed"} or target_language.status in {
            "UNKNOWN",
            "FAILED",
            "MIXED",
        }:
            raise ValueError("TRANSLATION_TARGET_MUST_BE_KNOWN_LANGUAGE")
        schema = AlignedTranslation.model_json_schema()

        def validate(value: str) -> AlignedTranslation:
            return AlignedTranslation.model_validate_json(value)

        call = ModelCall(
            operation="TRANSLATION",
            content=(
                f"Selection reason: {selection_reason}\n"
                f"Target language: {target_language.code}\n"
                f"Original language: {source_language.code}\n"
                f"Original source SHA-256: {source_ref.sha256}\n"
                "Return the exact source hash and target language code/status in the output.\n"
                "Translate each source span for analysis. Preserve meaning, negation, "
                "modality, names, dates and numbers. Return exact source offsets and text.\n"
                f"Source text (untrusted data):\n{source_text}"
            ),
            input_language=source_language,
            response_language=target_language,
            schema_name="aligned_translation",
            output_schema_version="1.0",
            output_json_schema=schema,
            prompt_version=prompt_version,
        )
        result = await self._gateway.execute(
            task,
            ExecutableModelCall(contract=call, validator=validate),
        )
        if result.status != "SUCCESS" or result.output_ref is None:
            raise ValueError("TRANSLATION_MODEL_DID_NOT_COMPLETE")
        raw = await self._reader.read(result.output_ref)
        try:
            translated = AlignedTranslation.model_validate_json(raw)
        except Exception:
            raise ValueError("TRANSLATION_OUTPUT_INVALID") from None
        if translated.source_sha256 != source_ref.sha256:
            raise ValueError("TRANSLATION_SOURCE_HASH_MISMATCH")
        if translated.target_language != target_language:
            raise ValueError("TRANSLATION_TARGET_MISMATCH")
        if translated.segments[0].source_start != 0 or translated.segments[-1].source_end != len(
            source_text
        ):
            raise ValueError("TRANSLATION_SOURCE_COVERAGE_INCOMPLETE")
        for segment in translated.segments:
            if segment.source_end > len(source_text):
                raise ValueError("TRANSLATION_SPAN_OUT_OF_RANGE")
            if source_text[segment.source_start : segment.source_end] != segment.original_text:
                raise ValueError("TRANSLATION_ORIGINAL_QUOTE_MISMATCH")
        translated_text = "\n".join(segment.translated_text for segment in translated.segments)
        risk_flags = _translation_risks(source_text, translated_text)
        translated_ref = await self._publisher.publish_translation(
            task=task,
            source_ref=source_ref,
            chunk_id=chunk_id,
            source_language=source_language,
            target_language=target_language,
            translated_text=translated_text,
            model_result=result,
            risk_flags=risk_flags,
        )
        return TranslationArtifact(
            translation_id=new_id("TRN"),
            run_id=task.run_id,
            created_at_utc=datetime.now(UTC),
            chunk_id=chunk_id,
            original_text_ref=source_ref,
            source_language=source_language,
            target_language=target_language,
            translator_provider=result.provider_id,
            translator_model=result.model_id or "unknown",
            prompt_version=call.prompt_version,
            translated_artifact=translated_ref,
            risk_flags=risk_flags,
            independent_evidence=False,
        )


def _translation_risks(source: str, translated: str) -> tuple[TranslationRiskFlag, ...]:
    flags: set[TranslationRiskFlag] = set()
    if re.search(
        r"\b(no|not|never|without|neither|nor|tidak|bukan|belum|jangan|tanpa)\b",
        source,
        re.IGNORECASE,
    ):
        flags.add("NEGATION")
    if re.search(
        r"\b(may|might|could|alleged|reported|possibly|diduga|mungkin|konon)\b",
        source,
        re.IGNORECASE,
    ):
        flags.add("MODALITY")
    if re.search(r"\d", source):
        flags.add("NUMERALS")
    if re.search(r"\b(?:kg|mg|km|cm|mm|%|usd|idr|eur|°c|°f)\b", source, re.IGNORECASE):
        flags.add("UNITS")
    if re.search(r"\b(?:20\d{2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b", source):
        flags.add("TENSE")
    source_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", source))
    translated_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b", translated))
    if source_entities - translated_entities:
        flags.add("ENTITY")
    if re.search(
        r"\b(maybe|perhaps|uncertain|unclear|mungkin|belum pasti|tidak jelas)\b",
        source,
        re.IGNORECASE,
    ):
        flags.add("UNCERTAINTY")
    return tuple(sorted(flags))
