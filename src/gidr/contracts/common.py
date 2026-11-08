"""Strict primitives shared by source, evidence, model and control contracts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Literal, Self, get_args
from uuid import uuid4

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    StringConstraints,
    ValidationError,
    model_validator,
)

Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Name = Annotated[str, StringConstraints(pattern=r"^[a-zA-Z][a-zA-Z0-9_.-]{0,127}$")]
Digest = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]
Nonnegative = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(gt=0)]
Fraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
UTCDateTime = Annotated[AwareDatetime, AfterValidator(lambda dt: dt.astimezone(UTC))]


def freeze_identifiers(value: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(dict(value))


def serialize_identifiers(value: Mapping[str, str]) -> dict[str, str]:
    return dict(value)


ExternalIds = Annotated[
    Mapping[Name, Text],
    AfterValidator(freeze_identifiers),
    PlainSerializer(serialize_identifiers),
]


def id_type(prefix: str) -> StringConstraints:
    return StringConstraints(pattern=rf"^{prefix}-[A-Za-z0-9][A-Za-z0-9_-]{{0,127}}$")


SourceId = Annotated[str, id_type("SRC")]
DocumentId = Annotated[str, id_type("DOC")]
ChunkId = Annotated[str, id_type("CHK")]
ClaimId = Annotated[str, id_type("CLM")]
EvidenceId = Annotated[str, id_type("EVD")]
ArtifactId = Annotated[str, id_type("ART")]
RunId = Annotated[str, id_type("RUN")]
ProjectId = Annotated[str, id_type("PRJ")]
SessionId = Annotated[str, id_type("SES")]
TaskId = Annotated[str, id_type("TASK")]
AttemptId = Annotated[str, id_type("ATT")]
ModelExecutionId = Annotated[str, id_type("MOD")]
QuestionId = Annotated[str, id_type("RQ")]
GrantId = Annotated[str, id_type("GRANT")]
EventId = Annotated[str, id_type("EVT")]
TraceId = Annotated[str, id_type("TRACE")]
CommandId = Annotated[str, id_type("CMD")]
ProposalId = Annotated[str, id_type("PROP")]
DecisionId = Annotated[str, id_type("DEC")]
TranslationId = Annotated[str, id_type("TRN")]
QueryId = Annotated[str, id_type("QUERY")]
CredentialAlias = Literal[
    "provider/typesafe",
    "provider/tavily",
    "provider/exa",
    "provider/gemini",
    "provider/groq",
    "provider/siliconflow",
    "provider/openalex",
    "provider/github",
]

IdFamily = Literal[
    "SRC",
    "DOC",
    "CHK",
    "CLM",
    "EVD",
    "ART",
    "RUN",
    "PRJ",
    "SES",
    "TASK",
    "ATT",
    "MOD",
    "RQ",
    "GRANT",
    "EVT",
    "TRACE",
    "CMD",
    "PROP",
    "DEC",
    "TRN",
    "QUERY",
]


def new_id(family: IdFamily) -> str:
    """Generate identity once at creation; serializing never changes an ID."""
    if family not in get_args(IdFamily):
        raise ValueError("Unknown ID family")
    return f"{family}-{uuid4().hex}"


class Contract(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        validate_default=True,
        allow_inf_nan=False,
        revalidate_instances="always",
    )
    schema_version: Literal["1.0"] = "1.0"


class Record(Contract):
    run_id: RunId
    created_at_utc: UTCDateTime


class ValueState(StrEnum):
    KNOWN = "KNOWN"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    FAILED = "FAILED"


class ObservedValue[T](Contract):
    state: ValueState
    value: T | None = None
    reason_code: Name | None = None

    @model_validator(mode="after")
    def check_state(self) -> Self:
        if (self.state == ValueState.KNOWN) != (self.value is not None):
            raise ValueError("Only KNOWN metadata has a value")
        if self.state == ValueState.FAILED and self.reason_code is None:
            raise ValueError("Failed observation requires a reason code")
        return self


class Language(Contract):
    code: Annotated[
        str, StringConstraints(pattern=r"^(?:unknown|mixed|[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*)$")
    ]
    status: Literal["DECLARED", "DETECTED", "UNKNOWN", "MIXED", "FAILED"]

    @model_validator(mode="after")
    def check_code(self) -> Self:
        if self.code == "unknown" and self.status not in {"UNKNOWN", "FAILED"}:
            raise ValueError("Unknown language must remain unknown or failed")
        if self.code == "mixed" and self.status != "MIXED":
            raise ValueError("Mixed language requires MIXED status")
        if self.status in {"UNKNOWN", "FAILED"} and self.code != "unknown":
            raise ValueError("Unknown or failed detection cannot invent a language")
        if self.status == "MIXED" and self.code != "mixed":
            raise ValueError("MIXED status requires mixed language")
        return self


class ArtifactRef(Contract):
    artifact_id: ArtifactId
    sha256: Digest
    size_bytes: Nonnegative


class ContractInputError(ValueError):
    """Public validation failure with paths/types only, never raw input."""


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def parse_contract[M: Contract](model: type[M], payload: str | bytes) -> M:
    """Load explicit v1 JSON; future migrations must be explicitly registered."""
    try:
        data = json.loads(payload, object_pairs_hook=unique_object)
    except (ValueError, UnicodeDecodeError):
        raise ContractInputError("Invalid contract JSON") from None
    if not isinstance(data, dict) or data.get("schema_version") != "1.0":
        raise ContractInputError("Missing or unsupported schema_version")
    try:
        return model.model_validate_json(payload)
    except ValidationError as exc:
        codes = sorted({error["type"] for error in exc.errors(include_input=False)})
        raise ContractInputError("Contract validation failed: " + ", ".join(codes)) from None
