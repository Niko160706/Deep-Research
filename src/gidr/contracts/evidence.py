"""Evidence candidates retain original locators; schemas are not adjudication."""

from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from gidr.contracts.common import (
    ArtifactRef,
    ChunkId,
    ClaimId,
    EvidenceId,
    Name,
    Record,
    SourceId,
    Text,
)
from gidr.contracts.sources import TextLocator
from gidr.domain.states import Verdict


class Claim(Record):
    claim_id: ClaimId
    proposition: Text
    claim_types: Annotated[
        tuple[
            Literal[
                "STATEMENT",
                "FACTUAL",
                "STATISTICAL",
                "HISTORICAL",
                "ORGANIZATIONAL",
                "SCIENTIFIC",
                "CAUSAL",
                "ALLEGATION",
                "PREDICTION",
                "ANONYMOUS_TESTIMONY",
                "SPECIAL_REVIEW_REQUIRED",
            ],
            ...,
        ],
        Field(min_length=1),
    ]
    scope: Text
    valid_time: Text | None
    definition: Text | None
    original_formulation: Text
    derivation_ref: ArtifactRef
    status: Verdict = Verdict.UNASSESSED


class EvidenceRecord(Record):
    evidence_id: EvidenceId
    claim_id: ClaimId
    chunk_id: ChunkId
    verbatim_excerpt: Text
    original_locator: TextLocator
    locator_verification: Literal["UNCHECKED", "MATCH", "MISMATCH", "UNAVAILABLE"] = "UNCHECKED"
    relation: Literal["SUPPORTS", "REFUTES", "INSUFFICIENT", "CONTEXTUALIZES"]
    context_ref: ArtifactRef
    dependency_source_ids: tuple[SourceId, ...] = ()


class SourceDependency(Record):
    from_source_id: SourceId
    to_source_id: SourceId
    kind: Literal["QUOTES", "DERIVES", "REPOSTS", "INDEPENDENT", "UNKNOWN"]
    checking_method: Name
    evidence_refs: tuple[ArtifactRef, ...] = ()

    @model_validator(mode="after")
    def distinct_sources(self) -> Self:
        if self.from_source_id == self.to_source_id:
            raise ValueError("Source dependence must relate distinct identities")
        return self
