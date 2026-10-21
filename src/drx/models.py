"""Pydantic v2 data schemas for the Deep Research Agent.

Every object crossing a module boundary is a validated model here. Keeping
schemas in one place guarantees mechanical modules (0 tokens) and cognitive
modules (LLM) speak the same language.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """Research track. HYBRID routes one task through both extraction paths."""

    ENTITY = "entity"
    HEURISTIC = "heuristic"
    HYBRID = "hybrid"


class TaskOrigin(str, Enum):
    """Where a sub-query came from — drives queue vs stack placement."""

    INITIAL = "initial"   # produced by Def2 (breadth-first)
    CRITIC = "critic"     # produced by a critic (depth-first)
    EXPAND = "expand"     # user-requested level-1 document pool expansion
    RETRY = "retry"


class Layer(str, Enum):
    """Scheduling layer inside the hybrid frontier."""

    QUEUE = "queue"   # FIFO — breadth-first mapping
    STACK = "stack"   # LIFO — depth-first dive


class QueryTask(BaseModel):
    """Unit of work inside the frontier. `depth` belongs to the TASK, not
    the loop — this is what lets BFS/DFS share one scheduler safely."""

    id: str
    parent_id: str | None = None
    query: str
    mode: Mode = Mode.ENTITY
    depth: int = 1
    origin: TaskOrigin = TaskOrigin.INITIAL
    priority: int = 0
    hypothesis: str | None = None
    # Decomposition metadata (for the transparency tree + 70/30 strategy).
    rationale: str = ""
    backend: str = "hybrid"
    is_anchored: bool = False
    context: list[str] = Field(default_factory=list)
    score: float = 0.0
    result_count: int = 0
    status: str = "pending"  # pending | searched | empty | failed | skipped
    # Canvas hop level: 0 root, 1 initial/Expand, parent.level + 1 per pivot.
    level: int = 1
    # Optional per-task search depth (0 = AutoSearch default). Expand uses a
    # larger limit to pull results beyond the initial top-N without a new LLM
    # query planner call.
    search_limit: int = 0


class SearchResult(BaseModel):
    """One candidate URL returned by a search backend, pre-scoring."""

    url: str
    title: str = ""
    snippet: str = ""
    domain: str = ""
    score: float = 0.0
    reason: str = ""


class Document(BaseModel):
    """Raw (or cleaned) page body, ready for storage or extraction."""

    url: str
    title: str = ""
    text: str = ""
    fetched_at: float = 0.0
    method: Literal["aiohttp", "playwright", "pdf", "cache"] = "aiohttp"
    lang: str | None = None


class SourceRef(BaseModel):
    """Provenance pointer attached to extracted facts.

    Phase 3 grounding: every triple carries the chunk id (``source_id``) and a
    verbatim ``quote`` that a Python validator must find in that chunk before
    the fact is allowed into the graph (anti-hallucination, zero extra tokens).
    """

    url: str
    title: str | None = None
    extracted_at: float = 0.0
    source_id: str = ""
    quote: str = ""


class Fact(BaseModel):
    """Entity-track extraction: subject-predicate-object triple."""

    subject: str
    predicate: str
    object: str
    confidence_score: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: list[SourceRef] = Field(default_factory=list)


class EvidenceCard(BaseModel):
    """Heuristic-track extraction: a structured argument chunk."""

    chunk_id: str
    url: str
    claim: str = ""
    evidence: str = ""
    methodology: str | None = None
    limitations: str | None = None
    score: float = 0.0


class Gap(BaseModel):
    """A contradiction or hole found by a critic, driving a new sub-query."""

    description: str
    suggested_query: str
    mode: Mode = Mode.ENTITY
    source: Literal["entity_critic", "heuristic_critic"] = "entity_critic"


class ResearchPlan(BaseModel):
    """Def1 output: hypothesis + strategy + seed queries for the survey."""

    hypothesis: str
    strategy: str = ""
    seed_queries: list[str] = Field(default_factory=list)


class SurveyResult(BaseModel):
    """Empirical lexical survey output (Python-only, 0 tokens)."""

    query: str
    hits: list[SearchResult] = Field(default_factory=list)


class Budget(BaseModel):
    """Hard limits echoed in `config/settings.yaml` (PRD §3.2)."""

    max_tokens: int = 150000
    max_depth: int = 4
    max_runtime_seconds: int = 600
    max_urls: int = 200
    novelty_floor: float = 0.10
    stagnation_limit: int = 3


class Report(BaseModel):
    """Final artifact: Markdown body + machine-readable sidecar."""

    run_id: str
    query: str
    markdown: str
    sidecar: dict = Field(default_factory=dict)
