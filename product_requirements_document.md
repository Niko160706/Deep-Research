# Product Requirements Document (PRD)
## Deep Research AI Agent (Investigative Multi-Agent System)

---

## 1. Executive Summary & Goals

### 1.1 Overview
The **Deep Research AI Agent** is an autonomous web-research system inspired by systematic detective investigation methodologies. It converts high-level, ambiguous, or multi-faceted research queries into structured investigation plans, autonomously gathering, verifying, and synthesizing information from the web.

### 1.2 Core Philosophy: Dual-Stream Architecture
To minimize token costs and operate efficiently within hardware constraints, the system separates **Mechanical Data Collection** (100% Python, 0 LLM tokens) from **Cognitive Analysis** (LLM-driven evaluation, extraction, and synthesis).

### 1.3 Key Constraints & Hardware Targets
- **Max RAM Allocation:** $\le \text{8 GB}$
- **Primary LLM Provider:** DeepSeek API (`deepseek-chat` and `deepseek-reasoner`), fully configurable per-role via YAML.
- **Local Embeddings:** Local ONNX/INT8 quantized embeddings (`BGE-small-en-v1.5` / `BGE-small-zh-v1.5`).
- **Cost Discipline:** Zero token consumption for fetching, scraping, cleaning, or initial keyword filtering.

---

## 2. System Architecture & Pipeline

```
  [User Query]
       │
       ▼
[LLM Router] ───────────────────┐
       │                        │
       ▼                        ▼
[Super Agent (Def1)]   [Empirical Lexical Survey]
       │                 (Python / DDG HTML)
       └──────────┬─────────────┘
                  ▼
        [Super Agent (Def2)]
                  │
                  ▼
     [Hybrid Queue / Stack Memory]
     ├── Layer 1: FIFO (Breadth-First)
     └── Layer 2+: LIFO (Depth-First)
                  │
                  ▼
         [AutoSearch & Scraper] ◄── [Local Cache Check]
         (aiohttp -> Playwright Fallback)
                  │
                  ▼
        [HTML Cleaner & BM25]
                  │
       ┌──────────┴──────────────────────┐
       ▼                                 ▼
[Parquet Bypass]               [Bifurcation Router]
(Raw Text Backup)                        │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [Entity Stream]                             [Heuristic Stream]
                   │                                           │
         [LLM Fact Extractor]                        [Argument Card Extractor]
         (JSON + Provenance)                                   │
                   │                                 [INT8 Local Embedding]
         [Fuzzy Matching / rapidfuzz]                          │
                   │                                           ▼
                   ▼                                    [Vector DB / FAISS]
       [Knowledge Graph / NetworkX]                            │
                   │                                           ▼
                   ▼                                  [Heuristic Critic]
            [Entity Critic]                                    │
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                             [Cycle Evaluator & Circuit]
                                         │
                         ┌───────────────┴───────────────┐
                         │                               │
                (Sub-query Generated)           (Stop Conditions Met)
                         │                               │
                         ▼                               ▼
                 [Stack Deduplicator]             [Context Packer]
                         │                               │
                         └──► [Push to Stack]            ▼
                                                 [Synthesis Agent]
                                                         │
                                                         ▼
                                               [Markdown + JSON Sidecar]
```

### 2.1 Stage 1: Initial Planning & Empirical Survey
1. **LLM Router:** Evaluates the primary query and assigns research mode (`Entity` vs. `Heuristic` vs. `Hybrid`) per sub-query.
2. **Super Agent (Def1):** Generates an initial research hypothesis and high-level strategy.
3. **Empirical Lexical Survey (Python):** Executes a lightweight search (DuckDuckGo HTML) without LLMs to retrieve top titles and snippets.
4. **Super Agent (Def2):** Refines the research plan based on real-world snippets and decomposes it into discrete, actionable sub-queries.

### 2.2 Stage 2: Memory & Task Scheduling
- **Hybrid Queue/Stack Memory:**
  - **Layer 1 (FIFO / Queue):** Executes Breadth-First Search (BFS) to map out broad perspectives.
  - **Layer 2+ (LIFO / Stack):** Executes Depth-First Search (DFS) when critical anomalies, leads, or contradictions are identified.
- **Sub-Query Deduplication:** Python calculates string/embedding similarity against previously executed sub-queries before pushing to Stack.

### 2.3 Stage 3: Mechanical Data Collection (The Muscle)
1. **Cache Check:** Check local run history for cached URLs or sub-query answers.
2. **AutoSearch:** Ranks candidate URLs using heuristic domain scoring and keyword relevance.
3. **Parallel Scraper:**
   - **Primary:** `aiohttp` for lightweight static web pages.
   - **Fallback:** Headless `Playwright` with a strict `Semaphore(2)` concurrency limit for JavaScript-heavy or protected sites.
4. **HTML Cleaner & BM25 Filter:** Strips DOM junk, ads, and navigation. Filters paragraphs using BM25 relevance scoring relative to the active sub-query.
5. **Parquet Bypass:** Raw clean text is saved directly to `.parquet` storage in parallel, bypassing LLM processing to avoid token output bloat.

### 2.4 Stage 4: Data Processing & Storage Bifurcation

#### Path A: Entity Track (Pelacakan Entitas / Detektif)
- **LLM Fact Extractor:** Extracts facts into strict JSON schema containing:
  - `subject`, `predicate`, `object`
  - `confidence_score` ($0.0 - 1.0$)
  - `provenance` (source URL, extracted timestamp)
- **Entity Resolution:** Python (`rapidfuzz`) matches and normalizes aliases before graph insertion.
- **Knowledge Graph:** Maintained locally using `NetworkX` and exported as GraphML.

#### Path B: Heuristic Track (Pencarian Solusi / Riset Scientific)
- **Argument Card Extractor:** LLM extracts structured Argument Cards containing:
  - `claim`, `evidence`, `methodology`, `limitations`
- **Vector Indexing:** Arguments are embedded using local INT8 `BGE-small` models and stored in an on-disk `FAISS` index.

---

## 3. Evaluation, Feedback Loops, & Safety Breakers

### 3.1 Cognitive Sparring (Critics)
- **Entity Critic (Graph):** Analyzes the NetworkX graph for timeline contradictions, conflicting affiliations, or missing alibis/links.
- **Heuristic Critic (Vector):** Evaluates logical fallacies, weak causality, or gaps in retrieved evidence.

### 3.2 Cycle Evaluator & Stop Conditions
A Python-driven orchestrator checks loop continuation criteria after each iteration:
1. **Max Depth Limit:** `depth >= MAX_DEPTH` (default: $4$).
2. **Token Budget Limit:** `total_tokens >= MAX_TOKEN_BUDGET` (default: $150,000$ tokens).
3. **Wall-Clock Timeout:** `elapsed_time >= MAX_RUNTIME` (default: $600$ seconds).
4. **Novelty Rate Threshold:** `new_claims / total_claims < 0.10` ($<10\%$ new information added in the last loop).
5. **Stack Exhaustion:** Stack is empty (`len(stack) == 0`).

---

## 4. Data Models & Storage Layout

### 4.1 Per-Run Directory Structure
Every research session is isolated inside the `runs/` directory:

```text
runs/<run_id>/
├── parquet/
│   └── raw_documents.parquet       # Direct bypass raw text storage
├── graph.graphml                   # Exported NetworkX graph
├── vectors.faiss                   # FAISS vector index
├── vectors_metadata.json           # Argument card lookup metadata
├── report.md                       # Final synthesis report
├── report_sidecar.json             # Structured entities, edges, & sources
└── manifest.json                   # Execution logs, metrics, & token counters
```

### 4.2 Context Packer
Before invoking the final Synthesis Agent, the **Context Packer (Python)** aggregates data under a strict token allowance:
1. Top $N$ Graph Triples (summarized network structure).
2. Top $K$ Reranked Argument Cards from FAISS.
3. Relevant verbatim text snippets retrieved from `.parquet` files for tone and quote context.

### 4.3 Final Output Specification
The system produces two primary artifacts:
1. **`report.md`:** Comprehensive Markdown report containing Executive Summary, Key Findings, Entity/Logic Analysis, Contradictions, and an explicit **Sources & Provenance** section with verified URLs.
2. **`report_sidecar.json`:** Machine-readable JSON export of all discovered entities, edges, claims, confidence scores, and source URLs.

---

## 5. Technical Requirements & Dependencies

### 5.1 Environment Requirements
- **Python Version:** $\ge 3.10$
- **Operating System:** Linux / macOS / Windows WSL2

### 5.2 Dependency Matrix (`requirements.txt`)

```text
# Core Orchestration & Data Models
pydantic>=2.0.0
pyyaml>=6.0
rich>=13.0.0

# Async Web Scraper & HTML Cleaning
aiohttp>=3.9.0
playwright>=1.40.0
beautifulsoup4>=4.12.0
trafilatura>=1.6.0
rank-bm25>=0.2.2

# Storage & Mathematics
pandas>=2.1.0
pyarrow>=14.0.0
numpy>=1.24.0

# Local Embeddings & Vector Search
sentence-transformers>=2.2.2
faiss-cpu>=1.7.4
onnxruntime>=1.16.0

# Graph Engine & Entity Matching
networkx>=3.2.1
rapidfuzz>=3.5.0

# API Client
openai>=1.3.0
```

### 5.3 Model Role Configuration (`config/models.yaml`)

```yaml
default_provider: "deepseek"

roles:
  router:
    model: "deepseek-chat"
    temperature: 0.1
  def1_planner:
    model: "deepseek-chat"
    temperature: 0.3
  def2_decomposer:
    model: "deepseek-chat"
    temperature: 0.2
  extractor:
    model: "deepseek-chat"
    temperature: 0.0
  critic:
    model: "deepseek-reasoner"
    temperature: 0.2
  synthesis:
    model: "deepseek-reasoner"
    temperature: 0.3

budget_limits:
  max_tokens: 150000
  max_depth: 4
  max_runtime_seconds: 600
  playwright_max_tabs: 2
```

---

## 6. Phased Implementation Plan

### Phase 1: Skeleton & Worker Muscle (Current Focus)
- Set up directory layouts, YAML configuration loaders, and Rich-based logger.
- Implement `AutoSearch`, async `aiohttp` scraper, `Playwright` fallback with semaphore, HTML cleaning, BM25 filtering, and `Parquet` direct write.
- Verify zero-LLM web collection pipeline.

### Phase 2: Knowledge Engine & Vector Indexing
- Build local INT8 `BGE-small` embedding worker and `FAISS` vector store manager.
- Implement `NetworkX` graph manager and `rapidfuzz` entity resolution module.
- Build LLM JSON extraction parsers with Pydantic validation.

### Phase 3: Loop Evaluation & Cognitive Sparring
- Implement `EntityCritic` and `HeuristicCritic` modules using `deepseek-reasoner`.
- Build `CycleEvaluator` with token budget, depth, runtime, and novelty threshold enforcement.
- Implement sub-query deduplication logic.

### Phase 4: Context Packer & Synthesis
- Build `ContextPacker` to assemble multi-store contexts within strict token constraints.
- Implement final `SynthesisAgent` producing `report.md` and `report_sidecar.json`.
- Execute end-to-end testing against self-verification (Golden Dataset) targets.