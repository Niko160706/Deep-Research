# Product Requirements Document (PRD) - UI/UX Canvas & Graph Design
## Deep Research AI Agent (Investigative Node Canvas)

---

## 1. Design Philosophy & Core Principles

### 1.1 Aesthetic & Feel
- **Visual Style:** Minimalist Dark/Light Hybrid Industrial (Inspired by n8n workflow canvas and Notion minimalist dark mode).
- **Theme Palette:**
  - Background: Off-black / Charcoal `#0F172A` or Dark Slate `#1E293B`.
  - Grid Pattern: Subtle dot-grid background (`#334155`, spacing: 20px) to give an interactive canvas feel.
  - Primary Accent (Nodes): Electric Violet `#8B5CF6` or Cyan `#06B6D4`.
  - Card/Panel Fill: `#1E293B` with 1px border `#475569`.
  - Typography: Clean sans-serif (`Inter`, `System-UI`, or `Fira Code` for node IDs).

### 1.2 Performance & Weight Targets
- **Zero-Heavy Framework:** Pure HTML5 + CSS3 + Vanilla JS / PyVis / Vis.js. No heavy React/Vue build steps required.
- **Max Bundle Size:** < 1.5 MB total (loads instantly in any browser).
- **Memory Footprint:** < 50 MB browser RAM.

---

## 2. Interactive Canvas Specifications (n8n-Inspired Flow)

### 2.1 Node Hierarchy & Visual Coding
Every entity extracted into the NetworkX graph must be rendered as a card-style Node on the canvas with distinct visual states:

1. **Node Types (Color-Coded Badges):**
   - **PejabatPublik / Individu:** Purple Accent (`#A855F7`) with user/person icon.
   - **InstansiPemerintah / Lembaga:** Blue Accent (`#3B82F6`) with building icon.
   - **Perusahaan / Vendor:** Amber Accent (`#F59E0B`) with brief/office icon.
   - **DokumenResmi / SK / LHKPN:** Emerald Accent (`#10B981`) with file-text icon.
   - **Peristiwa / Pelantikan:** Rose Accent (`#F43F5E`) with flag/event icon.

2. **Edge / Link Styling (Verification Status):**
   - **Solid Green Line (`#22C55E`):** Verified relation (supported by $\ge 2$ independent sources in Parquet).
   - **Dashed Slate Line (`#94A3B8`):** Single-source claim (unverified / 1 source).
   - **Solid Red Line (`#EF4444`):** Contradictory relation or flagged by Critic.

### 2.2 Interactive Multi-Selection & Pivot UI
- **Multi-Node Selection:** User can click multiple node boxes or drag a selection box around multiple nodes.
- **Branch Action Bar (Top Control Panel):**
  - **Button 1: "Pivot Selected Nodes" (Execute Batch Query):** Generates new sub-queries for all selected nodes in one go.
  - **Button 2: "Expand Document Pool":** Pulls next $N$ chunks without making new LLM queries.
  - **Button 3: "Stop & Generate Final Report":** Ends searching and triggers single-pass LLM Report Synthesis.

---

## 3. Sidebar & Inspection Modal (Provenance Panel)

### 3.1 Node Inspector (On-Click Panel)
Clicking any Node or Edge opens a lightweight slide-over sidebar showing:
1. **Node ID & Normalized Name:** (e.g., `[n12] PT Jaya Abadi`).
2. **Verification Badge:** (e.g., `Verified (2 Sources)`).
3. **Evidence Snippets (Grounding):**
   - Direct quotes from scraped text.
   - Clickable Source URLs (opens source in new tab).
   - Extraction Timestamp & Confidence Score.

---

## 4. Output File & Storage

- Rendered File Location: `runs/<run_id>/graph_interactive.html`
- Sidecar Data Source: `runs/<run_id>/report_sidecar.json`
- NetworkX Source: `runs/<run_id>/graph.graphml`