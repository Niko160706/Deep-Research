# Deep Research AI Agent

Investigative multi-agent system with a **Dual-Stream Pipeline**: mechanical
data collection (100% Python, 0 LLM tokens) separated from cognitive analysis
(LLM). Inspired by systematic detective investigation.

## Install

```bash
python3 -m venv .venv && . .venv/bin/activate

# 1. deps: mechanical pipeline + Phase 3 extras
#    (pyvis, tiktoken, rapidfuzz, pdfplumber, pypdf)
pip install -r requirements.txt
#    minimal/offline saja: pip install -r requirements-base.txt

# 2. PyTorch CPU-only (AVOID the ~5 GB CUDA wheel)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. ML stack (embeddings + vector index + browser lib)
pip install -r requirements-ml.txt

# 4. Chromium ONLY (not Firefox/WebKit)
playwright install --with-deps chromium
```

Verify CPU torch: `python -c "import torch; print(torch.cuda.is_available())"` → `False`.

## Configuration

Copy your API keys into `.env` (lihat `.env.example`; file `.env` sudah di-gitignore):

```bash
TAVILY_API_KEY=tvly-...            # https://tavily.com/
EXA_API_KEY=...                    # https://exa.ai/
BRAVE_SEARCH_API_KEY=...           # https://brave.com/search/api/
SEARXNG_URL=http://localhost:8080  # SearXNG lokal (docker)
TOR_PROXY=socks5://127.0.0.1:9050  # TOR SOCKS proxy (docker)

# Phase 2 LLM hierarchy (auto: Groq gpt-oss-120b -> Groq qwen3.8-27b -> Gemini)
GROQ_API_KEY=...                   # PRIMARY gpt-oss-120b + SECONDARY qwen3.8-27b
GEMINI_API_KEY=...                 # TERTIARY / last resort (kuota ~20 request/hari)
# OPENROUTER_API_KEY=...           # explicit-only (tidak masuk rantai auto)
#   Groq free tier: ITPM ~7.000 input token/menit + OTPM ~1.000 output token/menit
#   -> prompt dipangkas otomatis, output critic/extractor 350-400 token, jeda 2 detik.
```

### Local search stack (opsional, untuk SearXNG + TOR)

```bash
docker compose up -d     # SearXNG (200MB) + Valkey (50MB) + TOR (50MB)
docker compose down
```

### Search backends (`config/settings.yaml` → `search.backend`)

| `backend` | Sumber | Catatan |
|---|---|---|
| `hybrid` *(default)* | SearXNG + Tavily + Exa (paralel) | Dedup gate + region lock |
| `searxng` | SearXNG lokal | Meta-search tanpa key; auto-off setelah koneksi gagal/timeout/5xx |
| `exa` | Exa.ai API | Semantic search, butuh key |
| `tavily` | Tavily API | LLM-oriented, butuh key |
| `wikipedia` | MediaWiki API | Andal, tanpa key |
| `failover` | Tavily → Brave → Wikipedia → DDG → Bing | Otomatis pindah saat gagal |

**Region lock:** `search.region` (default `id`), di-override via CLI `--region id|us|jp|global`.

## Run

```bash
# offline, deterministic smoke test (no network, mock search)
python main.py --query "who is Alan Turing" --mock-search

# real search (hybrid: SearXNG + Tavily + Exa) + region lock
python main.py --query "Niko Rajani Syahputra Pane" --region id

# debug logging
python main.py --query "..." --verbose

# skip Phase 2 (agentic critic + LLM synthesis) — offline / hemat token
python main.py --query "..." --no-llm

# checkpoint interaktif: kanvas + pivot/expand/selesai tiap siklus
python main.py --query "..." --interactive

# Mode Web App: server lokal + browser, aksi kanvas lewat REST (pivot/expand/stop)
python main.py --query "..." --web
python main.py --query "..." --web --port 8123 --no-prompt   # tanpa prompt

# wizard tanpa --query otomatis interaktif bila stdin TTY; prompt awal menawarkan
# [1] Mode CLI atau [2] Mode Web App
python main.py
```

Dual-mode: pada TTY (wizard atau `--interactive`) prompt pertama memilih mode
eksekusi — **[1] CLI** (checkpoint terminal) atau **[2] Web App** (server lokal
`http://localhost:8000`, browser dibuka otomatis, mindmap ter-update real-time).
`--web` memilih mode Web App tanpa prompt (boleh digabung `--no-prompt` untuk
CI). Bila port 8000 dipakai, server otomatis mencoba 8001..8009.

Checkpoint interaktif hanya muncul bila `stdin` TTY dan `--no-prompt` tidak
dipakai. Di pipeline non-TTY / CI / `--no-prompt`, run berjalan seperti biasa
tanpa pernah menunggu input atau menghapus file.

### Internship Hunter (Mode 2)

Wizard interaktif sekarang dimulai dengan pilihan modul utama:
`[1] General Investigation (Sistem Lama)` atau
`[2] Internship Hunter (Sistem Baru)`, lalu sub-menu pendekatan A/B/C
(`[A]` role spesifik, `[B]` open discovery, `[C]` skill-based matching).
Untuk otomasi:

```bash
python main.py --mode internship --internship-approach role \
  --internship-role "AI Engineer" --no-prompt

python main.py --mode internship --internship-approach skill \
  --internship-skills "Python, Go, SQL, Docker" --no-prompt

# offline deterministic smoke of the new module
python main.py --mode internship --internship-approach role \
  --internship-role "AI Engineer" --mock-search --no-llm --no-prompt
```

Planner menghasilkan **role murni** (contoh `AI Engineer`, bukan
`AI Engineer Intern Indonesia`); lokasi/moda hanya metadata.  URL target
dibangun langsung per sumber (A–G: JobStreet, Glints, Kalibrr, ITS Career
Center, MSIB, LinkedIn-via-Tavily/Exa-dork, NodeFlair).  Setiap halaman
diambil via Playwright (`page.wait_for_timeout(3000)`) bila browser
tersedia, dengan fallback HTTP + BeautifulSoup; kartu lowongan diekstrak dari
selector stabil (mis. `article[data-automation="normalJob"]`,
`a[itemprop="name"]`, `div.inner-box`, `div.border.border-gray-200`) dan bila
selector kosong, teks `<body>` dipecah baris untuk dikurasi LLM. NodeFlair
selalu memakai Playwright + stealth.

Lalu mode menulis `report.md`, `report_llm.md`, `internship_results.json`,
`graph_interactive.html`, dan `internship_source_diagnostics.json` di folder
run.  Tabel lowongan siap apply berada di bagian paling atas `report_llm.md`;
mode kerja dinormalisasi ke `WFH`/`WFO`/`Hybrid` dan hanya kandidat dengan
`is_valid_internship=true` yang disimpan.  Tabel diagnostik sumber
(SUCCESS/EMPTY_RESULT/BLOCKED/TIMEOUT + HTTP + latensi + raw_chars_scraped +
item + alasan) dicetak di terminal pada akhir eksekusi.  Mindmap memakai
hierarki `Magang [Role] -> Perusahaan -> Posisi` dengan badge
`KUERI/SARAN` (`#082F49`).  Seluruh kode baru berada di `src/drx/internship/`
dan tidak mengubah loop Mode 1.

Output lands in `runs/<run_id>/`:

```
report.md                 # Markdown report (ekstraktif, Phase 1)
report_llm.md             # laporan investigasi LLM (Phase 2, bila API key tersedia)
report_sidecar.json       # {target_entity, verified_facts, disambiguated_entities, sources_used}
                          #   (+ phase1: ringkasan ekstraktif, tidak dihapus)
manifest.json             # budget / counters (+ ringkasan critic & llm_synthesis)
graph.graphml             # knowledge graph (incremental, stable node IDs n1, n2, ...)
graph_interactive.html    # mindmap canvas (hierarchical UD, dark neon - PRD_design.md)
vectors_metadata.json     # argument cards
parquet/raw_documents.parquet   # raw cleaned text bypass
```

## Test

```bash
pytest
```

## Architecture

```
Router -> Def1 -> Lexical Survey -> Def2 -> Hybrid Queue/Stack
   -> AutoSearch -> Scraper -> Cleaner -> Parquet Bypass
   -> [Entity: facts -> Graph] | [Heuristic: cards -> Vector]
   -> Critics -> Evaluator -> Synthesis
```

Phase 1 (this tree) implements the full **mechanical** pipeline for real
(0 LLM tokens). Phase 2 adds the **LLM Report Synthesis Engine**
(`drx/brain/llm_synthesizer.py`): the cleaned parquet documents are scored in
pure Python, ≤ 15–20 segments (~30k token cap) are sent to Google AI Studio
(`gemini-3.8-flash`, fallback chain `gemini-3.7-flash → gemini-3.5-flash`) —
or an OpenRouter/Groq fallback — and the final structured investigation report
is written to `runs/<run_id>/report_llm.md`. **Failover dua lapis**: setiap
model Gemini yang 404 (retire) / 503 (high demand) dilewati ke model berikutnya,
dan bila seluruh Gemini gagal, sintesis otomatis pindah provider ke OpenRouter
lalu Groq (qwen) selama key-nya ada. **Auto-downscale**: bila provider menolak
ukuran prompt (mis. Groq free tier ITPM 7.000 — error 413 `Request too large`),
konteks otomatis dipangkas (18 → 9 → 6 segmen) dan request diulang, jadi laporan
tetap jadi tanpa perlu upgrade tier. Tanpa API key, tahap ini dilewati otomatis
tanpa menggagalkan run.

Phase 2 juga menghidupkan **agentic critic loop** (`drx/brain/critic.py`):

```
Phase 1 scraping -> Critic (OSINT Fact Verifier, JSON verdict)
   -> is_sufficient?  ya  -> Synthesis
   ->                  tidak (dan loop < max_loops=2)
        -> sub-kueri gap: search -> scrape -> clean -> parquet
        -> Critic ulang ...
```

* Verdict JSON (pydantic): `is_sufficient`, `critique_summary`,
  `suggested_sub_queries` — dikirim dengan *native JSON mode* (retry tanpa mode
  itu bila provider menolaknya).
* Sub-kueri gap tetap melewati `Cleaner.clean()` — strict `--context` dan gate
  identitas berlaku sama seperti Phase 1.
* Setelah sintesis, LLM juga mengembalikan blok `json` terstruktur yang
  divalidasi Python: setiap `source_url` di-clamp ke URL bukti yang benar-benar
  dipakai (anti-halusinasi). Hasilnya disimpan ke `report_sidecar.json` dan
  `verified_facts`-nya dimasukkan ke NetworkX (`graph.graphml`).
* Batas loop/token di `settings.yaml` (`llm.critic_enabled`,
  `llm.critic_max_loops`, `llm.critic_max_sub_queries`, `llm.critic_max_segments`,
  `llm.critic_max_input_tokens`). `--no-llm` mematikan critic + sintesis
  sekaligus; tanpa API key kedua tahap dilewati dengan pesan singkat.

## Phase 3 — Interactive Graph Research

Phase 3 mengubah graph menjadi pusat investigasi: setiap siklus pencarian
menghasilkan triple ber-grounding, graph bertambah, kanvas interaktif dirender,
lalu terminal berhenti di checkpoint untuk keputusan user.

```
Run -> P0 Query Planner (~400 token, 1x)
   -> [cycle] search -> scrape -> clean -> grounded triple extraction (Map-Reduce)
        -> GraphStore (rapidfuzz + stable IDs) -> graph.graphml -> graph_interactive.html
        -> CHECKPOINT: [1] pivot by node ID  [2] expand level-1 pool
                       [3] finish          [4] list all nodes (CLI)
   -> (3) LLM synthesis SEKALI -> report_llm.md + report_sidecar.json
```

* **Level per cabang**: level anak = level parent + 1 (bukan nomor ronde
  global). Pivot dari node level 1 menghasilkan level 2, pivot dari level 2
  menghasilkan level 3, dst. — tanpa batas level; hanya budget yang membatasi.
  Node yang sudah dikenal mempertahankan level pertama saat ditemukan.

* **Batch guard 3.0–3.5k token** (`drx/brain/batching.py`): semua input LLM
  (planner, ekstraksi, critic, sintesis) dipotong ke
  `llm.batch_max_input_tokens` (default **3200**) memakai `tiktoken`
  (`cl100k_base`, fallback `len/4`). Parquet besar dipecah per batch
  (Map-Reduce) dan MMR (`mmr_indices`) dipakai untuk memilih segmen yang tidak
  redundan — anti 413 Groq / 503 Gemini.
* **Ekstraksi triple ber-grounding** (`drx/brain/extractors.py`): LLM hanya
  mengusulkan `{subject, predicate, object, source_id, short_quote}`; Python
  memverifikasi entitas **dan** kutipan benar-benar ada di chunk teks. Triple
  yang gagal grounding dibuang tanpa biaya token, sisanya dikonversi ke `Fact`
  dengan provenance kaya (URL + kutipan).
* **Critic hanya baca JSON triple** (`analyze_graph_gaps`): bukan teks parquet.
  Output JSON berisi `is_sufficient`, `contradictions`,
  `suggested_sub_queries`, dan **Saran ID Node untuk Pivot**
  (`suggested_nodes`, `node_id` wajib ada di triple — halusinasi ID dibuang).
* **Entity resolution** (`rapidfuzz`): `"PT Jaya Abadi"` ≡ `"PT. Jaya Abadi"`,
  typo `"Gajah Mada"` ≡ `"Gadjah Mada"`; node memakai ID stabil `n1, n2, ...`
  (aman dijadikan referensi pivot). Predikat fungsional (jabatan, alamat, NIK,
  …) yang bertentangan otomatis ditandai `contradiction` dan digambar merah.
* **Checkpoint terminal** setelah tiap siklus: estimasi biaya + token kumulatif
  + sisa token/URL/waktu, saran node critic beserta ID dan level
  (`n12 - PT Jaya Abadi (level 2)`), lalu pilihan:
  - `[1]` pivot pada ID node (boleh banyak, pisah koma) — daftar tervalidasi
    ditampilkan dan harus dikonfirmasi `y/n` sebelum dikirim;
  - `[2]` expand — deep-search kueri awal dengan limit lebih besar + sudut
    kueri baru (profil/karier/berita/...), sehingga **benar-benar menambah
    dokumen dan node level 1** (bukan sekadar membaca ulang parquet);
  - `[3]` selesai & susun laporan final (dengan konfirmasi);
  - `[4]` cetak semua node per level (bukan hanya top-5 saran).
  Mode interaktif **tidak** berhenti otomatis saat frontier kosong atau
  novelty stagnan; hanya `[3]` yang mengakhiri. Batas token/URL/waktu tetap
  dicek dan user diminta konfirmasi bila ingin melanjutkan melewatinya.
* **Laporan final sekali**: LLM hanya menulis `report_llm.md` saat user memilih
  `[3] Selesai` (idempotent — `main.py` tidak menulis dua kali).
* **Sintesis 100% ter-grounding**: prompt LLM final **hanya** memuat fakta
  Knowledge Graph yang sudah tervalidasi (subject–predicate–object + sumber)
  beserta chunk/kutipan yang melandasinya. Chunk parquet mentah yang tidak
  pernah diekstrak/diverifikasi tidak pernah dikirim ke LLM sintesis.
* **Kanvas** (`drx/visualizer/graph_view.py`) mengikuti `PRD_design.md`
  (Dark Neon): latar `#0B0F17` + dot-grid `#1E293B`, kartu node `#151C2C`
  berborder `#334155` dengan lebar tetap **180px**, teks `#F8FAFC`
  (`Inter`/system-ui), dan badge kategori — INDIVIDU `#C084FC`, INSTANSI
  `#FCD34D`, AKTOR `#FCA5A5`, VERIFIED `#6EE7B7`, KUERI `#38BDF8`.
  Setiap node menyimpan **`short_label`** (1-3 kata, maks 20 karakter) dari
  extractor LLM; `label` lengkap, `type`, `sources`, `grounding`, dan
  `critic_reason` tetap utuh. Kartu kanvas hanya menampilkan `short_label`;
  hover 200ms menampilkan tooltip (label lengkap + ringkasan 1 kalimat +
  `Verified X Sumber`), dan klik membuka **Inspector Drawer 360px** berisi ID,
  badge tipe, kutipan grounding, URL sumber interaktif, serta saran critic.
  Layout **Hierarchical Top-to-Bottom** (`direction: 'UD'`,
  `sortMethod: 'directed'`, `levelSeparation: 130`, `nodeSpacing: 180`,
  `treeSpacing: 200`, physics off) dengan level node eksplisit per cabang:
  `0` root query, `1` fakta pencarian awal/Expand, dan `parent level + 1`
  untuk setiap pivot (tanpa batas level). Kurva `cubicBezier
  forceDirection: 'vertical'`; relasi mundur (*cross-edge*) disembunyikan
  default dan bisa ditampilkan lewat toggle toolbar. Semantik edge dari
  `PRD_design.md` tetap berlaku (hijau *verified*, putus-putus satu sumber,
  merah kontradiksi). Node terpilih mendapat border/glow `#38BDF8`; pemilihan
  memakai **klik toggle** (bisa banyak), lalu tray `ID · label · level` +
  modal konfirmasi sebelum satu batch pivot dikirim. Node yang sudah pernah
  dipivot ditandai di tray/modal. Status token/URL/waktu berada di baris
  terpisah **di bawah tombol kontrol** agar tidak tumpang tindih. PyVis
  dimuat dari CDN (~30KB HTML); bila `pyvis` tidak terpasang, render jatuh ke
  HTML statis yang tetap informatif. Batas kanvas default dinaikkan ke
  **600 node** (`visualizer.max_nodes`).
* **Mode Web App** (`drx/web/server.py`, stdlib `http.server`): server lokal
  loopback menyajikan `graph_interactive.html` + endpoint REST `/api/state`,
  `/api/graph`, `/api/pivot`, `/api/expand`, `/api/stop`. Klik node = toggle
  seleksi; tray + modal konfirmasi memastikan satu POST berisi semua ID.
  Pivot/Expand/Stop mengeksekusi siklus berikutnya di orchestrator (perintah
  dikirim lintas-thread via `asyncio.Queue`), dan halaman polling `/api/graph`
  tiap 2 detik sehingga mindmap, saran critic, riwayat pivot, serta statistik
  token/URL/waktu/biaya ter-update real-time.
* **Hierarki LLM**: PRIMARY Groq `openai/gpt-oss-120b` → SECONDARY Groq
  `qwen/qwen3.8-27b` → TERTIARY Gemini. Gemini diletakkan paling akhir karena
  kuota harian free tier hanya ~20 request; OpenRouter hanya dipakai bila
  `llm.provider: openrouter` dipilih eksplisit.
* **OTPM guard (Groq free tier ~1000)**: output Critic & Extraction dibatasi
  350–400 token (`llm.critic_max_output_tokens`, `llm.extraction_max_output_tokens`)
  dan ada jeda **minimal 2 detik** antar request Groq
  (`llm.provider_min_interval_seconds`; nilai konfigurasi di bawah 2 detik
  tetap dinaikkan ke hard floor 2 detik).
* **SearXNG circuit breaker**: SearXNG lokal dimatikan otomatis untuk sisa
  sesi setelah connection refused, timeout, respons 5xx, atau body JSON rusak —
  query pivot/expand berikutnya tidak menunggu layanan yang sama berulang kali.
* Setelan terkait di `settings.yaml`: `llm.batch_max_input_tokens`,
  `llm.planner_enabled`, `llm.extraction_enabled`, `llm.critic_reads_triples`,
  `llm.use_mmr`, `llm.mmr_lambda`, `llm.groq_fallback_model`,
  `llm.provider_min_interval_seconds`, `llm.critic_max_output_tokens`,
  `llm.extraction_max_output_tokens`, `web.host`, `web.port`,
  `web.auto_open_browser`.
