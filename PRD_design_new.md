# Product Requirements Document (PRD) - UI/UX Canvas & Mindmap Design
## Deep Research AI Agent (Hierarchical Mindmap Canvas)

---

## 1. Design Philosophy & Aesthetic (Clean Minimalist Light Theme)

### 1.1 Aesthetic Palette
- **Style:** Clean, Minimalist Industrial Mindmap (Inspired by Notion Light & Modern Flowcharts).
- **Background:** Soft Off-White `#F8FAFC` with subtle dot-grid pattern (`#CBD5E1`, spacing: 20px).
- **Node Cards:** Solid White `#FFFFFF` with subtle border `#E2E8F0` and clean box shadow (`0 4px 6px -1px rgba(0,0,0,0.08)`).
- **Typography:** Dark Slate `#0F172A` for primary text, Sans-Serif (`Inter`, `system-ui`).
- **Accent Badges (Minimalist Accent Only):**
  - Root Input / Prompt: Soft Indigo (`#EEF2FF`, border `#6366F1`, text `#4338CA`).
  - Kueri / Sub-Kueri: Soft Cyan (`#ECFEFF`, border `#06B6D4`, text `#0E7490`).
  - Entitas / Individu: Soft Purple (`#F3E8FF`, border `#A855F7`, text `#6B21A8`).
  - Instansi / Perusahaan: Soft Amber (`#FEF3C7`, border `#F59E0B`, text `#B45309`).

---

## 2. Horizontal Hierarchical Mindmap Layout (Left-to-Right)

### 2.1 Vis.js Layout Configuration
Visualisasi graf WAJIB menggunakan konfigurasi **Hierarchical Layout (Left-to-Right)** agar membentuk alur *mindmap* cabang yang rapi:

```javascript
const options = {
  layout: {
    hierarchical: {
      enabled: true,
      direction: 'LR',        // Left to Right (Kiri ke Kanan)
      sortMethod: 'directed',
      nodeSpacing: 180,
      levelSeparation: 350
    }
  },
  physics: {
    enabled: false            // Matikan simulasi gravitasi agar posisi node stabil
  },
  edges: {
    smooth: {
      type: 'cubicBezier',    // Garis melengkung halus seperti n8n/mindmap
      forceDirection: 'horizontal'
    },
    color: { color: '#94A3B8', highlight: '#3B82F6' },
    width: 2
  }
};
```

### 2.2 Explicit Node Levels & Cross-Edges
- Setiap node membawa atribut `level` eksplisit — bukan hasil inferensi edge:
  - `0` = root query / target utama (satu kolom paling kiri),
  - `1` = fakta dari pencarian awal **dan** gelombang Expand,
  - `N+1` = fakta baru dari pivot pada node level `N` (per cabang).
- Level bersifat **per cabang dan tanpa batas**: pivot node level 1 → anak
  level 2, pivot node level 2 → anak level 3, dst. Bukan nomor ronde global.
- Node yang sudah dikenal mempertahankan level pertama saat ditemukan.
- Edge mundur (`to_level < from_level`, *cross-edge*) **disembunyikan secara
  default** agar alur kiri→kanan tetap dominan. Toolbar menyediakan toggle
  `↔ Cross (N)` untuk menampilkannya; saat tampil, cross-edge memakai garis
  tipis putus-putus `#CBD5E1`.
- Kartu node sengaja minimal: label saja (tanpa ikon/tipe di kartu). ID,
  tipe, level, status verifikasi, derajat, riwayat pivot, dan provenance
  tersedia di tooltip + inspector sidebar.

### 2.3 Selection Feedback, Tray & Pivot Confirmation
- Node terpilih: background & border `#3B82F6`, glow ring `#60A5FA`, border
  tebal (`4px`), teks putih — kontras jelas terhadap kartu putih.
- **Klik node = toggle select/deselect**; tidak perlu Ctrl/drag. Semua node
  yang dipilih tetap terlihat dan node lain tidak disembunyikan.
- **Selection tray** muncul di kiri-atas berisi daftar `ID · label · level`,
  penanda `sudah dipivot`, counter live, tombol hapus per chip, dan `Clear`.
- Tombol pivot menampilkan counter live:
  `⚡ Pivot Selected Nodes (X node terpilih)`.
- Menekan tombol pivot membuka **modal konfirmasi** berisi daftar node
  tervalidasi + total. Satu klik `Kirim Pivot` mengirim **satu** POST
  `/api/pivot` berisi seluruh ID; `Batal` tidak mengirim apa pun.
- Saran critic juga melewati modal konfirmasi sebelum dikirim.
- Seleksi dipertahankan saat kanvas di-refresh oleh polling `/api/graph`.

### 2.4 Budget, Expand & Checkpoint
- Toolbar web dan checkpoint terminal selalu menampilkan:
  token kumulatif + **sisa token/max**, URL + **sisa URL/max**, waktu +
  **sisa waktu/max**, estimasi biaya, dan peringatan `over budget`.
- `Expand Document Pool` menjalankan pencarian baru, bukan membaca ulang
  parquet: deep-search kueri awal dengan limit lebih besar (hasil di luar
  top-N lama) + sudut kueri baru (profil/karier/berita/wawancara/...).
  Seluruh dokumen baru diekstrak sebagai fakta **level 1**, dengan dedup URL.
- Terminal menyediakan `[4] Lihat semua node per level` (bukan hanya top-5
  saran), serta konfirmasi `y/n` untuk pivot, expand, dan stop.
- Mode interaktif tidak auto-stop saat frontier kosong atau novelty stagnan;
  hanya aksi Stop eksplisit yang mengakhiri. Jika batas token/URL/waktu
  terlampaui, CLI meminta konfirmasi lanjut; Web App menampilkan status
  over-budget dan tetap menerima aksi Stop.

---

## 3. Final Synthesis Grounding (Anti-Halusinasi)

Laporan final (`report_llm.md`) **hanya** boleh membaca:

1. fakta Knowledge Graph yang sudah tervalidasi
   (`subject–predicate–object` + `source_id` + `short_quote`), dan
2. chunk/kutipan parquet yang menjadi bukti fakta tersebut.

Chunk parquet mentah yang tidak pernah diekstrak/diverifikasi **tidak boleh**
dikirim ke prompt sintesis. Bila tidak ada fakta KG yang memiliki bukti teks,
sintesis dilewati dengan alasan eksplisit (tanpa fallback ke dump parquet).
