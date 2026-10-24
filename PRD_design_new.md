# Design System & UI/UX Requirements (`PRD_design.md`)

Dokumen ini adalah spesifikasi desain terikat untuk tampilan kanvas web interaktif Knowledge Graph (`src/drx/visualizer/graph_view.py`). Seluruh komponen UI, warna, tipografi, dan tata letak WAJIB mengikuti standar spesifikasi di bawah ini.

---

## 1. Layout Geometry & Hierarchy Strategy

* **Orientasi:** Vertikal / Top-to-Bottom (`direction: 'UD'` / Up-Down).
* **Root Node (Level 0):** Berada di posisi paling ATAS tengah kanvas.
* **Level Progression:** Node bertambah ke BAWAH. Cabang-cabang node menyebar secara HORIZONTAL memanfaatkan lebar layar laptop.
* **Maksimum Kedalaman:** Level 0 hingga Level 5/6.
* **Spacing Rules (Vis.js Configuration):**
  * `levelSeparation`: `130px` (Jarak vertikal antartingkat/level).
  * `nodeSpacing`: `180px` (Jarak horizontal antar-node dalam satu level agar tidak bertumpukan).
  * `treeSpacing`: `200px` (Jarak antar-cabang utama).
  * `physics`: `false` (Kalkulasi fisik dimatikan total agar node terkunci rapi di gridnya).

---

## 2. Typography System & Hierarchy

Gunakan *font-family* tunggal: `Inter`, `-apple-system`, `BlinkMacSystemFont`, `"Segoe UI"`, `Roboto`, `sans-serif`.

| Komponen | Font Size | Font Weight | Line Height | Letter Spacing | Color Code |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Header Status / Stats** | `11px` | `400` (Regular) | `1.2` | `0.02em` | `#94A3B8` |
| **Button Text** | `12px` | `600` (SemiBold) | `1.0` | `0.01em` | `#FFFFFF` |
| **Node Type Badge** | `9px` | `600` (SemiBold) | `1.0` | `0.05em` | Dynamic Badge Text |
| **Node Short Title** | `12px` | `500` (Medium) | `1.3` | `0.01em` | `#F8FAFC` |
| **Inspector Drawer Header** | `15px` | `600` (SemiBold) | `1.3` | `0.00em` | `#F8FAFC` |
| **Inspector Drawer Body** | `12px` | `400` (Regular) | `1.5` | `0.00em` | `#CBD5E1` |

---

## 3. Color Palette System (Dark Neon Theme)

### Base & Canvas Colors
* **Canvas Background:** `#0B0F17` (Dark Slate / Near Black)
* **Canvas Dot Grid:** `#1E293B` (Ukuran titik 1.5px, jarak antar-titik 20px)
* **Edge Line Color:** `#334155` (Garis relasi antar-node, tebal 1.5px, style `cubicBezier`)
* **Edge Line Active:** `#38BDF8` (Garis menyala/glow saat node terikat diklik)

### Node Card Colors
* **Default Card Background:** `#151C2C` (Dark Glass / Solid Slate)
* **Default Card Border:** `#334155` (Tekstur border tipis 1px, `border-radius: 8px`)
* **Selected Card Border:** `#38BDF8` (Biru Neon) + `box-shadow: 0 0 12px rgba(56, 189, 248, 0.35)`

### Badge Category Colors (Semantic Types)

| Tipe Node | Background | Text Color | Border Color |
| :--- | :--- | :--- | :--- |
| **INDIVIDU** | `#2E1065` | `#C084FC` | `#581C87` |
| **INSTANSI / PERUSAHAAN** | `#451A03` | `#FCD34D` | `#78350F` |
| **AKTOR / THREAT** | `#450A0A` | `#FCA5A5` | `#7F1D1D` |
| **VERIFIED / BUKTI** | `#022C22` | `#6EE7B7` | `#064E3B` |
| **KUERI / SARAN** | `#082F49` | `#38BDF8` | `#0369A1` |

---

## 4. Node Component Architecture & Dimensions

* **Dimensi Kartu Node:**
  * **Width:** Fixed `180px` (Mencegah kartu melebar atau merusak tata letak).
  * **Padding:** `8px 10px`.
  * **Height:** Auto (menyesuaikan isi, maks `55px`).
* **Format Teks Judul Node:**
  * Wajib berupa `short_label` (Maksimal 1–3 kata / 20 karakter).
  * Jika teks melebihi 20 karakter, secara otomatis dipotong dengan *ellipsis* (`...`).
  * Teks panjang/lengkap TIDAK BOLEH ditampilkan di kanvas, melainkan dipindah ke Hover Tooltip & Inspector Drawer.

---

## 5. Interaksi & Floating Controls

### Floating Action Header Bar (Paling Atas Tengah)
* **Background:** `#0F172A` dengan `backdrop-filter: blur(8px)` dan border `#1E293B`.
* **Kontras Tombol:**
  * **Tombol Pivot Selected:** Background `#2563EB` (Solid Blue), Teks `#FFFFFF`.
  * **Tombol Expand Pool:** Background `#0D9488` (Solid Teal), Teks `#FFFFFF`.
  * **Tombol Stop & Finalize:** Background `#DC2626` (Solid Red), Teks `#FFFFFF`.
* **Bug Fix Spacing:** Teks status (token/URL/waktu) ditempatkan di kontainer terpisah di bawah tombol dengan jarak `margin-top: 6px` agar teks tidak menumpuk.

### Hover Tooltip (Arahkan Kursor)
* **Delay:** `200ms`.
* **Tampilan:** Box melayang kecil di samping kursor.
* **Isi:** Judul Lengkap Entitas + Deskripsi Ringkas 1 Kalimat + Jumlah Sumber/Bukti.
* **Styling Tooltip:** Background `#1E293B`, Border `#475569`, Text `#F8FAFC` (`11px`).

### Inspector Drawer (Klik Node - Slide-in dari Kanan)
* **Width:** `360px` (Fixed side panel).
* **Background:** `#0F172A`, Border-left `#1E293B`.
* **Komponen:**
  * **Header:** ID Node (`n16`), Nama Lengkap, Badge Tipe, & Confidence Score.
  * **Section 1:** Bukti Grounding (Kutipan teks asli dari dokumen).
  * **Section 2:** Daftar Sumber (URL interaktif yang bisa diklik).
  * **Section 3:** Saran Critic (Alasan kenapa node ini direkomendasikan untuk di-pivot).