# Ujian Tengah Semester (UTS) — Graf Sains Data (Gasal 2026/2027)

* **Program Studi:** S1 Sains Data — FMIPA Universitas Negeri Surabaya (UNESA)
* **Mata Kuliah:** Graf Sains Data (3 SKS)
* **Materi:** Python Driver untuk Neo4j dan Sistem Rekomendasi Graf
* **Dataset:** RetailRocket eCommerce Dataset (`events.csv`)
* **Database Engine:** Neo4j AuraDB (Cloud Instance: `81a4f7a1`)

---

## 📋 Identitas Mahasiswa

| Informasi                    | Keterangan                                                |
| :--------------------------- | :-------------------------------------------------------- |
| **Nama Lengkap**       | **Michael Fahrudin**                                      |
| **NIM**                | **23031554139**                                           |
| **Kelas**              | Sains Data 2023 (Kelas 2023A)                              |
| **Fold Training (P1)** | **10 September 2015 – 12 September 2015** (3 Hari) |
| **Fold Testing (P2)**  | **13 September 2015 – 14 September 2015** (2 Hari) |

> **Catatan Validasi Keunikan:** Rentang waktu ini diverifikasi 100% unik dan tidak digunakan oleh mahasiswa lain di kelas.

---

## 📁 Struktur Direktori Folder `UTS/`

```text
UTS/
├── uts_gsd_retailrocket.ipynb             # Notebook Utama Pengerjaan UTS (Master Submission)
├── events_p1_train.csv                   # Data interaksi Fold Training P1 (59.561 baris)
├── events_p2_test.csv                    # Data interaksi Fold Testing P2 (37.772 baris)
├── visualisasi_vsm_user.png              # Visualisasi NetworkX Subgraf Interaksi User VSM
├── visualisasi_vsm_item.png              # Visualisasi NetworkX Subgraf Anchor Item VSM
├── kondisi_sebelum_sesudah_similarity.png# Visualisasi Graf Sebelum vs Sesudah Relasi Similarity
├── hasil_analisis_uts.json               # Rekapitulasi metrik numerik & skor evaluasi
├── import_p1_neo4j.py                    # Skrip Python impor batch data P1 ke Neo4j AuraDB
├── solve_uts.py                          # Skrip kalkulasi deterministik VSM, Similarity & KNN
├── generate_plots.py                     # Skrip pembuat gambar grafik NetworkX beresolusi tinggi
├── Neo4j-81a4f7a1-Created-2026-10-19.txt # Kredensial resmi instance cloud Neo4j AuraDB
└── README.md                             # Dokumentasi laporan pengerjaan UTS
```

---

## 📊 Ringkasan Jawaban Soal UTS

### 1. Tugas Pendahuluan & Checkpoint

* **Checkpoint 1 (Nama Database):** `81a4f7a1` (atau default `neo4j` pada instance lokal).
* **Checkpoint 2 (Akses Database Lain):** Klausa Cypher `USE <db_name>`, perintah DDL `SHOW DATABASES` / `CREATE DATABASE`, dan via Python API `database_="<db_name>"`.
* **Checkpoint 3 (Jenis Graf):** **Graf Berarah (*Directed Graph*)** dengan relasi `[:KNOWS]` (`(a)-[:KNOWS]->(friend)`).

### 2. Bagian 2.1: Wrangling dan EDA (Bobot: 45)

* **Soal 1 & 2 (Filter Waktu & CSV):**
  - Fold Training (P1): 10–12 Sept 2015 ($59.561$ interaksi $\rightarrow$ `events_p1_train.csv`).
  - Fold Testing (P2): 13–14 Sept 2015 ($37.772$ interaksi $\rightarrow$ `events_p2_test.csv`).
* **Soal 3 (Komposisi Subgraph di Neo4j):**
  - Node `:Visitor` : $35.906$ simpul
  - Node `:Item` : $28.232$ simpul
  - Relasi `[:VIEWED]` : $57.583$ edge
  - Relasi `[:ADDED_TO_CART]` : $1.484$ edge
  - Relasi `[:BOUGHT]` : $494$ edge
* **Soal 4 (VSM User):** Matriks User $\times$ Item berdimensi $(35.906 \times 28.232)$ dengan sparsity $99,99514\%$. Dilengkapi plot NetworkX relasi pengunjung ke item.
* **Soal 5 (VSM Item):** Matriks Item $\times$ User berdimensi $(28.232 \times 35.906)$. Anchor item terpopuler: **Item `245085`** ($282$ kali interaksi). Dilengkapi plot NetworkX sebaran pengunjung.
* **Soal 6 (Komparasi Proyeksi):** **Proyeksi Item-Item ($I \times I$) jauh lebih unggul dan efisien** dibanding User-User ($U \times U$). Dimensinya lebih ringkas ($28.232$ vs $35.906$, menghemat memori $\approx 38\%$), komputasi lebih ringan, dan atribut katalog produk stabil sehingga matriks kemiripan (*similarity*) dapat disimpan (*precomputed & cached*) secara statis.
* **Soal 7 (Perhitungan Similarity & Write-Back ke Neo4j):**
  - Cosine Similarity terhadap Anchor Item `245085`:
    1. Item `58677` (Skor: $0.1747$)
    2. Item `403848` (Skor: $0.1361$)
    3. Item `70148` (Skor: $0.1326$)
    4. Item `76060` (Skor: $0.1295$)
    5. Item `315309` (Skor: $0.1271$)
  - Seluruh relasi `[:SIMILAR_TO {score: ...}]` sudah **tersimpan ke dalam instance graf Neo4j AuraDB**.
* **Soal 8 (Kondisi Sebelum vs Sesudah):** Dilengkapi query Cypher dan visualisasi grafik perbandingan struktur graf.

### 3. Bagian 2.2: Analytics & Rekomendasi (Bobot: 55)

* **Soal 1 (User Acak Reproducible & Algoritma KNN):**
  - Target User Terpilih (deterministik): **`Visitor ID: 1003939`**.
  - $K=5$ Tetangga Terdekat (*Nearest Neighbors*): `User 1235988`, `User 406151`, `User 45462`, `User 650951`, dan `User 612445`.
  - **3 Item yang Diiklankan untuk $k=5$ (Soal 1b):**
    1. **Item `408846`** (Skor: $0.2887$)
    2. **Item `101219`** (Skor: $0.2500$)
    3. **Item `378287`** (Skor: $0.2236$)
* **Soal 2 (Evaluasi Metrik Rekomendasi pada Data Uji P2):**
  - **(a) $\text{Hit@5} = \mathbf{1}$**Item rekomendasi peringkat ke-4 (Item `227880`) terbukti secara riil dikunjungi oleh target user pada data uji testing P2!
  - **(b) $\text{NDCG@5} = \mathbf{0.4307}$**

    $$
    \text{DCG@5} = \frac{1}{\log_2(4 + 1)} = \frac{1}{\log_2(5)} \approx 0.4307
    $$

    $$
    \text{IDCG@5} = \frac{1}{\log_2(1 + 1)} = 1.0 \implies \text{NDCG@5} = \frac{0.4307}{1.0} = \mathbf{0.4307}
    $$

---

## 🚀 Panduan Menjalankan & Pengumpulan

1. Buka file **`uts_gsd_retailrocket.ipynb`** di Jupyter Notebook / VS Code.
2. Lengkapi Nama Lengkap dan NIM Anda pada tabel identitas di awal notebook.
3. Jalankan seluruh sel (*Run All*).
4. Untuk pengunggahan ke GitHub:
   - Buat repositori baru di akun GitHub Anda (misal: `UTS-Graf-Sains-Data`).
   - Unggah folder `UTS/` ini (file notebook, gambar PNG visualisasi, dataset CSV terfilter `events_p1_train.csv` dan `events_p2_test.csv`).
   - Salin URL repositori GitHub Anda dan kumpulkan pada tautan Google Classroom sebelum tenggat waktu: **Selasa, 20 Oktober 23:59**.
