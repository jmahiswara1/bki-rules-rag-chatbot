# RAG Chatbot Practice Project

Proyek ini dibuat untuk latihan magang AI/LLM + RAG secara bertahap. Fokusnya bukan sekadar "chatbot jadi", tapi memahami alur lengkap dari PDF mentah sampai jawaban berbasis konteks dokumen.

## Tujuan Belajar

Setiap implementasi di repo ini mewakili satu tahap penting pada sistem RAG:

1. Setup proyek Python yang rapi dan mudah dipelajari.
2. Ekstraksi teks dari PDF.
3. Chunking dokumen agar cocok untuk retrieval.
4. Embedding dan penyimpanan ke vector database.
5. Retrieval chunk paling relevan dari pertanyaan user.
6. Generasi jawaban dengan LLM.
7. Chat CLI sederhana.
8. Evaluasi manual dengan daftar pertanyaan.

## Struktur Proyek

```text
rag-chatbot/
|- data/
|  |- eval_questions.sample.json
|  |- raw/
|  `- processed/
|- src/rag_chatbot/
|  |- chunking.py
|  |- config.py
|  |- embeddings.py
|  |- llm.py
|  |- pdf_loader.py
|  |- pipeline.py
|  |- retrieval.py
|  `- vector_store.py
|- ingest.py
|- chat.py
|- evaluate.py
`- pyproject.toml
```

## Setup

1. Buat virtual environment:

```powershell
py -3.12 -m venv .venv
```

2. Aktifkan virtual environment:

```powershell
.venv\Scripts\Activate.ps1
```

3. Install dependency:

```powershell
py -3.12 -m pip install -e .
```

4. Salin file environment:

```powershell
Copy-Item .env.example .env
```

5. Siapkan Ollama dan pull model sesuai isi `.env`, contoh:

```powershell
ollama pull qwen2.5:7b
```

6. Salin PDF kamu ke folder `data/raw/`, misalnya `data/raw/dokumen.pdf`.

## Variabel Lingkungan

Proyek ini default menggunakan Ollama lokal. Semua setting utama bisa diatur lewat `.env`.

Jika ingin setting lewat file, salin `.env.example` menjadi `.env` lalu ubah nilainya sesuai kebutuhan.

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
EMBEDDING_MODEL=intfloat/multilingual-e5-large
CHROMA_COLLECTION=pdf_chunks_e5_large
CHUNK_SIZE=1200
CHUNK_OVERLAP=200
TOP_K=8
MAX_RETRIEVAL_DISTANCE=1.10
FALLBACK_MAX_RETRIEVAL_DISTANCE=1.30
RETRIEVAL_DISTANCE_MARGIN=0.20
MIN_CONTEXT_MATCHES=1
FINAL_CONTEXT_K=4
MIN_KEYWORD_OVERLAP=0.00
KEYWORD_GATE_MIN_RATIO=0.20
CONSERVATIVE_MODE=1
MIN_QUOTE_TOKEN_OVERLAP=0.30
# Optional: untuk rate limit HF lebih longgar
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
# Optional: reranker untuk menaikkan akurasi retrieval (latensi naik)
# USE_CROSS_ENCODER_RERANKER=1
# RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

Keterangan penting:
- Ganti model LLM cukup dengan ubah `OLLAMA_MODEL` (contoh: `qwen2.5:7b`, `mistral`, `gemma`).
- Ganti embedding model cukup dengan ubah `EMBEDDING_MODEL`.
- Untuk saat ini provider LLM masih Ollama (belum support OpenAI/Anthropic tanpa ubah kode).
- Kamu juga bisa set env langsung di terminal, tetapi `.env` lebih praktis untuk dipakai ulang.

Opsional jika ingin set env langsung dari PowerShell (tanpa `.env`):

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="qwen2.5:7b"
$env:EMBEDDING_MODEL="intfloat/multilingual-e5-large"
```

## Cara Menjalankan

### 1. Ingest dokumen PDF

```powershell
py -3.12 ingest.py --pdf data/raw/dokumen.pdf
```

Yang terjadi di tahap ini:
- PDF dibaca per halaman.
- Teks dibersihkan agar tidak terlalu berisik.
- Teks dipecah menjadi chunk kecil.
- Tiap chunk diubah menjadi embedding.
- Hasilnya disimpan ke ChromaDB.

### 2. Chat dengan dokumen

```powershell
py -3.12 chat.py
```

Yang terjadi di tahap ini:
- Pertanyaan user diubah menjadi embedding query.
- Sistem mengambil chunk paling relevan dari vector store.
- Chunk itu dimasukkan ke prompt.
- LLM menjawab berdasarkan konteks yang ditemukan.

### 3. Evaluasi sederhana

1. Salin `data/eval_questions.sample.json` menjadi file baru, misalnya `data/eval_questions.json`.
2. Isi daftar pertanyaan dan expected answer berdasarkan dokumenmu.
3. Jalankan:

```powershell
py -3.12 evaluate.py --questions data/eval_questions.json
```

## Penjelasan Implementasi Per Tahap

### 1. Inisialisasi Proyek Python

Tujuan:
- membuat struktur yang rapi
- memisahkan logika inti dari script CLI

Input:
- dependency dan konfigurasi proyek

Proses:
- package `src/rag_chatbot/` menyimpan logika inti
- file `ingest.py`, `chat.py`, dan `evaluate.py` menjadi titik masuk yang mudah dipanggil

Output:
- repo yang gampang dibaca dan dikembangkan

Kenapa seperti ini:
- kalau semua kode ditaruh di satu file, akan cepat membingungkan saat proyek mulai bertambah

### 2. Ekstraksi PDF

Tujuan:
- mengambil teks mentah dari PDF

Input:
- satu file PDF

Proses:
- buka PDF dengan `PyMuPDF`
- baca isi per halaman
- simpan page number sebagai metadata

Output:
- teks mentah terstruktur per halaman

Kenapa perlu dibersihkan:
- PDF sering punya spasi aneh, line break tidak rapi, atau header/footer berulang
- noise seperti itu bisa mengganggu chunking dan retrieval

### 3. Chunking

Tujuan:
- memecah dokumen panjang menjadi unit kecil yang mudah dicari

Input:
- teks hasil ekstraksi

Proses:
- gabungkan teks antar halaman seperlunya
- potong berdasarkan jumlah karakter
- tambahkan overlap antarchunk

Output:
- daftar chunk dengan isi dan metadata

Kenapa chunk size penting:
- terlalu besar: context bercampur, retrieval kurang presisi
- terlalu kecil: informasi terpotong, jawaban bisa kehilangan konteks

### 4. Embedding dan Vector Store

Tujuan:
- mengubah teks menjadi representasi numerik yang bisa dibandingkan secara semantik

Input:
- daftar chunk

Proses:
- `sentence-transformers` membuat vector untuk setiap chunk
- `ChromaDB` menyimpan vector, teks, dan metadata

Output:
- index vektor yang siap dicari

Istilah penting:
- embedding = representasi angka dari makna teks
- similarity search = mencari vector yang paling mirip dengan query

### 5. Retrieval

Tujuan:
- mengambil beberapa chunk paling relevan dari pertanyaan

Input:
- pertanyaan user

Proses:
- pertanyaan di-embedding
- Chroma mencari top-k chunk terdekat

Output:
- context yang dipakai LLM untuk menjawab

Alur penting:
- query -> embedding query -> similarity search -> top-k results

### 6. Generation dengan LLM

Tujuan:
- menyusun jawaban akhir yang natural dari context retrieval

Input:
- pertanyaan user + chunk relevan

Proses:
- susun prompt yang memaksa model fokus pada dokumen
- kirim ke Ollama

Output:
- jawaban final berbasis dokumen

Perbedaan peran:
- retriever mencari potongan informasi
- generator menyusun jawaban dari potongan itu

### 7. Chat CLI

Tujuan:
- memberi cara tercepat untuk mencoba sistem

Input:
- pertanyaan dari terminal

Proses:
- jalankan retrieval lalu generation

Output:
- jawaban dan sumber dokumen

Kenapa mulai dari CLI:
- lebih cepat dibuat
- fokus belajar arsitektur inti, bukan UI

### 8. Evaluasi Sederhana

Tujuan:
- mengukur apakah sistem sudah cukup membantu

Input:
- daftar pertanyaan dan jawaban referensi

Proses:
- sistem menjawab semua pertanyaan uji
- hasil dibandingkan manual dengan expected answer

Output:
- gambaran awal kualitas sistem

Catatan:
- evaluasi sederhana sudah sangat berguna untuk latihan
- kamu belum perlu framework evaluasi yang rumit di tahap awal

## Pengembangan Lanjutan

Setelah versi ini stabil, kamu bisa lanjut ke:
- frontend `Streamlit`
- multi-PDF retrieval
- eksperimen `chunk_size`, `chunk_overlap`, dan `top_k`
- log evaluasi yang lebih formal

## Troubleshooting

### 1. Warning Hugging Face (HF_TOKEN)

Jika muncul warning seperti unauthenticated request ke HF Hub, sistem tetap bisa jalan.

Opsional untuk mengurangi limit download:
- isi `HF_TOKEN` di `.env`, atau
- set env langsung di terminal.

### 2. Error dimensi embedding (768 vs 1024)

Jika kamu ganti embedding model dari `multilingual-e5-base` (768) ke
`multilingual-e5-large` (1024), collection lama tidak bisa dipakai ulang.

Pilihan solusi:
- Ganti `CHROMA_COLLECTION` ke nama baru, contoh `pdf_chunks_e5_large`.
- Atau hapus index lama lalu ingest ulang dari nol.

### 3. Ingin re-ingest dari nol

Jika ingin menghapus index lama dan membangun ulang vector store:

```powershell
Remove-Item -Recurse -Force data\chroma
py -3.12 ingest.py --pdf data/raw/dokumen.pdf
```

### 4. Ganti model Ollama tapi gagal generate

Pastikan model sudah tersedia di local Ollama:

```powershell
ollama pull <nama-model>
```

Lalu pastikan `.env` memakai nama model yang sama pada `OLLAMA_MODEL`.
