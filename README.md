# Proyek UTS STKI: Boolean Retrieval & Vector Space Model

Proyek ini adalah implementasi sistem temu kembali informasi (STKI) mini untuk memenuhi Ujian Tengah Semester Ganjil 2025/2026 mata kuliah Sistem Temu Kembali Informasi (A11.4703).

- **Nama:** Igdo Ragil Manuel
- **NIM:** A11.2023.14986
- **Dosen:** Abu Salam, M.Kom

## 📝 Tentang Proyek

Sistem ini dirancang untuk melakukan pencarian pada korpus mini berisi 15 dokumen teks fiktif (`.txt`) bertema lowongan kerja dan magang di Semarang dan sekitarnya. Proyek ini mengimplementasikan dua model pencarian fundamental:

1.  **Boolean Retrieval Model:** Pencarian presisi berbasis kata kunci dengan operator `AND`, `OR`, dan `NOT`.
2.  **Vector Space Model (VSM):** Pencarian relevansi berbasis ranking, menggunakan pembobotan **TF-IDF** dan **Cosine Similarity**.

Antarmuka pengguna (UI) untuk sistem ini dibangun menggunakan **Streamlit**.

## ✨ Fitur Utama

Proyek ini mencakup implementasi dari Soal 02 hingga Soal 05:

- **Document Preprocessing**

  - Menggunakan `nltk` untuk _tokenization_ dan _stopword removal_ (daftar NLTK + daftar _custom_).
  - Menggunakan `Sastrawi` untuk _stemming_ Bahasa Indonesia.
  - Melakukan _case folding_ dan normalisasi (menghapus tanda baca & angka) menggunakan `re`.
  - Menyimpan hasil teks bersih ke direktori `data/processed/`.

- **Boolean Retrieval Model**

  - Membangun **Inverted Index** (menggunakan `dict`) dari korpus yang telah diproses.
  - Membangun **Incidence Matrix** (menggunakan `scipy.sparse.lil_matrix`).
  - Mengimplementasikan parser kueri sederhana yang mendukung `AND`, `OR`, dan `NOT`.
  - Dievaluasi menggunakan _Precision_ dan _Recall_ sederhana.

- **Vector Space Model (VSM)**

  - Menggunakan `sklearn.feature_extraction.text.TfidfVectorizer` untuk membuat matriks TF-IDF dari korpus.
  - Menggunakan `sklearn.metrics.pairwise.cosine_similarity` untuk menghitung skor relevansi antara kueri dan dokumen.
  - Menampilkan hasil pencarian teratas (Top-k) beserta _snippet_ dokumen.

- **Evaluasi dan Perbandingan**
  - Membandingkan dua skema pembobotan:
    1.  **Model A:** TF-IDF Standar.
    2.  **Model B:** TF-IDF dengan _Sublinear TF_ (`sublinear_tf=True`).
  - Mengevaluasi kedua model VSM menggunakan metrik `P@5`, `AP@5` (Average Precision), dan `nDCG@5`.

## 🚀 Cara Menjalankan (How to Run) 🚀

Proyek ini dapat dijalankan melalui dua cara:

1. Streamlit: Antarmuka web yang interaktif (direkomendasikan).
2. (Command Line): Interface berbasis terminal.

3. Persiapan Awal (Wajib)
   Sebelum menjalankan, pastikan Anda telah melakukan langkah-langkah berikut:

   1. Install Dependensi: Pastikan Anda berada di virtual environment Anda dan jalankan:
      **pip install -r requirements.txt**
      Ini akan meng-install streamlit, sastrawi, nltk, scikit-learn, dan library lain yang diperlukan.

   2. Unduh Data NLTK: Sistem ini memerlukan data punkt (untuk tokenisasi) dan stopwords (untuk stopword removal) dari NLTK. Jalankan perintah ini di terminal Anda:
      **python -m nltk.downloader punkt stopwords**
      (Atau, jalankan cell pertama di UTS_STKI_A11.2023.14986.ipynb ).

4. Cara Menjalankan (Streamlit / Web UI)
   Ini adalah cara yang disarankan untuk berinteraksi dengan sistem.
   Pastikan Anda berada di root folder proyek Anda.
   Jalankan file main.py yang ada di dalam folder app/ menggunakan Streamlit:

   **streamlit run app/main.py**  
   Streamlit akan otomatis membuka tab baru di browser Anda (biasanya di http://localhost:8501).
   Di dalam aplikasi web, Anda dapat memasukkan kueri dan memilih model (Boolean atau VSM) secara interaktif.

5. Cara Menjalankan (CLI / Terminal)
   Anda juga dapat menjalankan sistem pencarian langsung dari terminal menggunakan search.py.
   Script ini menerima argumen command-line seperti --model, --query, dan --k.
   Struktur Perintah:
   **python src/search.py --model [boolean/vsm] --query "..." [--k N]**

   -model: (Wajib) Pilih boolean atau vsm.
   -query: (Wajib) Masukkan kueri pencarian Anda dalam tanda kutip.

   **Contoh Penggunaan CLI:**

   Contoh 1: Model VSM (Top 3)
   **python src/search.py --model vsm --query "magang data analyst semarang" --k 3**

   Contoh 2: Model VSM (Default k)
   **python src/search.py --model vsm --query "lowongan finance di mranggen"**

   Contoh 3: Model Boolean
   **python src/search.py --model boolean --query "magang AND semarang NOT kendal"**

   Contoh 4: Model Boolean (Operator OR)
   **python src/search.py --model boolean --query "kopi OR barista"**
