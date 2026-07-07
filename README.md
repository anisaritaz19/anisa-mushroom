# Anisa — Klasifikasi Jenis Jamur dengan Transfer Learning Xception

**Anisa** adalah aplikasi web untuk mengidentifikasi jenis jamur dari foto,
dibangun menggunakan **Flask**, **TensorFlow/Keras**, dan **Bootstrap 5**.
Model klasifikasi menggunakan pendekatan **transfer learning** dari arsitektur
**Xception** (pretrained ImageNet) dengan **fine-tuning ringan** pada dataset
12 spesies jamur dari Kaggle.

> Proyek ini dibuat untuk keperluan tugas kuliah — pipeline training
> menggunakan **subset kecil dataset (±1.000–2.000 gambar)** dan
> **maksimal 10 epoch** agar proses pelatihan cepat namun tetap merepresentasikan
> alur kerja transfer learning yang benar (freeze → fine-tune → early stopping →
> simpan model terbaik → evaluasi).

---

## Daftar Isi

1. [Struktur Proyek](#struktur-proyek)
2. [Fitur](#fitur)
3. [Arsitektur & Metodologi](#arsitektur--metodologi)
4. [Instalasi Lokal](#instalasi-lokal)
5. [Melatih Model](#melatih-model)
6. [Menjalankan Aplikasi Web](#menjalankan-aplikasi-web)
7. [Menguji Prediksi via CLI/API](#menguji-prediksi-via-cliapi)
8. [Deployment](#deployment)
9. [Troubleshooting](#troubleshooting)
10. [Lisensi & Kredit](#lisensi--kredit)

---

## Struktur Proyek

```
anisa-mushroom-classifier/
├── app.py                  # Aplikasi Flask (routing, upload, prediksi)
├── train.py                # Pipeline lengkap training model
├── predict.py               # Modul load model & inferensi gambar
├── config.py                # Konfigurasi terpusat (path, hyperparameter)
├── requirements.txt          # Daftar dependensi Python
├── Procfile                  # Perintah start untuk Render/Railway (gunicorn)
├── runtime.txt                # Versi Python untuk platform deploy
├── render.yaml                # Blueprint konfigurasi deploy Render.com
├── .gitignore
├── README.md
├── templates/
│   ├── base.html             # Layout dasar (navbar, footer, flash message)
│   ├── index.html            # Halaman utama: upload, preview, hasil prediksi
│   ├── about.html            # Halaman dokumentasi metodologi proyek
│   └── 404.html
├── static/
│   ├── css/style.css         # Tema visual "field guide" bernuansa mikologi
│   ├── js/main.js            # Drag & drop upload + preview gambar
│   └── uploads/              # Folder penyimpanan gambar yang diunggah user
└── model/                    # Hasil training (dibuat oleh train.py):
    ├── best_model.keras
    ├── class_names.json
    ├── training_history.png
    ├── confusion_matrix.png
    └── classification_report.txt
```

---

## Fitur

- **Upload gambar** dengan drag & drop maupun klik untuk memilih file, disertai
  **preview** langsung di browser sebelum dikirim ke server.
- **Prediksi jenis jamur** menggunakan model Xception hasil transfer learning,
  ditampilkan sebagai "kartu spesimen" bergaya buku panduan lapangan (field guide).
- **Confidence score** untuk prediksi utama, ditampilkan sebagai bar visual,
  serta **top-3 kemungkinan lain** beserta persentasenya.
- **Endpoint JSON** (`/api/predict`) untuk pengujian lewat `curl`/Postman atau
  integrasi dengan front-end lain.
- **Desain responsif** (Bootstrap 5) — tampil baik di desktop maupun mobile.
- **Pipeline training reproducible**: download dataset otomatis, subset
  dataset, split train/val/test, augmentasi ringan, EarlyStopping,
  ModelCheckpoint, dan evaluasi lengkap (accuracy, classification report,
  confusion matrix).

---

## Arsitektur & Metodologi

### Dataset

- Sumber: [`mdhasanahmad/12-mushroom-species-dataset`](https://www.kaggle.com/datasets/mdhasanahmad/12-mushroom-species-dataset) (Kaggle).
- Diunduh otomatis oleh `train.py` menggunakan pustaka **`kagglehub`** — tidak
  perlu mengunduh manual.
- **Subset**: ±150 gambar per kelas × 12 kelas ≈ **1.800 gambar** (bisa diatur
  lewat `config.IMAGES_PER_CLASS`), agar training cepat namun tetap cukup
  representatif untuk tugas kuliah.
- Split: **70% train / 15% validation / 15% test**.

### Model

| Tahap | Deskripsi |
|---|---|
| Base model | `Xception(weights="imagenet", include_top=False)` |
| Head | `GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.3) → Dense(12, softmax)` |
| Tahap 1 (Head training) | Base Xception **dibekukan**, hanya head yang dilatih (±4 epoch) |
| Tahap 2 (Fine-tuning ringan) | 30 layer teratas Xception **dibuka** (unfreeze), dilatih dengan learning rate sangat kecil (1e-5), ±6 epoch |
| Regularisasi | `EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)` |
| Penyimpanan | `ModelCheckpoint` menyimpan **hanya model dengan val_accuracy terbaik** ke `model/best_model.keras` |

Total epoch efektif dibatasi **maksimal 10** (4 + 6), namun EarlyStopping dapat
menghentikan proses lebih awal jika validasi tidak lagi membaik.

### Evaluasi

Setelah training, `train.py` otomatis menghasilkan:

- **Accuracy** pada data test (dicetak ke terminal & disimpan ke `.txt`)
- **Classification report** (precision, recall, F1-score per kelas)
- **Confusion matrix** (heatmap PNG)
- **Grafik training history** (akurasi & loss, train vs validation)

---

## Instalasi Lokal

### Prasyarat

- Python 3.10 atau 3.11
- Akun Kaggle (untuk autentikasi `kagglehub`) — lihat langkah di bawah.
- Disarankan memiliki GPU untuk mempercepat training (opsional, CPU tetap bisa
  jalan karena dataset sudah di-subset).

### 1. Clone / salin proyek dan buat virtual environment

```bash
cd anisa-mushroom-classifier
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Autentikasi Kaggle (untuk `kagglehub`)

`kagglehub` membutuhkan kredensial Kaggle API. Cara termudah:

1. Buka https://www.kaggle.com/settings → bagian **API** → **Create New Token**.
   File `kaggle.json` akan terunduh.
2. Letakkan file tersebut di:
   - Linux/Mac: `~/.kaggle/kaggle.json`
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
3. Atau, set environment variable:
   ```bash
   export KAGGLE_USERNAME=your_username
   export KAGGLE_KEY=your_api_key
   ```

`kagglehub` akan otomatis mendeteksi kredensial ini saat `train.py` dijalankan.

---

## Melatih Model

```bash
python train.py
```

Proses yang akan berjalan (lihat log di terminal, ditandai `[1/6]` s.d. `[6/6]`):

1. Mengunduh dataset dari Kaggle via `kagglehub`.
2. Membuat subset (default ±150 gambar/kelas, total ±1.800 gambar).
3. Membagi subset menjadi train/val/test.
4. Menyiapkan data generator (dengan augmentasi ringan untuk data train).
5. Membangun & melatih model (2 tahap: head training → fine-tuning ringan).
6. Mengevaluasi model terbaik pada data test, menyimpan confusion matrix &
   classification report.

Durasi training bergantung perangkat, namun dengan subset kecil dan ≤10 epoch,
proses ini biasanya selesai dalam **10–30 menit di CPU** atau lebih cepat di GPU.

Setelah selesai, folder `model/` akan berisi:

```
model/
├── best_model.keras
├── class_names.json
├── training_history.png
├── confusion_matrix.png
└── classification_report.txt
```

> **Catatan:** `app.py` membaca `model/best_model.keras` dan
> `model/class_names.json`. Aplikasi web **tidak bisa memprediksi** apa pun
> sebelum kedua file ini ada — jalankan `train.py` terlebih dahulu.

---

## Menjalankan Aplikasi Web

```bash
python app.py
```

Buka browser ke **http://localhost:5000**. Untuk mode produksi lokal:

```bash
gunicorn app:app --bind 0.0.0.0:5000
```

---

## Menguji Prediksi via CLI/API

### Command line

```bash
python predict.py path/ke/gambar_jamur.jpg
```

### REST API (JSON)

```bash
curl -X POST -F "image=@path/ke/gambar_jamur.jpg" http://localhost:5000/api/predict
```

Contoh respons:

```json
{
  "predicted_class": "Amanita_muscaria",
  "confidence": 92.35,
  "top_k": [
    ["Amanita_muscaria", 92.35],
    ["Amanita_phalloides", 4.12],
    ["Boletus_edulis", 1.87]
  ]
}
```

---

## Deployment

Proyek ini sudah menyertakan konfigurasi untuk tiga platform populer.
**Pastikan folder `model/` (berisi `best_model.keras` & `class_names.json`)
sudah dilatih dan ikut di-commit / diunggah**, karena aplikasi tidak melatih
ulang model saat deploy.

> Model Xception hasil training bisa berukuran puluhan MB. Jika platform
> membatasi ukuran repo, pertimbangkan **Git LFS** atau unggah model ke
> penyimpanan eksternal (misalnya Google Drive/S3) dan unduh saat build
> (tambahkan langkah download di `buildCommand`).

### Render.com

1. Push proyek ke GitHub.
2. Di Render dashboard: **New +** → **Blueprint** → pilih repo ini.
   Render akan otomatis membaca `render.yaml`.
3. Atau manual: **New +** → **Web Service**, gunakan:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

### Railway

1. Push proyek ke GitHub, lalu buat **New Project** → **Deploy from GitHub repo**.
2. Railway otomatis mendeteksi `requirements.txt` & `Procfile`.
3. Set environment variable `SECRET_KEY` (opsional) di tab **Variables**.

### PythonAnywhere

1. Upload proyek (via Git clone atau upload manual) ke direktori home.
2. Buat virtualenv & install dependencies:
   ```bash
   mkvirtualenv --python=python3.11 anisa-env
   pip install -r requirements.txt
   ```
3. Di tab **Web**, buat aplikasi Flask baru, arahkan **WSGI file** untuk
   mengimpor `app` dari `app.py`:
   ```python
   from app import app as application
   ```
4. Set **Working directory** & **Virtualenv path** sesuai lokasi proyek,
   lalu klik **Reload**.

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `FileNotFoundError: Model tidak ditemukan` | Jalankan `python train.py` terlebih dahulu. |
| `kagglehub` gagal autentikasi | Pastikan `~/.kaggle/kaggle.json` ada atau env var `KAGGLE_USERNAME`/`KAGGLE_KEY` sudah di-set. |
| Training sangat lambat di CPU | Kecilkan `config.IMAGES_PER_CLASS` atau `config.EPOCHS_FINE_TUNE`. |
| Error memori saat training | Kecilkan `config.BATCH_SIZE` (misalnya jadi 8). |
| Model besar sulit di-deploy | Gunakan Git LFS, atau kompres/kuantisasi model (`TFLite`) untuk versi produksi ringan. |

---

## Lisensi & Kredit

- Dataset: [12 Mushroom Species Dataset](https://www.kaggle.com/datasets/mdhasanahmad/12-mushroom-species-dataset) oleh mdhasanahmad (Kaggle).
- Arsitektur dasar: **Xception** (Chollet, 2017), bobot pretrained ImageNet
  dari `tensorflow.keras.applications`.
- Dibangun untuk keperluan tugas kuliah — silakan disesuaikan lebih lanjut
  (nama kelas, jumlah data, arsitektur head, dsb.) sesuai kebutuhan.

**Disclaimer:** Aplikasi ini bersifat edukatif. Hasil klasifikasi **tidak
boleh** dijadikan satu-satunya acuan untuk menentukan keamanan konsumsi jamur
di dunia nyata. Selalu konsultasikan dengan ahli mikologi.
