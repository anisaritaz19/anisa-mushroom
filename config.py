"""
config.py
==========
Konfigurasi terpusat untuk proyek Anisa - Klasifikasi Jenis Jamur
menggunakan Transfer Learning Xception.

Semua path, hyperparameter, dan pengaturan dataset didefinisikan di sini
supaya train.py, predict.py, dan app.py memakai sumber kebenaran yang sama.
"""

import os

# ---------------------------------------------------------------------------
# PATH DASAR
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")              # dataset asli hasil kagglehub
SUBSET_DIR = os.path.join(BASE_DIR, "data_subset")      # subset kecil yang dipakai untuk training
SPLIT_DIR = os.path.join(BASE_DIR, "data_split")        # hasil split train/val/test

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")
HISTORY_PLOT_PATH = os.path.join(MODEL_DIR, "training_history.png")
CONFUSION_MATRIX_PATH = os.path.join(MODEL_DIR, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(MODEL_DIR, "classification_report.txt")

UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# ---------------------------------------------------------------------------
# DATASET KAGGLE
# ---------------------------------------------------------------------------
KAGGLE_DATASET = "mdhasanahmad/12-mushroom-species-dataset"

# Berapa banyak gambar per kelas yang diambil untuk subset (tugas kuliah -> cepat).
# 12 kelas x ~150 gambar =~ 1800 gambar total (bisa disesuaikan 1000-2000).
IMAGES_PER_CLASS = 150

# Rasio split dataset subset
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# PARAMETER MODEL / TRAINING
# ---------------------------------------------------------------------------
IMG_SIZE = (299, 299)     # ukuran input standar Xception
IMG_SHAPE = (299, 299, 3)
BATCH_SIZE = 16

# Jumlah epoch dibatasi maksimal sesuai kebutuhan tugas kuliah (cepat & efisien)
EPOCHS_HEAD = 4            # tahap 1: melatih head classifier, base Xception dibekukan
EPOCHS_FINE_TUNE = 6       # tahap 2: fine-tuning ringan beberapa layer teratas Xception
# total maksimum epoch efektif (dipangkas otomatis oleh EarlyStopping bila perlu)

FINE_TUNE_AT_LAYER = -30   # buka (unfreeze) 30 layer teratas Xception untuk fine-tuning ringan
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINE_TUNE = 1e-5

EARLY_STOPPING_PATIENCE = 3
DROPOUT_RATE = 0.3

# Nama kelas default (akan ditimpa otomatis oleh train.py berdasarkan folder dataset asli,
# lalu disimpan ke class_names.json supaya predict.py & app.py konsisten).
DEFAULT_CLASS_NAMES = [
    "Class_1", "Class_2", "Class_3", "Class_4",
    "Class_5", "Class_6", "Class_7", "Class_8",
    "Class_9", "Class_10", "Class_11", "Class_12",
]

# ---------------------------------------------------------------------------
# PENGATURAN FLASK
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH_MB = 8
SECRET_KEY = os.environ.get("SECRET_KEY", "anisa-mushroom-classifier-dev-key")
TOP_K_PREDICTIONS = 3
