"""
train.py
========
Pipeline lengkap untuk melatih model klasifikasi jenis jamur menggunakan
Transfer Learning Xception (ImageNet) dengan fine-tuning ringan.

Alur:
1. Unduh dataset "12 Mushroom Species Dataset" dari Kaggle via kagglehub.
2. Ambil SUBSET kecil per kelas (config.IMAGES_PER_CLASS) agar training cepat.
3. Split subset menjadi train / validation / test.
4. Bangun model Xception (include_top=False) + classifier head kustom.
5. Tahap 1: latih hanya head classifier (base Xception dibekukan).
6. Tahap 2: fine-tuning ringan -> unfreeze beberapa layer teratas Xception.
7. EarlyStopping + ModelCheckpoint menyimpan model TERBAIK saja.
8. Evaluasi akhir: accuracy, classification report, confusion matrix (PNG).

Jalankan:
    python train.py
"""

import os
import json
import random
import shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # supaya bisa jalan tanpa display (server / headless)
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Input
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import config


def set_seed(seed=config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# 1. DOWNLOAD DATASET (kagglehub)
# ---------------------------------------------------------------------------
def download_dataset():
    """
    Mengunduh dataset dari Kaggle memakai kagglehub.
    Mengembalikan path folder root dataset hasil download.
    """
    import kagglehub

    print(f"[1/6] Mengunduh dataset Kaggle: {config.KAGGLE_DATASET} ...")
    dataset_path = kagglehub.dataset_download(config.KAGGLE_DATASET)
    print(f"      Dataset tersedia di: {dataset_path}")
    return dataset_path


def find_class_folders(dataset_root):
    """
    Dataset Kaggle biasanya berbentuk:
        dataset_root/
            <mungkin subfolder pembungkus>/
                ClassA/*.jpg
                ClassB/*.jpg
                ...
    Fungsi ini mencari folder yang langsung berisi banyak subfolder gambar
    (folder kelas), tidak peduli berapa level nesting-nya.
    """
    dataset_root = Path(dataset_root)
    candidates = []

    for root, dirs, files in os.walk(dataset_root):
        image_dirs = []
        for d in dirs:
            sub = Path(root) / d
            has_images = any(
                f.lower().endswith((".jpg", ".jpeg", ".png"))
                for f in os.listdir(sub)[:20]
            ) if sub.exists() else False
            if has_images:
                image_dirs.append(d)
        if len(image_dirs) >= 2:  # minimal 2 folder kelas dianggap kandidat valid
            candidates.append((root, image_dirs))

    if not candidates:
        raise RuntimeError(
            "Tidak ditemukan struktur folder kelas pada dataset yang diunduh. "
            "Periksa kembali isi dataset secara manual."
        )

    # Ambil kandidat dengan jumlah folder kelas terbanyak (biasanya level yang benar)
    best_root, best_dirs = max(candidates, key=lambda x: len(x[1]))
    return best_root, sorted(best_dirs)


# ---------------------------------------------------------------------------
# 2. BUAT SUBSET DATASET (1000-2000 gambar total)
# ---------------------------------------------------------------------------
def build_subset(dataset_root):
    print("[2/6] Membuat subset dataset (mempercepat training)...")
    class_root, class_names = find_class_folders(dataset_root)
    print(f"      Ditemukan {len(class_names)} kelas: {class_names}")

    if os.path.exists(config.SUBSET_DIR):
        shutil.rmtree(config.SUBSET_DIR)
    os.makedirs(config.SUBSET_DIR, exist_ok=True)

    rng = random.Random(config.RANDOM_SEED)
    total_copied = 0

    for cls in class_names:
        src_dir = Path(class_root) / cls
        images = [
            f for f in os.listdir(src_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        rng.shuffle(images)
        selected = images[: config.IMAGES_PER_CLASS]

        dst_dir = Path(config.SUBSET_DIR) / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        for img_name in selected:
            shutil.copy2(src_dir / img_name, dst_dir / img_name)

        total_copied += len(selected)
        print(f"      {cls}: {len(selected)} gambar disalin")

    print(f"      Total gambar dalam subset: {total_copied}")
    return class_names


# ---------------------------------------------------------------------------
# 3. SPLIT TRAIN / VAL / TEST
# ---------------------------------------------------------------------------
def split_dataset(class_names):
    print("[3/6] Membagi subset menjadi train / val / test...")
    if os.path.exists(config.SPLIT_DIR):
        shutil.rmtree(config.SPLIT_DIR)

    rng = random.Random(config.RANDOM_SEED)

    for split in ["train", "val", "test"]:
        for cls in class_names:
            os.makedirs(os.path.join(config.SPLIT_DIR, split, cls), exist_ok=True)

    for cls in class_names:
        src_dir = Path(config.SUBSET_DIR) / cls
        images = os.listdir(src_dir)
        rng.shuffle(images)

        n = len(images)
        n_train = int(n * config.TRAIN_RATIO)
        n_val = int(n * config.VAL_RATIO)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for fname in train_files:
            shutil.copy2(src_dir / fname, Path(config.SPLIT_DIR) / "train" / cls / fname)
        for fname in val_files:
            shutil.copy2(src_dir / fname, Path(config.SPLIT_DIR) / "val" / cls / fname)
        for fname in test_files:
            shutil.copy2(src_dir / fname, Path(config.SPLIT_DIR) / "test" / cls / fname)

    print("      Selesai membagi dataset.")


# ---------------------------------------------------------------------------
# 4. DATA GENERATOR
# ---------------------------------------------------------------------------
def build_generators():
    print("[4/6] Menyiapkan data generator (augmentasi ringan untuk data train)...")

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(config.SPLIT_DIR, "train"),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
    )
    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(config.SPLIT_DIR, "val"),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )
    test_gen = val_test_datagen.flow_from_directory(
        os.path.join(config.SPLIT_DIR, "test"),
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


# ---------------------------------------------------------------------------
# 5. BANGUN MODEL (Xception + head kustom)
# ---------------------------------------------------------------------------
def build_model(num_classes):
    print("[5/6] Membangun model Xception (pretrained ImageNet) + classifier head...")

    base_model = Xception(
        weights="imagenet",
        include_top=False,
        input_shape=config.IMG_SHAPE,
    )
    base_model.trainable = False  # bekukan dulu untuk tahap 1

    inputs = Input(shape=config.IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(config.DROPOUT_RATE)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base_model


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def get_callbacks():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1
        ),
    ]


# ---------------------------------------------------------------------------
# 6. EVALUASI: accuracy, classification report, confusion matrix
# ---------------------------------------------------------------------------
def evaluate_model(model, test_gen, class_names):
    print("[6/6] Mengevaluasi model pada data test...")

    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    print(f"\n>>> Test Accuracy: {acc * 100:.2f}% <<<\n")

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    with open(config.CLASSIFICATION_REPORT_PATH, "w") as f:
        f.write(f"Test Accuracy: {acc * 100:.2f}%\n\n")
        f.write(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title(f"Confusion Matrix - Test Accuracy {acc*100:.2f}%")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    print(f"      Confusion matrix disimpan: {config.CONFUSION_MATRIX_PATH}")

    return acc


def plot_history(history_head, history_fine_tune):
    acc = history_head.history["accuracy"] + history_fine_tune.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_fine_tune.history["val_accuracy"]
    loss = history_head.history["loss"] + history_fine_tune.history["loss"]
    val_loss = history_head.history["val_loss"] + history_fine_tune.history["val_loss"]

    epochs_range = range(len(acc))
    split_point = len(history_head.history["accuracy"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs_range, acc, label="Train Accuracy")
    axes[0].plot(epochs_range, val_acc, label="Val Accuracy")
    axes[0].axvline(split_point - 0.5, color="gray", linestyle="--", label="Mulai Fine-tuning")
    axes[0].set_title("Akurasi Training vs Validasi")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs_range, loss, label="Train Loss")
    axes[1].plot(epochs_range, val_loss, label="Val Loss")
    axes[1].axvline(split_point - 0.5, color="gray", linestyle="--", label="Mulai Fine-tuning")
    axes[1].set_title("Loss Training vs Validasi")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(config.HISTORY_PLOT_PATH, dpi=150)
    plt.close()
    print(f"Grafik histori training disimpan: {config.HISTORY_PLOT_PATH}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    set_seed()
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    dataset_root = download_dataset()
    class_names = build_subset(dataset_root)
    split_dataset(class_names)

    train_gen, val_gen, test_gen = build_generators()

    # class_names dari generator (urutan index Keras) -> ini yang WAJIB dipakai
    # supaya konsisten antara training & prediksi
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    ordered_class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    with open(config.CLASS_NAMES_PATH, "w") as f:
        json.dump(ordered_class_names, f, indent=2)
    print(f"Nama kelas disimpan di: {config.CLASS_NAMES_PATH}")

    model, base_model = build_model(num_classes=len(ordered_class_names))

    # ---------------- TAHAP 1: latih head classifier (base dibekukan) -------
    print("\n===== TAHAP 1: Melatih classifier head (Xception dibekukan) =====")
    compile_model(model, lr=config.LEARNING_RATE_HEAD)
    history_head = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_HEAD,
        callbacks=get_callbacks(),
    )

    # ---------------- TAHAP 2: fine-tuning ringan ----------------------------
    print("\n===== TAHAP 2: Fine-tuning ringan (unfreeze layer teratas Xception) =====")
    base_model.trainable = True
    for layer in base_model.layers[: config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False  # hanya buka N layer teratas

    compile_model(model, lr=config.LEARNING_RATE_FINE_TUNE)
    history_fine_tune = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_FINE_TUNE,
        callbacks=get_callbacks(),
    )

    plot_history(history_head, history_fine_tune)

    # Model terbaik (val_accuracy tertinggi) sudah otomatis tersimpan oleh
    # ModelCheckpoint di config.MODEL_PATH. Muat ulang untuk evaluasi final.
    print(f"\nMemuat kembali model terbaik dari: {config.MODEL_PATH}")
    best_model = tf.keras.models.load_model(config.MODEL_PATH)

    evaluate_model(best_model, test_gen, ordered_class_names)

    print("\n=== TRAINING SELESAI ===")
    print(f"Model terbaik : {config.MODEL_PATH}")
    print(f"Class names   : {config.CLASS_NAMES_PATH}")
    print(f"Confusion mtx : {config.CONFUSION_MATRIX_PATH}")
    print(f"Report        : {config.CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
