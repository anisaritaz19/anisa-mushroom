"""
predict.py
==========
Modul untuk memuat model Xception hasil training dan melakukan prediksi
jenis jamur dari sebuah file gambar.

Dipakai oleh app.py (Flask) dan juga bisa dijalankan langsung dari CLI:

    python predict.py path/ke/gambar.jpg
"""

import os
import sys
import json

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.xception import preprocess_input

import config

_model = None
_class_names = None


def load_class_names():
    global _class_names
    if _class_names is not None:
        return _class_names

    if os.path.exists(config.CLASS_NAMES_PATH):
        with open(config.CLASS_NAMES_PATH, "r") as f:
            _class_names = json.load(f)
    else:
        _class_names = config.DEFAULT_CLASS_NAMES
    return _class_names


def load_model():
    """Memuat model sekali saja (lazy loading), lalu di-cache di memori."""
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(config.MODEL_PATH):
        raise FileNotFoundError(
            f"Model tidak ditemukan di '{config.MODEL_PATH}'. "
            "Jalankan 'python train.py' terlebih dahulu untuk melatih model."
        )

    _model = tf.keras.models.load_model(config.MODEL_PATH)
    return _model


def preprocess_image(image_path_or_pil):
    """
    Menerima path file ATAU objek PIL.Image, mengembalikan array siap
    dimasukkan ke model (batch dimension sudah ditambahkan).
    """
    if isinstance(image_path_or_pil, (str, os.PathLike)):
        img = Image.open(image_path_or_pil)
    else:
        img = image_path_or_pil

    img = img.convert("RGB").resize(config.IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def predict_image(image_path_or_pil, top_k=config.TOP_K_PREDICTIONS):
    """
    Menjalankan prediksi pada satu gambar.

    Returns
    -------
    dict dengan keys:
        - predicted_class : str
        - confidence      : float (0-100)
        - top_k           : list of (class_name, confidence_percent)
    """
    model = load_model()
    class_names = load_class_names()

    x = preprocess_image(image_path_or_pil)
    probs = model.predict(x, verbose=0)[0]  # shape (num_classes,)

    top_indices = probs.argsort()[::-1][:top_k]
    top_k_results = [
        (class_names[i], float(probs[i] * 100.0)) for i in top_indices
    ]

    best_idx = int(np.argmax(probs))
    result = {
        "predicted_class": class_names[best_idx],
        "confidence": float(probs[best_idx] * 100.0),
        "top_k": top_k_results,
    }
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Pemakaian: python predict.py path/ke/gambar.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"File tidak ditemukan: {image_path}")
        sys.exit(1)

    result = predict_image(image_path)
    print("\n=== HASIL PREDIKSI ===")
    print(f"Jenis jamur : {result['predicted_class']}")
    print(f"Confidence  : {result['confidence']:.2f}%")
    print("\nTop-K prediksi:")
    for cls, conf in result["top_k"]:
        print(f"  - {cls:30s} {conf:6.2f}%")
