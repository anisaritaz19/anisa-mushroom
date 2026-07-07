"""
app.py
======
Aplikasi web Flask untuk klasifikasi jenis jamur.

Fitur:
- Upload gambar (drag & drop / pilih file) dengan preview di browser
- Prediksi jenis jamur menggunakan model Xception hasil transfer learning
- Menampilkan confidence score & top-3 kemungkinan jenis jamur lain
- Halaman "Tentang" berisi ringkasan proyek dan metodologi

Jalankan lokal:
    python app.py
Produksi (Render/Railway/PythonAnywhere):
    gunicorn app:app
"""

import os
import uuid
from datetime import datetime

from flask import (
    Flask, render_template, request, redirect, url_for, flash, jsonify
)
from werkzeug.utils import secure_filename

import config
import predict as predictor

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH_MB * 1024 * 1024
app.config["UPLOAD_FOLDER"] = config.UPLOAD_DIR

os.makedirs(config.UPLOAD_DIR, exist_ok=True)

MODEL_READY = os.path.exists(config.MODEL_PATH)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS
    )


@app.context_processor
def inject_globals():
    return {
        "current_year": datetime.now().year,
        "model_ready": MODEL_READY,
    }


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_route():
    if not MODEL_READY:
        flash("Model belum tersedia. Jalankan 'python train.py' terlebih dahulu.", "danger")
        return redirect(url_for("index"))

    if "image" not in request.files:
        flash("Tidak ada file yang diunggah.", "warning")
        return redirect(url_for("index"))

    file = request.files["image"]

    if file.filename == "":
        flash("Silakan pilih gambar terlebih dahulu.", "warning")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.", "warning")
        return redirect(url_for("index"))

    original_name = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{original_name}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        result = predictor.predict_image(save_path)
    except Exception as exc:
        flash(f"Terjadi kesalahan saat memproses gambar: {exc}", "danger")
        return redirect(url_for("index"))

    image_url = url_for("static", filename=f"uploads/{unique_name}")

    return render_template(
        "index.html",
        prediction=result,
        image_url=image_url,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Endpoint JSON, berguna untuk pengujian via curl/Postman atau front-end lain."""
    if not MODEL_READY:
        return jsonify({"error": "Model belum tersedia. Jalankan train.py."}), 503

    if "image" not in request.files:
        return jsonify({"error": "Field 'image' tidak ditemukan pada request."}), 400

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "File tidak valid."}), 400

    unique_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
    file.save(save_path)

    try:
        result = predictor.predict_image(save_path)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/about")
def about():
    return render_template("about.html")


@app.errorhandler(413)
def file_too_large(e):
    flash(f"Ukuran file melebihi batas maksimum {config.MAX_CONTENT_LENGTH_MB} MB.", "danger")
    return redirect(url_for("index"))


@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
