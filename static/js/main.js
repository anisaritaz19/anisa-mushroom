/**
 * main.js
 * Menangani interaksi dropzone upload: drag & drop, klik untuk memilih file,
 * preview gambar sebelum dikirim, dan validasi dasar sisi klien.
 */

document.addEventListener("DOMContentLoaded", function () {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const dropzoneContent = document.getElementById("dropzoneContent");
  const previewImage = document.getElementById("previewImage");
  const submitBtn = document.getElementById("submitBtn");
  const uploadForm = document.getElementById("uploadForm");

  if (!dropzone || !fileInput) return;

  const ALLOWED_TYPES = ["image/png", "image/jpeg", "image/webp"];
  const MAX_SIZE_MB = 8;

  function showPreview(file) {
    if (!ALLOWED_TYPES.includes(file.type)) {
      alert("Format file tidak didukung. Gunakan PNG, JPG, JPEG, atau WEBP.");
      return false;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      alert(`Ukuran file melebihi ${MAX_SIZE_MB}MB.`);
      return false;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
      previewImage.src = e.target.result;
      previewImage.classList.remove("d-none");
      dropzoneContent.classList.add("d-none");
      submitBtn.disabled = false;
    };
    reader.readAsDataURL(file);
    return true;
  }

  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", function () {
    if (this.files && this.files[0]) {
      showPreview(this.files[0]);
    }
  });

  ["dragenter", "dragover"].forEach((evt) => {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((evt) => {
    dropzone.addEventListener(evt, function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", function (e) {
    const files = e.dataTransfer.files;
    if (files && files[0]) {
      fileInput.files = files;
      showPreview(files[0]);
    }
  });

  if (uploadForm) {
    uploadForm.addEventListener("submit", function () {
      submitBtn.disabled = true;
      submitBtn.innerHTML =
        '<span class="spinner-border spinner-border-sm me-2"></span> Menganalisis...';
    });
  }
});
