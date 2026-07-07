Folder ini akan berisi artefak hasil training setelah menjalankan `python train.py`:

- best_model.keras           -> model Xception terbaik (disimpan otomatis oleh ModelCheckpoint)
- class_names.json           -> daftar nama kelas sesuai urutan index model
- training_history.png       -> grafik akurasi & loss selama training
- confusion_matrix.png       -> confusion matrix pada data test
- classification_report.txt -> precision, recall, f1-score, dan accuracy per kelas

File-file ini SENGAJA belum ada di repo karena harus dihasilkan dengan
menjalankan proses training terlebih dahulu (lihat README.md).
