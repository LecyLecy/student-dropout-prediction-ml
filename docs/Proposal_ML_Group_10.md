# Klasifikasi Risiko Dropout Mahasiswa Menggunakan Fitur Enrollment dan Background Awal

_Source file: Proposal ML - Group 10(2).docx_

## I. Problem

Angka dropout di perguruan tinggi berdampak serius pada individu, institusi, dan produktivitas nasional. Sistem deteksi yang hanya mengandalkan nilai akademik semester berjalan sering terlambat karena informasi tersebut baru tersedia setelah mahasiswa mengikuti perkuliahan.

Proyek ini menggunakan dataset *Predict Students' Dropout and Academic Success* untuk membangun model klasifikasi biner yang memprediksi risiko dropout menggunakan fitur awal yang lebih masuk akal untuk MVP: informasi enrollment, academic path, dan background mahasiswa.

## II. Dataset

Dataset yang digunakan:

[Predict students' dropout and academic success | Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)

Target asli memiliki tiga kelas: Graduate, Dropout, dan Enrolled. Untuk proyek ini, Enrolled dihapus sehingga tugas model menjadi klasifikasi biner:

- Graduate = 0
- Dropout = 1

## III. Feature Scope

Final MVP menggunakan 10 fitur tetap:

```text
Marital status
Course
Previous qualification
Mother's qualification
Father's qualification
Displaced
Educational special needs
Gender
Age at enrollment
International
```

Fitur semester akademik, status administrasi setelah diterima, macroeconomic variables, application mode/order, occupation variables, dan nationality tidak digunakan agar MVP lebih mudah dipahami, lebih ringkas, dan mengurangi risiko leakage.

## IV. Modeling

Proyek ini membandingkan dua model:

1. Logistic Regression
2. Random Forest

Logistic Regression digunakan sebagai baseline sederhana dan interpretable. Random Forest digunakan sebagai model pembanding non-linear yang lebih cocok untuk pola tabular dengan banyak fitur kategorikal.

## V. Preprocessing

Tahapan preprocessing:

- Hapus Enrolled.
- Encode target menjadi Graduate = 0 dan Dropout = 1.
- Simpan hanya 10 fitur MVP + Target ke `data/processed/processed.csv`.
- Gunakan OneHotEncoder untuk fitur kategorikal dalam pipeline model.
- Gunakan StandardScaler untuk `Age at enrollment`.

## VI. Metrics Evaluation

Evaluasi menggunakan:

1. **F1-Score** untuk menyeimbangkan precision dan recall pada kelas risiko dropout.
2. **Recall** untuk mengurangi mahasiswa berisiko yang tidak terdeteksi.
3. **Precision** untuk menjaga prediksi risiko tetap tepat sasaran.
4. **ROC-AUC** untuk mengukur pemisahan kelas berdasarkan probabilitas.
5. **Confusion Matrix** untuk melihat pola kesalahan prediksi.

## VII. Deployment

Model disimpan sebagai pipeline scikit-learn dan digunakan oleh aplikasi Streamlit. User mengisi form berdasarkan 10 fitur MVP, lalu sistem menampilkan prediksi Graduate atau Dropout beserta probabilitasnya.

## VIII. References

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predict Students' Dropout and Academic Success* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89
