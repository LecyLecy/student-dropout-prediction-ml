# Klasifikasi Risiko Dropout Mahasiswa Perguruan Tinggi Menggunakan Algoritma Naive Bayes Berbasis Data Enrollment Non-Akademik

_Source file: Proposal ML - Group 10(2).docx_

## I. Problem

Angka dropout di perguruan tinggi berdampak serius pada individu, institusi, dan produktivitas nasional. Sistem deteksi yang ada selama ini mengandalkan data nilai akademik yang baru tersedia setelah satu atau dua semester berjalan, sehingga intervensi sering terlambat. Padahal, faktor risiko seperti latar belakang sosial-ekonomi, pekerjaan orang tua, status beasiswa, dan jalur masuk sudah tersedia sejak awal enrollment.

Penelitian ini memanfaatkan dataset *Predict Students' Dropout and Academic Success* (Realinho et al., 2022) untuk membangun model klasifikasi biner berbasis fitur non-akademik menggunakan XGBoost, guna mengidentifikasi mahasiswa berisiko dropout sedini mungkin sebelum performa akademik mereka terpengaruh (Carballo-Mendívil et al., 2025).

## II. Dataset

Berikut telampir dataset yang kami gunakan:

[Predict students' dropout and academic success | Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)

Penelitian ini menggunakan dataset *Predict Students' Dropout and Academic Success*, sebuah kumpulan data komprehensif yang mencakup rekam jejak mahasiswa pada berbagai program studi sarjana di sebuah institusi pendidikan tinggi. Dataset ini sangat unik karena menggabungkan berbagai database terpisah (*disjoint databases*) untuk memberikan gambaran menyeluruh mengenai faktor-faktor yang memengaruhi persistensi mahasiswa, mulai dari disiplin ilmu agronomi, desain, keperawatan, jurnalisme, manajemen, hingga teknologi.

## III. Baseline

Fokus utama proyek ini adalah melakukan klasifikasi biner untuk memprediksi apakah mahasiswa berstatus dropout atau success berdasarkan fitur non-akademik.

### Algoritma

Algoritma utama yang digunakan adalah Naive Bayes. Model ini merupakan algoritma klasifikasi probabilistik yang menggunakan Teorema Bayes dengan asumsi independensi antar fitur.

### Model Baseline

Sebagai model baseline, penelitian ini menggunakan Logistic Regression karena merupakan algoritma klasifikasi biner klasik yang sederhana, mudah diinterpretasikan, dan umum digunakan sebagai acuan performa awal. Model ini digunakan untuk membandingkan efektivitas model utama, yaitu Naive Bayes, dalam memprediksi risiko dropout mahasiswa berdasarkan fitur non-akademik.

### Rencana Pelatihan

#### Pembagian Data & Isolasi Fitur

Dataset akan dibagi dengan rasio 80:20 untuk stabilitas performa. Kami akan menghapus fitur nilai akademik untuk mengisolasi pengaruh variabel demografi dan sosial terhadap dropout.

#### Preprocessing

Fitur kategorikal akan dilakukan encoding agar dapat diproses model. Missing values akan ditangani menggunakan imputasi sederhana sesuai tipe data.

#### Penanganan Data Tidak Seimbang

Menggunakan teknik SMOTE pada data latih agar model tidak bias terhadap kelas mayoritas, yaitu mahasiswa yang sukses.

## IV. Metrics Evaluation

Untuk mengukur efektivitas model XGBoost dalam mengidentifikasi mahasiswa yang berisiko dropout berdasarkan data non-akademik, kami menetapkan rencana evaluasi sebagai berikut:

### Metrik Evaluasi

1. **F1-Score**: Menyeimbangkan Precision dan Recall untuk mengukur performa model secara keseluruhan pada klasifikasi risiko dropout.
2. **Recall**: Fokus utama untuk meminimalisir mahasiswa berisiko yang tidak terdeteksi atau *False Negative*, sehingga intervensi dapat dilakukan tepat waktu.
3. **Precision**: Memastikan akurasi prediksi agar sumber daya intervensi seperti beasiswa atau konseling tersalurkan tepat sasaran kepada yang membutuhkan.
4. **ROC-AUC**: Mengukur kemampuan model dalam membedakan kelas Dropout dan Success pada berbagai ambang batas probabilitas.
5. **Confusion Matrix**: Alat visualisasi untuk menganalisis detail kesalahan prediksi dan efektivitas klasifikasi model.

## V. Deployment

Model akan diimplementasikan dalam aplikasi berbasis Streamlit agar mudah digunakan sebagai sistem prediksi awal. Pengguna memasukkan data non-akademik mahasiswa melalui form, lalu sistem akan menghasilkan prediksi kelas Dropout atau Success beserta probabilitasnya. Hasil ini dapat digunakan sebagai alat bantu bagi pihak akademik atau konselor untuk melakukan intervensi lebih awal.

## VI. References

Realinho, V., Machado, J., Baptista, L., & Martins, M. V. (2022). *Predict Students' Dropout and Academic Success* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5MC89

Carballo-Mendívil, B., Arellano-González, A., Ríos-Vázquez, N., & Lizardi-Duarte, M. (2025). Predicting student dropout from day one: XGBoost-based early warning system using pre-enrollment data. *Applied Sciences, 15*(16), 9202. https://doi.org/10.3390/app15169202

Bishop, C. M. (2016). *Pattern Recognition and Machine Learning*. Springer New York.
