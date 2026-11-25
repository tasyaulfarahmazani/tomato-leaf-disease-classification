# 🍅 Tomato Leaf Disease Classification Based on Color, Texture, and Shape Feature Extraction

[![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange.svg)](https://scikit-learn.org/)
[![Computer Vision](https://img.shields.io/badge/OpenCV-Image%20Processing-green.svg)](https://opencv.org/)
[![Best Accuracy](https://img.shields.io/badge/Accuracy-95.5%25%20(Random%20Forest)-success.svg)](#hasil-dan-evaluasi)

Sistem Computer Vision dan Machine Learning untuk mengklasifikasikan kondisi kesehatan daun tomat menjadi 3 kelas utama: **Daun Sehat (Healthy)**, **Infeksi Bakteri (Bacterial Spot)**, dan **Busuk Daun (Late Blight)**.

Proyek riset ini menggabungkan teknik ekstraksi fitur digital multimodal (**Warna RGB**, **Tekstur GLCM**, dan **Bentuk**) yang dievaluasi menggunakan empat algoritma Machine Learning: **Random Forest**, **Support Vector Machine (SVM)**, **K-Nearest Neighbor (KNN)**, dan **Naive Bayes**.

---

## 🔬 Metodologi Penelitian

### 1. Preprocessing Citra
- **Image Resizing**: Standardisasi resolusi input menjadi `128 × 128` piksel.
- **Color Space Transformation**: Konversi citra `BGR` ke ruang warna `RGB` dan `Grayscale`.

### 2. Ekstraksi Fitur Digital (Feature Extraction)
1. **Fitur Warna (Color Features)**:
   - Nilai rata-rata (*Mean*) komponen Red, Green, Blue: $\mu_R, \mu_G, \mu_B$
   - Standar deviasi (*Standard Deviation*) tiap kanal warna: $\sigma_R, \sigma_G, \sigma_B$
2. **Fitur Tekstur GLCM (Gray-Level Co-occurrence Matrix)**:
   - Menangkap pola keteraturan spasial intensitas piksel daun pada jarak $d=1$ dan sudut $\theta=0^\circ$:
     - **Contrast**: Variasi intensitas lokal.
     - **Dissimilarity**: Perbedaan nilai pasangan piksel.
     - **Homogeneity**: Keseragaman tekstur permukaan daun.
     - **Energy (Angular Second Moment)**: Keteraturan struktur tekstur.
     - **Correlation**: Ketergantungan linier intensitas piksel.
3. **Reduksi Dimensi & Normalisasi**:
   - Penskalaan fitur menggunakan `StandardScaler`.
   - Analisis komponen utama (*PCA*) untuk eliminasi redundansi dimensi.

---

## 📊 Hasil dan Evaluasi Model

Pembagian dataset dilakukan secara terstratifikasi dengan rasio **80% Training Data** dan **20% Testing Data**.

| Model Klasifikasi | Akurasi (%) | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🌲 **Random Forest** | **95.47% (~95.5%)** | **0.96** | **0.96** | **0.96** |
| 🎯 **K-Nearest Neighbor (KNN)** | 92.98% (~93.0%) | 0.93 | 0.93 | 0.93 |
| ⚡ **Support Vector Machine (SVM)** | 92.72% (~92.7%) | 0.93 | 0.93 | 0.93 |
| 📉 **Naive Bayes** | 76.64% (~76.6%) | 0.80 | 0.76 | 0.75 |

> **Kesimpulan Riset**: Model **Random Forest** menghasilkan performa paling optimal karena pendekatan *ensemble decision trees* mampu menangkap korelasi non-linear antarinformasi tekstur GLCM dan warna tanpa mengalami *overfitting*.

---

## 🛠️ Tech Stack & Library

- **Bahasa**: Python 3.10+
- **Image Processing**: OpenCV (`cv2`), Scikit-Image (`skimage.feature.graycomatrix`)
- **Machine Learning**: Scikit-Learn (`sklearn.ensemble`, `sklearn.svm`, `sklearn.neighbors`, `sklearn.naive_bayes`)
- **Data & Visualisasi**: Pandas, NumPy, Matplotlib, Seaborn

---

## 🚀 Cara Menjalankan

1. **Clone repository ini**:
   ```bash
   git clone https://github.com/tasyaulfarahmazani/tomato-leaf-disease-classification.git
   cd tomato-leaf-disease-classification
   ```

2. **Install dependensi**:
   ```bash
   pip install opencv-python numpy pandas matplotlib seaborn scikit-learn scikit-image
   ```

3. **Jalankan script ekstraksi dan klasifikasi**:
   ```bash
   python ekstraksi_daun.py
   ```

---

## 👤 Penulis
- **Tasya Ulfa Rahmazani**  
  Teknik Informatika - Politeknik Negeri Lhokseumawe  
  [LinkedIn](https://www.linkedin.com/in/tasyaulfarahmazani/) • [GitHub](https://github.com/tasyaulfarahmazani)
