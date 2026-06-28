<div align="center">

# 🥛 Milk Quality Prediction
### UAS Kecerdasan Buatan

Prediksi kualitas susu menggunakan algoritma **Decision Tree** dan **K-Nearest Neighbors (KNN)**.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

</div>

---

# Persyaratan

Pastikan perangkat telah terpasang:

- Python 3.10 atau lebih baru
- Git
- Jupyter Notebook atau Visual Studio Code

Cek versi Python:

```bash
python --version
```

atau

```bash
python3 --version
```

---

# Clone Repository

Clone repository ke komputer Anda.

```bash
git clone https://github.com/keyzakyy-dev/UAS-KecerdasanBuatan.git
```

Masuk ke folder project.

```bash
cd UAS-KecerdasanBuatan
```

---

# Membuat Virtual Environment (Disarankan)

### Windows

```bash
python -m venv .venv
```

Aktifkan environment

```bash
.venv\Scripts\activate
```

---

### Linux / macOS

```bash
python3 -m venv .venv
```

Aktifkan

```bash
source .venv/bin/activate
```

---

# Install Dependency

Jika tersedia file `requirements.txt`

```bash
pip install -r requirements.txt
```

Apabila belum tersedia, install library berikut.

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install jupyter
```

atau sekaligus

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

# Menjalankan Project

Jalankan Jupyter Notebook.

```bash
jupyter notebook
```

atau

```bash
jupyter lab
```

Kemudian buka file

```
uas_model.ipynb
```

Lalu pilih menu

```
Run → Run All Cells
```

atau tekan

```
Shift + Enter
```

pada setiap cell.

---

# Output

Notebook akan menghasilkan:

- Exploratory Data Analysis (EDA)
- Visualisasi Distribusi Data
- Heatmap Korelasi
- Training Decision Tree
- Training K-Nearest Neighbors
- Confusion Matrix
- Classification Report
- Perbandingan Akurasi Model

---

# Troubleshooting

### ModuleNotFoundError

Install library yang belum tersedia.

```bash
pip install nama_library
```

Contoh:

```bash
pip install scikit-learn
```

---

### Jupyter tidak dikenali

Install Jupyter.

```bash
pip install notebook
```

Kemudian jalankan kembali.

```bash
jupyter notebook
```

---

### Virtual Environment tidak aktif

Aktifkan kembali.

Windows

```bash
.venv\Scripts\activate
```

Linux/macOS

```bash
source .venv/bin/activate
```

---

# 

**Sayyid Dzaky Farhan**

Teknik Informatika  
Institut Teknologi Garut

---