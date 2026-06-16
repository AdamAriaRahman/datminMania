# 📡 Customer Churn Dashboard
**Tugas Besar Data Mining — SI-48-10 | Telkom University 2025/2026**

Dashboard Streamlit untuk prediksi customer churn IBM Telco menggunakan K-Means Clustering, Logistic Regression, dan Naïve Bayes.

---

## 🚀 Cara Menjalankan di VS Code

### 1. Install dependencies
Buka terminal di VS Code, lalu jalankan:
```bash
pip install -r requirements.txt
```

### 2. Jalankan dashboard
```bash
streamlit run app.py
```

Dashboard akan otomatis terbuka di browser di `http://localhost:8501`

---

## 📋 Fitur Dashboard

| Halaman | Isi |
|---|---|
| 🏠 Overview | KPI utama, distribusi churn, performa model |
| 🔍 Eksplorasi Data | Distribusi numerik, kategorikal, heatmap korelasi |
| 📦 K-Means Clustering | Elbow method, visualisasi PCA, profil cluster |
| 📈 Logistic Regression | Confusion matrix, ROC curve, feature importance |
| 🧮 Naïve Bayes | Confusion matrix, ROC curve, log ratio features |
| ⚖️ Perbandingan Model | Tabel & chart perbandingan LR vs NB |
| 🎯 Prediksi Pelanggan | Input data pelanggan baru → prediksi churn |

---

## 📁 Struktur File
```
dashboard/
├── app.py           ← File utama Streamlit
├── requirements.txt ← Dependencies
└── README.md        ← Panduan ini
```

---

## ⚠️ Catatan
- Dataset diload langsung dari Google Sheets (butuh koneksi internet)
- Pertama kali load bisa memakan waktu ~30 detik untuk preprocessing & training
- Setelah itu data di-cache sehingga navigasi antar halaman cepat
