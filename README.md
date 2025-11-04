# Career Prediction System for ITB Alumni
**Version 1.0 – Initial System with Baseline Models**

## Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi kesesuaian bidang kerja alumni dengan program studi yang ditempuh menggunakan pendekatan *machine learning*.
Sistem ini dikembangkan sebagai bagian dari kegiatan Tracer Study ITB untuk membantu Career Center ITB menganalisis keterkaitan antara pendidikan dan karier alumni.
Pada tahap ini, sistem berfokus pada pengembangan model prediktif berbasis data tracer alumni tahun 2016–2024, yang menjadi fondasi bagi pengembangan dashboard analitik dan sistem *AI explanation* pada versi berikutnya.

## Capaian Utama v1.0
- Mengembangkan pipeline *machine learning* lengkap untuk prediksi kesesuaian karier alumni.
- Melatih dan membandingkan empat model utama: Random Forest, MLP, XGBoost, dan CatBoost.
- Membangun model *soft voting ensemble* dengan F1-Score validasi sebesar 0.7568 (tanpa fitur leaky).
- Mengimplementasikan *Optuna hyperparameter optimization* dan analisis interpretabilitas model menggunakan SHAP.
- Mengidentifikasi dan memitigasi dua permasalahan utama pada data tracer: *data leakage* dan *temporal concept drift* (2016–2017).

## Struktur Proyek
```
career-predictions-system/
├── data/               # Data mentah dan hasil pemrosesan
├── models/             # Model terlatih dan artefak pipeline
├── notebooks/          # Notebook eksplorasi dan analisis
├── reports/            # Hasil visualisasi dan laporan
├── src/                # Source code utama proyek
│   ├── data/           # Pembuatan dan validasi dataset
│   ├── features/       # Feature engineering
│   └── models/         # Training, prediksi, dan visualisasi hasil
├── results/            # Output hasil prediksi dan evaluasi
├── plots/              # SHAP dan perbandingan model
├── README.md
└── requirements.txt
```

## Instalasi dan Menjalankan Proyek

### 1. Clone Repository
```bash
git clone [repository-url]
cd career-predictions-system
```

### 2. Buat Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. (Opsional) Setup Jupyter Kernel
```bash
python -m ipykernel install --user --name=career-alignment --display-name="Career Alignment ML"
```

## Dataset
Proyek ini menggunakan data tracer alumni ITB dalam format Excel/CSV:
- `data 2016 - daffari_raw.csv` – Dataset pelatihan utama
- `DATA TS SARJANA 2024.xlsx` – Dataset pengujian temporal (cross-year validation)

Total data: sekitar 6.000 record alumni, dengan 12 fitur utama hasil seleksi dari 316 kolom awal.

## Alur Pengembangan

| Tahap | Deskripsi | Notebook/Script |
|-------|-----------|-----------------|
| **Fase 1: Eksplorasi dan Validasi Data** | Analisis 316 kolom data tracer, identifikasi fitur penting, validasi format antar tahun | `notebooks/01_eda_dan_validasi.ipynb` |
| **Fase 2: Pengembangan Model Baseline** | Training baseline (Random Forest, MLP) dan analisis performa | `notebooks/02_baseline_model_development.ipynb` |
| **Fase 3: Model Lanjutan dan Ensemble** | XGBoost + CatBoost + Soft Voting Ensemble + SHAP analysis | `src/models/train_models.py` |
| **Fase 4: Prediksi dan Evaluasi Temporal** | Testing model 2016→2017 dan visualisasi hasil prediksi | `src/models/predict_model.py`, `notebooks/03_analisis_hasil_akhir.ipynb` |

## Model yang Digunakan
- **Baseline Models:** Random Forest, MLP
- **Advanced Models:** XGBoost, CatBoost
- **Ensemble:** Soft Voting Classifier
- **Optimization:** Optuna (50 percobaan per model)
- **Interpretability:** SHAP (Feature Importance dan Summary Plots)

## Metrik Evaluasi
- Accuracy
- Precision / Recall
- F1-Score (utama)
- ROC-AUC
- Confusion Matrix

**Hasil utama:**
- Model terbaik: Soft Voting Ensemble
- F1-Score: 0.7568
- Fitur paling berpengaruh: IPK (kontribusi +0.45 SHAP)

## Output Utama
- `results/advanced/validation_results.json` – Hasil validasi model
- `plots/advanced/...` – Visualisasi SHAP dan perbandingan model
- `reports_figures/` – Grafik analisis untuk laporan
- `models/advanced/without_leaky/` – Artefak model final (tanpa data leakage)

## Kontributor
- Daffari Adiyatma (18222003) – Data Analyst, ITB Tracer Study
- Pembimbing: Career Center ITB

## Lisensi
© 2025 Institut Teknologi Bandung.
Proyek ini dikembangkan untuk keperluan penelitian dan non-komersial.

## Catatan Versi

### Version 1.0 – Initial System with Baseline Models
- Menyelesaikan pipeline pengembangan dan evaluasi model prediksi karier.
- Mengatasi data leakage dan temporal drift.
- Mengintegrasikan analisis SHAP untuk interpretabilitas model.
- Menyusun struktur proyek terstandarisasi dan terdokumentasi.
- Siap dikembangkan lebih lanjut menuju integrasi Dashboard dan FastAPI Backend (v2.0).