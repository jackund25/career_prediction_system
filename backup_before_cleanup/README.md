# Sistem Prediksi Kelulusan Tepat Waktu

## Deskripsi Proyek
Proyek ini bertujuan untuk memprediksi kelulusan tepat waktu mahasiswa menggunakan machine learning. Sistem ini menggunakan data historis mahasiswa untuk melatih model prediktif yang dapat membantu identifikasi dini mahasiswa yang berisiko tidak lulus tepat waktu.

## Struktur Proyek
```
career-predictions-system/
├── data/               # Data mentah dan yang sudah diproses
├── models/             # Model yang sudah dilatih
├── notebooks/          # Jupyter notebooks untuk eksplorasi dan analisis
├── reports/            # Laporan dan visualisasi hasil
├── src/                # Source code untuk proyek
│   ├── data/          # Script untuk memproses data
│   ├── features/      # Script untuk feature engineering
│   └── models/        # Script untuk training dan prediksi
├── README.md
└── requirements.txt
```

## Instalasi

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

### 4. Setup Jupyter Kernel
```bash
python -m ipykernel install --user --name=kelulusan --display-name="Python (Kelulusan)"
```

## Data
Proyek ini menggunakan dua dataset:
1. **DATA TS SARJANA 2024.xlsx** - Data training dari tahun 2024
2. **data 2016 - daffari_raw.csv** - Data untuk validasi temporal

## Workflow

### Fase 1: Eksplorasi Data
```bash
jupyter lab notebooks/01_eda_dan_validasi.ipynb
```

### Fase 2: Pengembangan Model Baseline
```bash
jupyter lab notebooks/02_baseline_model_development.ipynb
```

### Fase 3: Training Model Final
```bash
python src/models/train_models.py
```

### Fase 4: Prediksi
```bash
python src/models/predict_model.py --input [path_to_data]
```

## Model yang Digunakan
- **Baseline Models:** Random Forest, MLP
- **Advanced Models:** XGBoost, CatBoost
- **Ensemble:** Voting Classifier

## Metrik Evaluasi
- Accuracy
- F1-Score
- ROC-AUC
- Confusion Matrix

## Hasil
Hasil analisis dan perbandingan model dapat dilihat di:
- `notebooks/03_analisis_hasil_akhir.ipynb`
- `reports/figures/`

## Kontributor
- [Nama Anda]
- [Supervisor/Pembimbing]

## Lisensi
[Tipe Lisensi]

## Catatan Penting
- Pastikan data mentah sudah ditempatkan di folder `data/01_raw/`
- Model yang sudah dilatih akan disimpan di folder `models/`
- Semua visualisasi akan disimpan di folder `reports/figures/`