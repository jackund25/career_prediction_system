# Contoh Penggunaan predict_model.py untuk DATA TS SARJANA 2024

## 1. Menggunakan Command Line Interface (CLI)

### Prediksi pada DATA TS SARJANA 2024:

```bash
python src/models/predict_model.py --predict-2024
```

### Prediksi pada file custom:

```bash
python src/models/predict_model.py --input "path/to/your/data.xlsx" --output "path/to/predictions.csv"
```

### Evaluasi jika data memiliki label:

```bash
python src/models/predict_model.py --input "path/to/test_data.xlsx" --evaluate
```

## 2. Menggunakan Function dalam Python Script

### Import yang diperlukan:

```python
import sys
sys.path.append('src')
from models.predict_model import predict_ts_sarjana_2024, ModelPredictor, make_prediction
import pandas as pd
```

### Prediksi pada DATA TS SARJANA 2024 (cara termudah):

```python
# Prediksi dan simpan hasil
results = predict_ts_sarjana_2024(save_results=True)

# Atau prediksi tanpa menyimpan
results = predict_ts_sarjana_2024(save_results=False)
```

### Prediksi pada DataFrame custom:

```python
# Muat data Anda
df = pd.read_excel('path/to/your/data.xlsx')

# Buat prediksi
predictions, probabilities = make_prediction(df)

# Atau gunakan ModelPredictor untuk kontrol lebih lanjut
predictor = ModelPredictor()
results = predictor.predict_and_save(df, 'output/predictions.csv')
```

### Evaluasi model pada data dengan label:

```python
predictor = ModelPredictor()
metrics = predictor.evaluate_on_test_data('path/to/test_data.xlsx')
print(metrics)
```

## 3. Output yang Dihasilkan

File hasil prediksi akan berisi:

- Semua kolom asli dari data input
- `prediction`: Hasil prediksi (ya/tidak atau sesuai/tidak sesuai)
- `probability_class_0`: Probabilitas untuk kelas 0
- `probability_class_1`: Probabilitas untuk kelas 1
- `confidence`: Tingkat kepercayaan prediksi (maksimum probabilitas)

## 4. Tips Penggunaan

1. **Format Data**: Pastikan data input memiliki struktur kolom yang sama dengan data training
2. **Missing Values**: Model akan menangani missing values secara otomatis
3. **Temporal Drift**: Model sudah didesain untuk menangani perbedaan format antara data 2016 dan 2024
4. **Performance**: Untuk dataset besar, gunakan CLI interface untuk efisiensi memori yang lebih baik

## 5. Troubleshooting

Jika ada error:

1. Pastikan model sudah dilatih dan file `final_model.pkl` tersedia di folder `models/`
2. Periksa format data input (CSV atau Excel)
3. Pastikan kolom yang diperlukan tersedia dalam data
4. Untuk data 2016, pastikan format kolom relationship sudah sesuai
