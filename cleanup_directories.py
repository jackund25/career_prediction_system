"""
Script untuk membersihkan direktori proyek dari file dan folder yang tidak berguna
Jalankan dengan hati-hati dan backup terlebih dahulu!

Usage:
    python cleanup_directories.py
"""

import os
import shutil
from pathlib import Path
import glob

def backup_important_files():
    """Backup file penting sebelum cleanup"""
    backup_dir = Path('backup_before_cleanup')
    backup_dir.mkdir(exist_ok=True)
    
    important_files = [
        'requirements.txt',
        'README.md', 
        'USAGE_PREDICT_MODEL.md'
    ]
    
    for file in important_files:
        if Path(file).exists():
            shutil.copy2(file, backup_dir / file)
    
    print(f"✅ Backup created in {backup_dir}")

def remove_duplicate_model_folders():
    """Hapus folder model duplikat"""
    folders_to_remove = [
        'models/with_leaky_column',
        'models/with_leaky_no_reg', 
        'models/without_leaky_column',
        'models/without_leaky_no_reg',
        'models/advanced_v2'  # Folder eksperimen lama
    ]
    
    for folder in folders_to_remove:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"🗑️ Removed: {folder}")

def remove_duplicate_plot_folders():
    """Hapus folder plot duplikat"""
    folders_to_remove = [
        'plots/with_leaky_column',
        'plots/with_leaky_no_reg',
        'plots/without_leaky_column', 
        'plots/without_leaky_no_reg',
        'plots/baseline',
        'plots/advanced_v2'  # Folder eksperimen lama
    ]
    
    for folder in folders_to_remove:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"🗑️ Removed: {folder}")

def remove_duplicate_result_folders():
    """Hapus folder results duplikat"""
    folders_to_remove = [
        'results/with_leaky_column',
        'results/with_leaky_no_reg',
        'results/without_leaky_column',
        'results/without_leaky_no_reg',
        'results/experiments',  # Folder kosong
        'results/advanced_v2'   # Folder eksperimen lama
    ]
    
    for folder in folders_to_remove:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"🗑️ Removed: {folder}")

def remove_other_folders():
    """Hapus folder lain yang tidak perlu"""
    folders_to_remove = [
        'catboost_info',
        'reports'  # duplikat dengan reports_figures (jika ada)
    ]
    
    files_to_remove = [
        'PROJECT_STRUCTURE.txt'  # File generated, bisa dibuat ulang
    ]
    
    for folder in folders_to_remove:
        if Path(folder).exists():
            shutil.rmtree(folder)
            print(f"🗑️ Removed: {folder}")
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️ Removed: {file}")

def clean_models_root():
    """Bersihkan file model duplikat di root models/"""
    files_to_remove = [
        'models/baseline_mlp.pkl',
        'models/baseline_rf.pkl', 
        'models/catboost_model.pkl',
        'models/ensemble_model.pkl',
        'models/feature_builder.pkl',
        'models/feature_engineer.pkl',
        'models/final_model.pkl',
        'models/label_encoder.pkl',
        'models/model_metadata.json', 
        'models/preprocessor.pkl',
        'models/scaler.pkl',
        'models/selected_features.json',
        'models/xgboost_model.pkl'
    ]
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️ Removed: {file}")

def clean_processed_data():
    """Bersihkan data/02_processed yang tidak perlu"""
    files_to_remove = [
        'data/02_processed/baseline_results.csv',
        'data/02_processed/benchmark_metrics.json',
        'data/02_processed/column_analysis.json',
        'data/02_processed/data_compatibility_report.json',
        'data/02_processed/feature_importance.csv',
        'data/02_processed/model_results.csv',
        'data/02_processed/predictions_2016.csv',
        'data/02_processed/predictions_2024_ts_sarjana.csv',
        'data/02_processed/shap_summary.png'
    ]
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️ Removed: {file}")

def clean_old_predictions():
    """Hapus file prediksi lama, simpan hanya yang terbaru"""
    pred_dir = Path('results/predictions')
    if not pred_dir.exists():
        return
        
    # Ambil semua file CSV prediksi
    csv_files = list(pred_dir.glob('predictions_*.csv'))
    json_files = list(pred_dir.glob('predictions_*.json'))
    
    if len(csv_files) > 2:  # Simpan 2 file terbaru
        # Sort berdasarkan waktu modifikasi
        csv_files.sort(key=lambda x: x.stat().st_mtime)
        json_files.sort(key=lambda x: x.stat().st_mtime)
        
        # Hapus yang lama
        for file in csv_files[:-2]:  # Hapus semua kecuali 2 terakhir
            file.unlink()
            print(f"🗑️ Removed old prediction: {file.name}")
            
        for file in json_files[:-2]:  # Hapus semua kecuali 2 terakhir  
            file.unlink()
            print(f"🗑️ Removed old prediction: {file.name}")

def clean_old_figures():
    """Bersihkan visualisasi lama di reports_figures"""
    fig_dir = Path('reports_figures')
    if not fig_dir.exists():
        return
    
    # Kelompokkan berdasarkan jenis visualisasi
    figure_types = ['model_comparison', 'confusion_matrix', 'feature_importance', 
                   'prediction_distribution', 'temporal_analysis']
    
    for fig_type in figure_types:
        # Ambil semua file untuk jenis ini
        png_files = list(fig_dir.glob(f'{fig_type}_*.png'))
        pdf_files = list(fig_dir.glob(f'{fig_type}_*.pdf'))
        
        if len(png_files) > 1:  # Simpan hanya 1 terbaru
            png_files.sort(key=lambda x: x.stat().st_mtime)
            for file in png_files[:-1]:  # Hapus semua kecuali terakhir
                file.unlink()
                print(f"🗑️ Removed old figure: {file.name}")
                
        if len(pdf_files) > 1:  # Simpan hanya 1 terbaru
            pdf_files.sort(key=lambda x: x.stat().st_mtime) 
            for file in pdf_files[:-1]:  # Hapus semua kecuali terakhir
                file.unlink()
                print(f"🗑️ Removed old figure: {file.name}")

def clean_logs_folder():
    """Bersihkan log lama, simpan yang terbaru"""
    log_dir = Path('logs')
    if not log_dir.exists():
        return
    
    log_files = list(log_dir.glob('training_*.log'))
    
    if len(log_files) > 2:  # Simpan 2 terbaru
        log_files.sort(key=lambda x: x.stat().st_mtime)
        for file in log_files[:-2]:  # Hapus semua kecuali 2 terakhir
            file.unlink()
            print(f"🗑️ Removed old log: {file.name}")

def clean_root_scripts():
    """Hapus script sementara di root"""
    files_to_remove = [
        'analyze_distribution.py',
        'data_leakage_checkers.py', 
        'generate_structure.py'
    ]
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️ Removed: {file}")

def add_folder_structure():
    """Tambahkan folder yang mungkin diperlukan"""
    folders_to_create = [
        'data/03_features',  # Folder untuk menyimpan feature engineered data
        'src/scripts'        # Untuk utility scripts
    ]
    
    for folder in folders_to_create:
        Path(folder).mkdir(parents=True, exist_ok=True)
        gitkeep = Path(folder) / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
        print(f"✅ Created/ensured: {folder}")
    """Hapus script sementara di root"""
    files_to_remove = [
        'analyze_distribution.py',
        'data_leakage_checkers.py', 
        'generate_structure.py'
    ]
    
    for file in files_to_remove:
        if Path(file).exists():
            os.remove(file)
            print(f"🗑️ Removed: {file}")

def main():
    """Jalankan semua proses cleanup"""
    print("🧹 Starting directory cleanup...")
    print("⚠️  PERINGATAN: Script ini akan menghapus banyak file!")
    
    response = input("Lanjutkan? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Cleanup dibatalkan")
        return
    
    # Backup dulu
    backup_important_files()
    
    # Cleanup
    remove_duplicate_model_folders()
    remove_duplicate_plot_folders() 
    remove_duplicate_result_folders()
    remove_other_folders()
    clean_models_root()
    clean_processed_data()
    clean_old_predictions()
    clean_old_figures()
    clean_logs_folder()
    clean_root_scripts()
    
    # Tambahkan struktur yang diperlukan
    add_folder_structure()
    
    print("\n✅ Cleanup selesai!")
    print("\n📊 Struktur direktori yang tersisa:")
    
    # Tampilkan struktur yang tersisa
    try:
        # Windows command
        import subprocess
        result = subprocess.run(['dir', '/s', '/b'], 
                              capture_output=True, text=True, shell=True)
        if result.returncode == 0:
            # Filter only directories
            lines = result.stdout.split('\n')
            dirs = [line for line in lines if line and not '.' in Path(line).name]
            for dir_path in dirs[:20]:  # Show first 20 directories
                print(f"  {dir_path}")
        else:
            print("Tidak bisa menampilkan struktur direktori")
    except Exception as e:
        print(f"Error menampilkan struktur: {e}")

if __name__ == "__main__":
    main()