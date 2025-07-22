"""
Script untuk loading dan pemrosesan data mentah
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "01_raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "02_processed"

def load_raw_data(file_name, data_type='csv'):
    """
    Load data mentah dari folder 01_raw
    
    Parameters:
    -----------
    file_name : str
        Nama file yang akan dimuat
    data_type : str
        Tipe file ('csv' atau 'excel')
    
    Returns:
    --------
    pd.DataFrame
        Data yang sudah dimuat
    """
    file_path = RAW_DATA_DIR / file_name
    
    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    try:
        if data_type == 'csv':
            df = pd.read_csv(file_path)
            logger.info(f"✅ Berhasil memuat file CSV: {file_name}")
        elif data_type == 'excel':
            df = pd.read_excel(file_path)
            logger.info(f"✅ Berhasil memuat file Excel: {file_name}")
        else:
            raise ValueError("data_type harus 'csv' atau 'excel'")
        
        logger.info(f"📊 Shape data: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Error memuat file {file_name}: {str(e)}")
        raise

def load_2024_data():
    """Load data training 2024"""
    return load_raw_data("DATA TS SARJANA 2024.xlsx", "excel")

def load_2016_data():
    """Load data testing 2016"""
    return load_raw_data("data 2016 - daffari_raw.csv", "csv")

def get_data_info(df, data_name="Dataset"):
    """
    Tampilkan informasi dasar tentang dataset
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset yang akan dianalisis
    data_name : str
        Nama dataset untuk logging
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"📊 INFORMASI {data_name.upper()}")
    logger.info(f"{'='*50}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing values per column:")
    missing_info = df.isnull().sum()
    for col, missing in missing_info.items():
        if missing > 0:
            logger.info(f"  {col}: {missing} ({missing/len(df)*100:.1f}%)")
    
    logger.info(f"Data types:")
    for col, dtype in df.dtypes.items():
        logger.info(f"  {col}: {dtype}")

def identify_common_columns(df_2024, df_2016):
    """
    Identifikasi kolom yang sama antara kedua dataset
    
    Parameters:
    -----------
    df_2024 : pd.DataFrame
        Data 2024
    df_2016 : pd.DataFrame  
        Data 2016
        
    Returns:
    --------
    list
        List kolom yang sama
    """
    cols_2024 = set(df_2024.columns)
    cols_2016 = set(df_2016.columns)
    
    common_cols = list(cols_2024.intersection(cols_2016))
    
    logger.info(f"\n🔍 ANALISIS KOLOM:")
    logger.info(f"Kolom di data 2024: {len(cols_2024)}")
    logger.info(f"Kolom di data 2016: {len(cols_2016)}")
    logger.info(f"Kolom yang sama: {len(common_cols)}")
    
    if common_cols:
        logger.info(f"Kolom yang sama: {common_cols}")
    
    # Kolom yang hanya ada di 2024
    only_2024 = cols_2024 - cols_2016
    if only_2024:
        logger.info(f"Kolom hanya di 2024: {len(only_2024)} kolom")
    
    # Kolom yang hanya ada di 2016  
    only_2016 = cols_2016 - cols_2024
    if only_2016:
        logger.info(f"Kolom hanya di 2016: {len(only_2016)} kolom")
        logger.info(f"Detail: {list(only_2016)}")
    
    return common_cols

def save_processed_data(df, filename):
    """
    Simpan data yang sudah diproses ke folder 02_processed
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data yang akan disimpan
    filename : str
        Nama file output
    """
    # Pastikan direktori ada
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    file_path = PROCESSED_DATA_DIR / filename
    df.to_csv(file_path, index=False)
    logger.info(f"✅ Data tersimpan: {file_path}")

def basic_data_cleaning(df):
    """
    Pembersihan data dasar
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data yang akan dibersihkan
        
    Returns:
    --------
    pd.DataFrame
        Data yang sudah dibersihkan
    """
    df_clean = df.copy()
    
    # Hapus baris yang semuanya NaN
    df_clean = df_clean.dropna(how='all')
    
    # Hapus kolom yang semuanya NaN
    df_clean = df_clean.dropna(axis=1, how='all')
    
    # Trim whitespace untuk kolom string
    for col in df_clean.select_dtypes(include=['object']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Replace 'nan' string dengan NaN
        df_clean[col] = df_clean[col].replace('nan', np.nan)
    
    logger.info(f"🧹 Data cleaning: {df.shape} → {df_clean.shape}")
    
    return df_clean

def validate_target_column(df, target_col):
    """
    Validasi kolom target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataset
    target_col : str
        Nama kolom target
    """
    if target_col not in df.columns:
        logger.error(f"❌ Kolom target '{target_col}' tidak ditemukan!")
        logger.info(f"Kolom yang tersedia: {df.columns.tolist()}")
        return False
    
    # Analisis distribusi target
    target_dist = df[target_col].value_counts()
    logger.info(f"\n🎯 DISTRIBUSI TARGET '{target_col}':")
    for value, count in target_dist.items():
        percentage = count / len(df) * 100
        logger.info(f"  {value}: {count} ({percentage:.1f}%)")
    
    return True

if __name__ == "__main__":
    # Test loading data
    try:
        logger.info("🚀 Testing data loading...")
        
        # Load kedua dataset
        df_2024 = load_2024_data()
        df_2016 = load_2016_data()
        
        # Tampilkan info
        get_data_info(df_2024, "Data 2024")
        get_data_info(df_2016, "Data 2016")
        
        # Identifikasi kolom yang sama
        common_cols = identify_common_columns(df_2024, df_2016)
        
        # Pembersihan dasar
        df_2024_clean = basic_data_cleaning(df_2024)
        df_2016_clean = basic_data_cleaning(df_2016)
        
        logger.info("✅ Test berhasil!")
        
    except Exception as e:
        logger.error(f"❌ Test gagal: {str(e)}")
        logger.info("📝 Pastikan file data sudah ada di folder data/01_raw/")