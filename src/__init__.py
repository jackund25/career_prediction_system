# src/__init__.py
"""
Career Predictions System

Sistem prediksi kelulusan tepat waktu menggunakan machine learning.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Konfigurasi path
import os
from pathlib import Path

# Root directory
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# Data paths
RAW_DATA_DIR = DATA_DIR / "01_raw"
PROCESSED_DATA_DIR = DATA_DIR / "02_processed"

# Ensure directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)