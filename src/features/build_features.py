"""
Module untuk feature engineering dengan mapping kategori yang lebih komprehensif
Updated with column filtering and data leakage control
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Union, Tuple, Optional
import logging
from pathlib import Path
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== KONSTANTA KOLOM TARGET =====
# 12 kolom yang akan digunakan untuk kedua dataset
TARGET_COLUMNS = [
    'NIM',
    'Program Studi',
    'IP',
    'Apakah pekerjaan yang Anda lakukan di tempat bekerja sesuai dengan bidang keilmuan?',
    'Seberapa erat hubungan bidang studi dengan pekerjaan Anda?',
    'Seberapa besar program studi Anda bermanfaat untuk hal-hal di bawah ini? [memulai pekerjaan]',
    'Seberapa besar program studi Anda bermanfaat untuk hal-hal di bawah ini? [pembelajaran yang berkelanjutan dalam pekerjaan]',
    'Bidang usaha bekerja',
    'Organisasi apa yang paling aktif Anda ikuti selama menjalani perkuliahan? (nama organisasi)',
    'Sebutkan jenis kegiatan di organisasi yang aktif diikuti yang membantu mengasah kemampuan/skill Anda!',
    'Sebutkan kategori jenis pekerjaan yang Anda lakukan di tempat bekerja!',
    'Jelaskan tugas-tugas utama dalam pekerjaan Anda saat ini?'
]

# Kolom status pekerjaan
EMPLOYMENT_STATUS_COLUMN = 'Pekerjaan utama saat ini?'

# Nilai status pekerjaan yang diterima (alumni yang bekerja)
WORKING_STATUS_VALUES = ['Bekerja', 'Bekerja dan wiraswasta']

# Kolom yang berpotensi leaky
LEAKY_COLUMN = 'Seberapa erat hubungan bidang studi dengan pekerjaan Anda?'

# Target variable
TARGET_VARIABLE = 'Apakah pekerjaan yang Anda lakukan di tempat bekerja sesuai dengan bidang keilmuan?'

# ===== KONSTANTA KATEGORI (existing constants remain the same) =====

# 1. Program Studi - 45 values
PROGRAM_STUDI_VALUES = [
    'Meteorologi', 'Oseanografi', 'Teknik Geodesi dan Geomatika', 'Teknik Geologi',
    'Aktuaria', 'Astronomi', 'Fisika', 'Kimia', 'Matematika', 'Desain Interior',
    'Desain Komunikasi Visual', 'Desain Produk', 'Kriya', 'Seni Rupa',
    'Teknik Dirgantara', 'Teknik Material', 'Teknik Mesin', 'Teknik Geofisika',
    'Teknik Metalurgi', 'Teknik Perminyakan', 'Teknik Pertambangan',
    'Rekayasa Infrastruktur Lingkungan', 'Teknik dan Pengelolaan Sumber Daya Air',
    'Teknik Kelautan', 'Teknik Lingkungan', 'Teknik Sipil', 'Manajemen Rekayasa Industri',
    'Teknik Bioenergi dan Kemurgi', 'Teknik Fisika', 'Teknik Industri', 'Teknik Kimia',
    'Teknik Pangan', 'Arsitektur', 'Perencanaan Wilayah dan Kota', 'Kewirausahaan',
    'Manajemen', 'Farmasi Klinik dan Komunitas', 'Sains dan Teknologi Farmasi',
    'Biologi', 'Mikrobiologi', 'Rekayasa Hayati', 'Rekayasa Pertanian',
    'Rekayasa Kehutanan', 'Teknologi Pascapanen', 'Sistem dan Teknologi Informasi',
    'Teknik Biomedis', 'Teknik Elektro', 'Informatika', 'Teknik Telekomunikasi',
    'Teknik Tenaga Listrik', 'Lainnya'
]

# 2. Bidang Usaha Bekerja - 21 values
BIDANG_USAHA_VALUES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'Lainnya'
]

# 3. Kelompok Pekerjaan
KELOMPOK_PEKERJAAN = {
    'Teknologi Informasi': [
        'signalling engineer', 'Teknologi Informasi', 'IT App Development', 'IT',
        'Software engineer', 'informasi teknologi', 'Ngoding', 'IoT Engineer',
        'Software Developer', 'Web developer', 'Programming', 'Software Engineering',
        'Pemrograman', 'Aplikasi', 'Programmer', 'Application Support',
        'Tenaga Profesional (Bidang IT)', 'Software', 'SQL Developer',
        'Software Robotic Engineer', 'Autonomous Vehicle Software Engineer - Localization',
        'Backend Engineer', 'Kecerdasan Buatan', 'AI Engineer', 'Machine Learning Engineer',
        'Game Programming', 'Solution Architect', 'Technology Consulting',
        'embedded system engineer', 'System Design, Software Engineer',
        'Android Developer', 'Cloud Engineer', 'Network Engineer', 'IT security',
        'Otomasi dan Kontrol Industri', 'Data Engineer, Data Science, Software Engineering',
        'AI', 'Developer', 'Application development', 'System Controls Engineer',
        'Automation Engineer', 'Instrument Engineering', 'Control system',
        'data engineer', 'IT Data Engineer', 'Kontrol Sistem', 'Instrumentasi dan kontrol',
        'Database Administrator', 'Front End Developer',
        'System Design, Analyze, Managing Project, Propose an Automation Process',
        'IT Software', 'Pengembangan Aplikasi', 'Back End Engineer',
        'Software Quality Assurance', 'pengembangan perangkat lunak',
        'Engineering Productivity', 'Frontend Developer', 'iOS Developer',
        'Information Technology', 'Machine learning engineering',
        'Software Development Engineer in Test', 'Consultant IT', 'System Administrator',
        'Cloud Computing', 'System Engineer', 'Software Tester',
        'Pengembangan sistem informasi', 'IT ( Application Development) - Software Engineer'
    ],
    'Manajemen dan Bisnis': [
        'Manajemen dan Bisnis', 'Manajer', 'Strategic & Planning', 'Strategi',
        'Strategy Planning', 'management', 'Manajemen dan Perencanaan',
        'Management Trainee', 'Business Development', 'Supervisor',
        'Managing, coaching, mentoring', 'Managerial', 'Managing', 'Manager Kredit',
        'Operation support', 'Operations & Product Manager Associate',
        'Operation', 'Business Strategist', 'Manajemen',
        'Business Development (E-commerce : Jasa dan Teknologi)', 'Manajemen Produksi',
        'business development', 'Operations', 'Bisnis Analis', 'Business Planning',
        'Business Architect', 'Manajemen proyek, perencanaan bisnis, manajemen operasionsl',
        'problem solving', 'Demand planner', 'Supply Chain'
    ],
    'Analis': [
        'Analisa', 'analisa bisnis', 'Analisa Data', 'Analis kredit (akutansi & ekonomi)',
        'Data Analisis', 'Pricing Analyst', 'Data Analyst', 'Analis keuangan',
        'Analisis Data', 'Model risiko', 'Data Operation', 'analisa data',
        'Data Analis', 'Data Scientist', 'Strategic dan Analysis', 'Business Analyst',
        'Analisis data, pengembangan dan implementasi model machine learning',
        'Lab analyst', 'Analis Laboratorium', 'Surveillance Analyst',
        'Equity Research Analysis', 'data analyst', 'Data Analysis',
        'Analis Kebijakan (PNS)', 'Equity Research Analyst', 'Porepressure Analyst',
        'Bussinies Analyst', 'Analis Perencanaan', 'Credit Analysis', 'Analyst',
        'Analisis & Interpretasi', 'Product Analyst', 'Data Analyst Hulu Migas',
        'Analis Tambang', 'Data Engineering',
        'Data science, Data analytics, Data engineering, Market research',
        'Analisis data', 'Meocean Data Aanalysis', 'Analisa Sistem',
        'Business Analytics', 'Financial Analyst', 'Analytic & Relationship',
        'Campaign Analysis', 'Orbit Analyst', 'Marketing Analyst',
        'business analyst dalam perbankan'
    ],
    'Teknik': [
        'Naval Architect, construction, offshore installation', 'Teknik',
        'Insinyur kelautan', 'Geophysics', 'Engineer', 'Geotechnical engineer',
        'Engineering', 'Petroleum Engineer', 'Mining Engineer', 'Civil engineer',
        'Structural Engineer', 'Electrical Engineer', 'Process Engineer',
        'Mechanical engineering', 'Reservoir Engineer', 'Teknisi', 'Metallurgist',
        'Piping Engineer'
    ],
    'Konsultan': [
        'Konsultan', 'konsultan', 'Business Consultant', 'IT Consulting Firm',
        'Konsultan risiko keuangan', 'Consultant Staff', 'Consulting',
        'konsultan riset instrumen finansial', 'Konsultan Tambang',
        'konsultan pertambangan', 'Management Consulting', 'Konsultan Geosains dan Oseanografi',
        'Konsultan Geoteknik', 'Advisory', 'Konsultansi', 'Konsultan Manajemen dan Bisnis',
        'Konsultan Manajemen', 'Management Consultant', 'Sureveyor dan Konsultan',
        'Technology Consulting - ERP Implementation', 'Konsultan Human Resource',
        'Konsultan dan Trainer', 'Konsultansi Teknik Sipil', 'Konsultan Bantuan Teknis',
        'Konsultan teknik', 'Konsultasi Bisnis', 'akustik konsultan',
        'Konsultan dan Kontraktor', 'Konsultan Individu', 'Environmental Consultant',
        'Konsultan Rancang Kota dan Perencanaan', 'Jasa Konsultansi',
        'Konsultan Infrastruktur', 'Jasa Konsultasi Pekerjaan Pelabuhan',
        'Engineering Consultant',
        'Konsultan air bersih serta air limbah, terkadang limbah padat di Kawasan Industri'
    ],
    'Sales dan Marketing': [
        'Media', 'Sales', 'Marketing', 'Business Development', 'Account Manager',
        'Sales Representative', 'Marketing Specialist', 'Digital Marketing'
    ],
    'Riset dan Pengembangan': [
        'Riset sekunder, Riset primer, Analisis dan Sintesis, Presentasi',
        'Research', 'Development', 'R&D', 'Researcher', 'Research Assistant'
    ],
    'Energi dan Pertambangan': [
        'Energi Terbarukan', 'Mining', 'Energy', 'Oil', 'Gas', 'Renewable Energy',
        'Power Plant', 'Mining Engineer', 'Petroleum'
    ],
    'Kesenian dan Desain': [
        'Art Director', 'Design', 'Creative', 'Artist', 'Graphic Designer',
        'UI/UX Designer', 'Interior Designer'
    ],
    'Keuangan': [
        'Finance', 'Banking', 'Financial', 'Investment', 'Accounting',
        'Audit', 'Treasury', 'Financial Analyst'
    ],
    'Produksi': [
        'Pengawas Produksi', 'Production', 'Manufacturing', 'Quality Control',
        'Production Manager', 'Plant Manager'
    ],
    'Jasa Profesional': [
        'Pelayanan masyarakat', 'Professional Services', 'Consulting',
        'Legal Services', 'Healthcare', 'Education'
    ],
    'Lainnya': []
}

# 4. Konstanta untuk kategori lainnya
RELATIONSHIP_CATEGORIES = ['tidak sama sekali', 'kurang erat', 'cukup erat', 'erat', 'sangat erat']

IP_CATEGORIES = {
    'SANGAT_MEMUASKAN': 3.5,
    'MEMUASKAN': 3.0,
    'CUKUP': 2.75,
    'KURANG': 2.0
}

ORGANIZATION_CATEGORIES = {
    'TIDAK_MENGIKUTI': 'Tidak Mengikuti',
    'HIMPUNAN': 'Himpunan',
    'NON_HIMPUNAN': 'Non Himpunan'
}

ACTIVITY_CATEGORIES = {
    'WORKSHOP': 'Workshop',
    'PENGABDIAN_MASYARAKAT': 'Pengabdian Masyarakat',
    'KEPANITIAAN': 'Kepanitiaan',
    'KEPENGURUSAN': 'Kepengurusan',
    'KOMPETISI': 'Kompetisi',
    'SEMINAR': 'Seminar',
    'PELATIHAN': 'Pelatihan',
    'TIDAK_MENGIKUTI': 'Tidak Mengikuti',
    'LAINNYA': 'Lainnya'
}

# Keywords untuk mencari jenis aktivitas
ACTIVITY_KEYWORDS = {
    'workshop': ['workshop'],
    'pengabdian': ['pengabdian', 'masyarakat', 'bakti', 'sosial'],
    'kepanitiaan': ['panitia', 'event', 'acara'],
    'kepengurusan': ['pengurus', 'ketua', 'wakil', 'sekretaris', 'bendahara', 'kepala'],
    'kompetisi': ['kompetisi', 'lomba', 'contest'],
    'seminar': ['seminar'],
    'pelatihan': ['pelatihan', 'training']
}

# Keywords untuk mencari jenis organisasi
ORGANIZATION_KEYWORDS = {
    'himpunan': ['himpunan', 'hm', 'km']
}

# Default values
DEFAULT_VALUES = {
    'binary': 'tidak',
    'relationship': 'tidak sama sekali',
    'categorical': 'Tidak Ada',
    'text': 'tidak ada deskripsi',
    'organization': 'Tidak Mengikuti',
    'activity': 'Tidak Mengikuti',
    'ip': 'Tidak Ada',
    'company_count': 'Tidak Ditulis',
    'program_studi': 'Lainnya',
    'bidang_usaha': 'Lainnya',
    'job_category': 'Lainnya'
}


class FeatureEngineer:
    """Class untuk melakukan feature engineering dengan kontrol data leakage"""
    
    def __init__(self, include_leaky_column: bool = True):
        """
        Initialize feature engineer
        
        Args:
            include_leaky_column: Whether to include the potentially leaky column
        """
        self.include_leaky_column = include_leaky_column
        self.feature_mappings = self._load_feature_mappings()
        
        logger.info(f"Feature Engineer initialized with include_leaky_column={include_leaky_column}")

    def filter_working_alumni(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter dataframe to only include alumni who are working
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame filtered to only working alumni
        """
        # Check if employment status column exists
        if EMPLOYMENT_STATUS_COLUMN not in df.columns:
            logger.warning(f"Employment status column '{EMPLOYMENT_STATUS_COLUMN}' not found in dataframe")
            logger.warning("Available columns: " + ", ".join(df.columns))
            logger.warning("Returning original dataframe without filtering")
            return df
        
        # Get initial count
        initial_count = len(df)
        
        # Filter for working alumni
        df_filtered = df[df[EMPLOYMENT_STATUS_COLUMN].isin(WORKING_STATUS_VALUES)].copy()
        
        # Get final count
        final_count = len(df_filtered)
        
        # Log filtering results
        logger.info(f"Filtered alumni by employment status:")
        logger.info(f"  - Initial count: {initial_count}")
        logger.info(f"  - Final count (working alumni): {final_count}")
        logger.info(f"  - Removed: {initial_count - final_count}")
        logger.info(f"  - Retention rate: {final_count/initial_count*100:.1f}%")
        
        # Show distribution of employment status in original data
        if EMPLOYMENT_STATUS_COLUMN in df.columns:
            logger.info(f"\nEmployment status distribution (before filtering):")
            status_counts = df[EMPLOYMENT_STATUS_COLUMN].value_counts()
            for status, count in status_counts.items():
                logger.info(f"  - {status}: {count} ({count/initial_count*100:.1f}%)")
        
        return df_filtered
        
    def _load_feature_mappings(self) -> Dict:
        """Load feature mappings dari konstanta"""
        mappings = {
            'program_studi': PROGRAM_STUDI_VALUES,
            'bidang_usaha': BIDANG_USAHA_VALUES,
            'kelompok_pekerjaan': KELOMPOK_PEKERJAAN,
            'relationship': RELATIONSHIP_CATEGORIES,
            'ip_categories': IP_CATEGORIES,
            'organization': ORGANIZATION_CATEGORIES,
            'activity': ACTIVITY_CATEGORIES,
            'activity_keywords': ACTIVITY_KEYWORDS,
            'organization_keywords': ORGANIZATION_KEYWORDS,
            'defaults': DEFAULT_VALUES
        }
        return mappings
    
    def filter_target_columns(self, df: pd.DataFrame, is_2017_data: bool = False) -> pd.DataFrame:
        """
        Filter dataframe to only include the 12 target columns
        
        Args:
            df: Input DataFrame
            is_2017_data: Whether this is 2017 data (which needs column name mapping)
            
        Returns:
            DataFrame with only target columns
        """
        df_filtered = pd.DataFrame()
        
        # Column name mappings for 2017 data (if column names are different)
        column_mappings_2017 = {
            # Add any column name mappings here if 2017 data has different names
            # Example: 'Old Name': 'New Name'
        }
        
        # Apply column name mappings if this is 2017 data
        if is_2017_data and column_mappings_2017:
            df = df.rename(columns=column_mappings_2017)
        
        # Get columns to use (exclude leaky column if specified)
        columns_to_use = TARGET_COLUMNS.copy()
        if not self.include_leaky_column and LEAKY_COLUMN in columns_to_use:
            columns_to_use.remove(LEAKY_COLUMN)
            logger.info(f"Excluded leaky column: {LEAKY_COLUMN}")
        
        # Select only columns that exist
        missing_columns = []
        for col in columns_to_use:
            if col in df.columns:
                df_filtered[col] = df[col]
            else:
                missing_columns.append(col)
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            
        logger.info(f"Filtered to {len(df_filtered.columns)} columns out of {len(df.columns)} original columns")
        logger.info(f"Selected columns: {list(df_filtered.columns)}")
        
        return df_filtered
    
    def clean_text(self, text: str) -> str:
        """Clean text data"""
        if pd.isna(text) or text == '' or text is None:
            return ''
        
        text = str(text).lower().strip()
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text
    
    def map_program_studi(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map program studi to standardized values
        
        Args:
            df: DataFrame input
            col_name: Nama kolom program studi
            
        Returns:
            DataFrame dengan kolom program_studi_clean
        """
        # Find column name if not specified
        if col_name is None:
            possible_names = ['Program Studi', 'program_studi', 'prodi', 'Program studi']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("Program Studi column not found")
            return df
            
        df = df.copy()
        
        # Clean and standardize
        df['program_studi_clean'] = df[col_name].apply(self.clean_text)
        
        # Map to standardized values
        def map_prodi(value):
            if not value:
                return DEFAULT_VALUES['program_studi']
            
            # Exact match (case insensitive)
            for prodi in PROGRAM_STUDI_VALUES:
                if value.lower() == prodi.lower():
                    return prodi
            
            # Partial match
            for prodi in PROGRAM_STUDI_VALUES:
                if prodi.lower() in value or value in prodi.lower():
                    return prodi
            
            return DEFAULT_VALUES['program_studi']
        
        df['program_studi_clean'] = df['program_studi_clean'].apply(map_prodi)
        
        logger.info(f"Program Studi distribution:\n{df['program_studi_clean'].value_counts().head()}")
        
        return df
    
    def map_relationship_level(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map relationship level between study and work
        Handles both 2024 format (text) and 2016 format (item1-item5)
        
        Args:
            df: DataFrame input
            col_name: Nama kolom hubungan
            
        Returns:
            DataFrame dengan kolom relationship_level
        """
        # Skip if leaky column is excluded
        if not self.include_leaky_column:
            logger.info("Skipping relationship level mapping (leaky column excluded)")
            return df
            
        if col_name is None:
            possible_names = ['Seberapa erat hubungan bidang studi dengan pekerjaan Anda?',
                            'hubungan_bidang_studi', 'relationship']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("Relationship column not found")
            return df
            
        df = df.copy()
        
        # Clean and map
        df['relationship_level'] = df[col_name].apply(self.clean_text)
        
        # Mapping for 2016 data format (item1-item5)
        item_mapping = {
            'item1': 'sangat erat',
            'item2': 'erat',
            'item3': 'cukup erat',
            'item4': 'kurang erat',
            'item5': 'tidak sama sekali'
        }
        
        # Map to standardized values
        def map_relationship(value):
            if not value:
                return DEFAULT_VALUES['relationship']
            
            # Check if it's 2016 format (item1-item5)
            if value in item_mapping:
                return item_mapping[value]
            
            # Check for 2024 format (text descriptions)
            for rel in RELATIONSHIP_CATEGORIES:
                if rel in value:
                    return rel
            
            return DEFAULT_VALUES['relationship']
        
        df['relationship_level'] = df['relationship_level'].apply(map_relationship)
        
        # Convert to ordinal
        relationship_mapping = {cat: i for i, cat in enumerate(RELATIONSHIP_CATEGORIES)}
        df['relationship_level_ordinal'] = df['relationship_level'].map(relationship_mapping)
        
        logger.info(f"Relationship level distribution:\n{df['relationship_level'].value_counts()}")
        
        return df
    
    def map_bidang_usaha(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map bidang usaha to standardized categories
        
        Args:
            df: DataFrame input
            col_name: Nama kolom bidang usaha
            
        Returns:
            DataFrame dengan kolom bidang_usaha_clean
        """
        if col_name is None:
            possible_names = ['Bidang usaha bekerja', 'bidang_usaha', 'Bidang Usaha']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("Bidang usaha column not found")
            return df
            
        df = df.copy()
        
        # Clean
        df['bidang_usaha_clean'] = df[col_name].apply(self.clean_text).str.upper()
        
        # Map to standardized values
        def map_bidang(value):
            if not value:
                return DEFAULT_VALUES['bidang_usaha']
            
            # Check if it's a single letter A-U
            if len(value) == 1 and value in BIDANG_USAHA_VALUES[:-1]:
                return value
            
            return DEFAULT_VALUES['bidang_usaha']
        
        df['bidang_usaha_clean'] = df['bidang_usaha_clean'].apply(map_bidang)
        
        logger.info(f"Bidang usaha distribution:\n{df['bidang_usaha_clean'].value_counts()}")
        
        return df
    
    def map_pekerjaan_ke_kelompok(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map pekerjaan to job categories
        
        Args:
            df: DataFrame input
            col_name: Nama kolom pekerjaan
            
        Returns:
            DataFrame dengan kolom job_category
        """
        if col_name is None:
            possible_names = ['Sebutkan kategori jenis pekerjaan yang Anda lakukan di tempat bekerja!',
                            'pekerjaan', 'Pekerjaan', 'job', 'Job']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("Pekerjaan column not found")
            df['job_category'] = DEFAULT_VALUES['job_category']
            return df
            
        df = df.copy()
        
        # Clean
        df['pekerjaan_clean'] = df[col_name].apply(self.clean_text)
        
        # Map to categories
        def categorize_job(job_text):
            if not job_text:
                return DEFAULT_VALUES['job_category']
            
            # Check each category
            for category, keywords in KELOMPOK_PEKERJAAN.items():
                for keyword in keywords:
                    if keyword.lower() in job_text or job_text in keyword.lower():
                        return category
            
            return DEFAULT_VALUES['job_category']
        
        df['job_category'] = df['pekerjaan_clean'].apply(categorize_job)
        
        logger.info(f"Job category distribution:\n{df['job_category'].value_counts()}")
        
        return df
    
    def map_organization(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map organization to categories
        
        Args:
            df: DataFrame input
            col_name: Nama kolom organisasi
            
        Returns:
            DataFrame dengan kolom organization_category
        """
        if col_name is None:
            possible_names = ['Organisasi apa yang paling aktif Anda ikuti selama menjalani perkuliahan? (nama organisasi)',
                            'organisasi', 'Organisasi']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("Organization column not found")
            return df
            
        df = df.copy()
        
        # Clean
        df['org_clean'] = df[col_name].apply(self.clean_text)
        
        # Categorize
        def categorize_org(org_text):
            if not org_text or org_text in ['tidak', 'tidak ada', 'tidak mengikuti']:
                return ORGANIZATION_CATEGORIES['TIDAK_MENGIKUTI']
            
            # Check for himpunan
            for keyword in ORGANIZATION_KEYWORDS['himpunan']:
                if keyword in org_text:
                    return ORGANIZATION_CATEGORIES['HIMPUNAN']
            
            return ORGANIZATION_CATEGORIES['NON_HIMPUNAN']
        
        df['organization_category'] = df['org_clean'].apply(categorize_org)
        
        logger.info(f"Organization category distribution:\n{df['organization_category'].value_counts()}")
        
        return df
    
    def map_ip_category(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map IP to categories
        
        Args:
            df: DataFrame input
            col_name: Nama kolom IP
            
        Returns:
            DataFrame dengan kolom ip_category
        """
        if col_name is None:
            possible_names = ['IP', 'ip', 'IPK', 'ipk']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("IP column not found")
            return df
            
        df = df.copy()
        
        # Try to extract numeric IP value
        def extract_ip(value):
            if pd.isna(value):
                return None
            
            value = str(value)
            # Try to find float pattern
            match = re.search(r'(\d+[.,]\d+)', value)
            if match:
                ip_str = match.group(1).replace(',', '.')
                try:
                    return float(ip_str)
                except:
                    return None
            return None
        
        df['ip_numeric'] = df[col_name].apply(extract_ip)
        
        # Categorize
        def categorize_ip(ip_val):
            if pd.isna(ip_val):
                return DEFAULT_VALUES['ip']
            
            if ip_val >= IP_CATEGORIES['SANGAT_MEMUASKAN']:
                return 'Sangat Memuaskan'
            elif ip_val >= IP_CATEGORIES['MEMUASKAN']:
                return 'Memuaskan'
            elif ip_val >= IP_CATEGORIES['CUKUP']:
                return 'Cukup'
            else:
                return 'Kurang'
        
        df['ip_category'] = df['ip_numeric'].apply(categorize_ip)
        
        logger.info(f"IP category distribution:\n{df['ip_category'].value_counts()}")
        
        return df
    
    def map_activity_type(self, df: pd.DataFrame, col_name: str = None) -> pd.DataFrame:
        """
        Map activity type to categories
        
        Args:
            df: DataFrame input
            col_name: Nama kolom kegiatan
            
        Returns:
            DataFrame dengan kolom activity_type
        """
        if col_name is None:
            possible_names = ['Sebutkan jenis kegiatan di organisasi yang aktif diikuti yang membantu mengasah kemampuan/skill Anda!',
                            'kegiatan_organisasi', 'kegiatan', 'Kegiatan']
            for name in possible_names:
                if name in df.columns:
                    col_name = name
                    break
        
        if col_name is None or col_name not in df.columns:
            logger.warning("Activity column not found")
            return df
            
        df = df.copy()
        
        # Clean
        df['activity_clean'] = df[col_name].apply(self.clean_text)
        
        # Categorize
        def categorize_activity(activity_text):
            if not activity_text or activity_text in ['tidak', 'tidak ada']:
                return ACTIVITY_CATEGORIES['TIDAK_MENGIKUTI']
            
            # Check each category
            for activity_type, keywords in ACTIVITY_KEYWORDS.items():
                for keyword in keywords:
                    if keyword in activity_text:
                        # Map to proper category name
                        if activity_type == 'workshop':
                            return ACTIVITY_CATEGORIES['WORKSHOP']
                        elif activity_type == 'pengabdian':
                            return ACTIVITY_CATEGORIES['PENGABDIAN_MASYARAKAT']
                        elif activity_type == 'kepanitiaan':
                            return ACTIVITY_CATEGORIES['KEPANITIAAN']
                        elif activity_type == 'kepengurusan':
                            return ACTIVITY_CATEGORIES['KEPENGURUSAN']
                        elif activity_type == 'kompetisi':
                            return ACTIVITY_CATEGORIES['KOMPETISI']
                        elif activity_type == 'seminar':
                            return ACTIVITY_CATEGORIES['SEMINAR']
                        elif activity_type == 'pelatihan':
                            return ACTIVITY_CATEGORIES['PELATIHAN']
            
            return ACTIVITY_CATEGORIES['LAINNYA']
        
        df['activity_type'] = df['activity_clean'].apply(categorize_activity)
        
        logger.info(f"Activity type distribution:\n{df['activity_type'].value_counts()}")
        
        return df
    
    def handle_ordinal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle ordinal features (1-5 scale)
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame with processed ordinal features
        """
        df = df.copy()
        
        # Find ordinal columns (usually contain "bermanfaat" and have 1-5 scale)
        ordinal_patterns = ['bermanfaat', 'manfaat', 'seberapa besar']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in ordinal_patterns):
                # Clean and convert to numeric
                df[f'{col}_numeric'] = pd.to_numeric(df[col], errors='coerce')
                
                # Fill missing with median
                if df[f'{col}_numeric'].notna().sum() > 0:
                    median_val = df[f'{col}_numeric'].median()
                    df[f'{col}_numeric'].fillna(median_val, inplace=True)
                
                logger.info(f"Processed ordinal column: {col}")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important columns
        
        Args:
            df: DataFrame input
            
        Returns:
            DataFrame dengan interaction features
        """
        df = df.copy()
        
        # Only create relationship-based interactions if leaky column is included
        if self.include_leaky_column:
            # Interaction between program studi match and relationship level
            if 'relationship_level_ordinal' in df.columns and 'program_studi_clean' in df.columns:
                # Create binary feature for technical program
                tech_programs = ['Teknik', 'Informatika', 'Sistem']
                df['is_technical'] = df['program_studi_clean'].apply(
                    lambda x: 1 if any(tech in x for tech in tech_programs) else 0
                )
                
                df['technical_x_relationship'] = df['is_technical'] * df['relationship_level_ordinal']
        else:
            # Create technical program feature without relationship interaction
            if 'program_studi_clean' in df.columns:
                tech_programs = ['Teknik', 'Informatika', 'Sistem']
                df['is_technical'] = df['program_studi_clean'].apply(
                    lambda x: 1 if any(tech in x for tech in tech_programs) else 0
                )
        
        # Interaction between IP and organization participation
        if 'ip_numeric' in df.columns and 'organization_category' in df.columns:
            df['ip_x_org_active'] = df['ip_numeric'] * (df['organization_category'] != 'Tidak Mengikuti').astype(int)
        
        logger.info("Created interaction features")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'smart') -> pd.DataFrame:
        """
        Handle missing values dengan strategi yang tepat
        
        Args:
            df: DataFrame input
            strategy: 'smart', 'mean', 'median', 'mode', 'drop'
            
        Returns:
            DataFrame dengan missing values handled
        """
        df = df.copy()
        
        if strategy == 'smart':
            # Handle each type of column appropriately
            
            # Categorical columns with default values
            categorical_defaults = {
                'program_studi_clean': DEFAULT_VALUES['program_studi'],
                'bidang_usaha_clean': DEFAULT_VALUES['bidang_usaha'],
                'job_category': DEFAULT_VALUES['job_category'],
                'organization_category': DEFAULT_VALUES['organization'],
                'activity_type': DEFAULT_VALUES['activity'],
                'ip_category': DEFAULT_VALUES['ip'],
                'relationship_level': DEFAULT_VALUES['relationship']
            }
            
            # Skip relationship_level if leaky column is excluded
            if not self.include_leaky_column and 'relationship_level' in categorical_defaults:
                del categorical_defaults['relationship_level']
            
            for col, default_val in categorical_defaults.items():
                if col in df.columns:
                    df[col].fillna(default_val, inplace=True)
            
            # Numeric columns: use median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    if 'ordinal' in col or 'numeric' in col:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(0, inplace=True)
            
            # Other categorical columns: use mode or 'unknown'
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col not in categorical_defaults and df[col].isnull().sum() > 0:
                    if len(df[col].mode()) > 0:
                        df[col].fillna(df[col].mode()[0], inplace=True)
                    else:
                        df[col].fillna('unknown', inplace=True)
        
        elif strategy == 'drop':
            df = df.dropna()
        
        else:
            # Simple imputation
            df = df.fillna(method=strategy)
        
        logger.info(f"Handled missing values with strategy: {strategy}")
        logger.info(f"Remaining missing values: {df.isnull().sum().sum()}")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, encoding_type: str = 'mixed') -> pd.DataFrame:
        """
        Encode categorical features dengan strategi yang tepat untuk masing-masing
        
        Args:
            df: DataFrame input
            encoding_type: 'mixed', 'label', 'onehot'
            
        Returns:
            DataFrame dengan encoded features
        """
        df = df.copy()
        
        if encoding_type == 'mixed':
            # Label encoding untuk high cardinality
            label_encode_cols = ['program_studi_clean', 'job_category']
            
            # One-hot encoding untuk low cardinality
            onehot_encode_cols = ['bidang_usaha_clean', 'organization_category', 
                                 'activity_type', 'ip_category']
            
            # Skip relationship features if leaky column excluded
            if self.include_leaky_column:
                onehot_encode_cols.append('relationship_level')
            
            # Label encoding
            from sklearn.preprocessing import LabelEncoder
            for col in label_encode_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    # Handle unknown categories
                    df[col] = df[col].fillna(DEFAULT_VALUES.get(col.replace('_clean', ''), 'Lainnya'))
                    df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            
            # One-hot encoding
            for col in onehot_encode_cols:
                if col in df.columns:
                    # Get dummies with prefix
                    dummies = pd.get_dummies(df[col], prefix=col)
                    df = pd.concat([df, dummies], axis=1)
        
        elif encoding_type == 'label':
            from sklearn.preprocessing import LabelEncoder
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            target_cols = ['Apakah pekerjaan yang Anda lakukan di tempat bekerja sesuai dengan bidang keilmuan?',
                          'Lulus_label', 'lulus', 'target', 'label', 'NIM']
            categorical_cols = [col for col in categorical_cols if col not in target_cols]
            
            for col in categorical_cols:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                
        elif encoding_type == 'onehot':
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            target_cols = ['Apakah pekerjaan yang Anda lakukan di tempat bekerja sesuai dengan bidang keilmuan?',
                          'Lulus_label', 'lulus', 'target', 'label', 'NIM']
            categorical_cols = [col for col in categorical_cols if col not in target_cols]
            
            df = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
        
        logger.info(f"Encoded categorical features using {encoding_type} encoding")
        
        return df
    
    def process_target_column(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
        """
        Process target column (pekerjaan sesuai bidang keilmuan)
        
        Args:
            df: DataFrame input
            
        Returns:
            Tuple of (DataFrame, target_column_name)
        """
        df = df.copy()
        
        # Find target column
        target_patterns = ['Apakah pekerjaan yang Anda lakukan di tempat bekerja sesuai dengan bidang keilmuan',
                          'pekerjaan_sesuai', 'sesuai_bidang']
        
        target_col = None
        for col in df.columns:
            for pattern in target_patterns:
                if pattern.lower() in col.lower():
                    target_col = col
                    break
            if target_col:
                break
        
        if target_col:
            # Clean and standardize target values
            df['target'] = df[target_col].apply(self.clean_text)
            
            # Map to binary
            def map_target(value):
                if not value:
                    return DEFAULT_VALUES['binary']
                
                if value in ['ya', 'yes', '1', 'true']:
                    return 'ya'
                elif value in ['tidak', 'no', '0', 'false']:
                    return 'tidak'
                else:
                    return DEFAULT_VALUES['binary']
            
            df['target'] = df['target'].apply(map_target)
            
            logger.info(f"Target distribution:\n{df['target'].value_counts()}")
            logger.info(f"Target percentage:\n{df['target'].value_counts(normalize=True) * 100}")
            
            return df, 'target'
        
        logger.warning("Target column not found!")
        return df, None
    
    def select_common_features(self, df: pd.DataFrame, common_columns: List[str]) -> pd.DataFrame:
        """
        Select only common features between datasets
        
        Args:
            df: DataFrame input
            common_columns: List of common column names
            
        Returns:
            DataFrame with only common columns
        """
        # Check which columns exist in the dataframe
        existing_cols = [col for col in common_columns if col in df.columns]
        missing_cols = [col for col in common_columns if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
        
        # Also include engineered features if they exist
        engineered_features = [
            'program_studi_clean', 'program_studi_clean_encoded',
            'bidang_usaha_clean', 'bidang_usaha_clean_encoded',
            'job_category', 'job_category_encoded',
            'organization_category', 'activity_type',
            'ip_category', 'ip_numeric',
            'ip_x_org_active', 'is_technical', 'target'
        ]
        
        # Add relationship features only if leaky column is included
        if self.include_leaky_column:
            engineered_features.extend(['relationship_level', 'relationship_level_ordinal',
                                      'technical_x_relationship'])
        
        # Add one-hot encoded columns
        for col in df.columns:
            if any(prefix in col for prefix in ['bidang_usaha_clean_', 'organization_category_', 
                                                'activity_type_', 'ip_category_']):
                engineered_features.append(col)
            # Add relationship one-hot only if included
            if self.include_leaky_column and 'relationship_level_' in col:
                engineered_features.append(col)
        
        # Add ordinal columns
        for col in df.columns:
            if '_numeric' in col and 'bermanfaat' in col.lower():
                engineered_features.append(col)
        
        # Combine original common columns with engineered features
        all_features = existing_cols + [f for f in engineered_features if f in df.columns]
        all_features = list(set(all_features))  # Remove duplicates
        
        df_common = df[all_features].copy()
        logger.info(f"Selected {len(all_features)} features (original + engineered)")
        
        return df_common
    
    def process_features(self, df: pd.DataFrame, 
                        common_columns: List[str] = None,
                        is_training: bool = True,
                        is_2017_data: bool = False,
                        filter_working: bool = True) -> pd.DataFrame:
        """
        Main function untuk process semua features
        
        Args:
            df: DataFrame input
            common_columns: List of common columns to use
            is_training: Whether this is training data
            is_2017_data: Whether this is 2017 data
            
        Returns:
            Processed DataFrame
        """
        logger.info(f"Starting feature processing for {'training' if is_training else 'test'} data")
        logger.info(f"Initial shape: {df.shape}")
        logger.info(f"Include leaky column: {self.include_leaky_column}")
        logger.info(f"Filter working alumni: {filter_working}")

        # Filter for working alumni if requested (BEFORE filtering target columns)
        if filter_working:
            df = self.filter_working_alumni(df)
            logger.info(f"Shape after filtering working alumni: {df.shape}")
        
        # Filter to target columns first
        df = self.filter_target_columns(df, is_2017_data=is_2017_data)
        
        # Process target column first (if training)
        if is_training:
            df, target_col = self.process_target_column(df)
        
        # Apply all transformations
        df = self.map_program_studi(df)
        df = self.map_relationship_level(df)  # Will skip if leaky column excluded
        df = self.map_bidang_usaha(df)
        df = self.map_pekerjaan_ke_kelompok(df)
        df = self.map_organization(df)
        df = self.map_ip_category(df)
        df = self.map_activity_type(df)
        df = self.handle_ordinal_features(df)
        df = self.create_interaction_features(df)
        
        # Handle missing values before encoding
        df = self.handle_missing_values(df, strategy='smart')
        
        # Encode categorical features
        df = self.encode_categorical_features(df, encoding_type='mixed')
        
        # Select common features if specified
        if common_columns:
            df = self.select_common_features(df, common_columns)
        
        # Final check for remaining missing values
        df = self.handle_missing_values(df, strategy='smart')
        
        logger.info(f"Final shape: {df.shape}")
        logger.info(f"Columns: {list(df.columns)}")
        
        return df


def load_column_info(filepath: str = None) -> Dict:
    """
    Load column information from saved JSON file
    
    Args:
        filepath: Path to column_analysis.json
        
    Returns:
        Dictionary with column information
    """
    if filepath is None:
        # Default path
        from pathlib import Path
        filepath = Path(__file__).parent.parent.parent / 'data' / '02_processed' / 'column_analysis.json'
    
    try:
        with open(filepath, 'r') as f:
            column_info = json.load(f)
        logger.info(f"Loaded column info from {filepath}")
        return column_info
    except FileNotFoundError:
        logger.warning(f"Column info file not found at {filepath}")
        return {}

# Example usage for training
def process_training_data(file_path: str, include_leaky: bool = False, filter_working: bool = True):
    """
    Process training data with feature engineering
    
    Args:
        file_path: Path to data file
        include_leaky: Whether to include leaky column
        filter_working: Whether to filter for working alumni only
        
    Returns:
        Processed DataFrame
    """
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(include_leaky_column=include_leaky)
    
    # Process features
    df_processed = engineer.process_features(
        df, 
        is_training=True,
        filter_working=filter_working
    )

    return df_processed


# Example usage for prediction
def process_prediction_data(file_path: str, include_leaky: bool = False, filter_working: bool = True):
    """
    Process data for prediction with feature engineering
    
    Args:
        file_path: Path to data file
        include_leaky: Whether to include leaky column
        filter_working: Whether to filter for working alumni only
        
    Returns:
        Processed DataFrame
    """
    # Load data
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    # Initialize feature engineer
    engineer = FeatureEngineer(include_leaky_column=include_leaky)
    
    # Process features
    df_processed = engineer.process_features(
        df, 
        is_training=False,
        filter_working=filter_working
    )
    
    return df_processed


def main():
    """
    Example usage of feature engineering with leakage control
    """
    print("Feature engineering module with leakage control loaded successfully!")
    print("\nKey Updates:")
    print("1. Added filter_target_columns() to select only 12 specified columns")
    print("2. Added include_leaky_column parameter to control data leakage")
    print("3. Updated all methods to respect leakage control settings")
    print("4. Swapped dataset roles: 2016 as training, 2017 as test")
    
    print("\nTarget columns that will be used:")
    for i, col in enumerate(TARGET_COLUMNS, 1):
        if col == LEAKY_COLUMN:
            print(f"{i}. {col} (POTENTIALLY LEAKY)")
        elif col == TARGET_VARIABLE:
            print(f"{i}. {col} (TARGET VARIABLE)")
        else:
            print(f"{i}. {col}")
    
    print("\nUsage example:")
    print("# Without leaky column:")
    print("fe_no_leak = FeatureEngineer(include_leaky_column=False)")
    print("df_processed = fe_no_leak.process_features(df, is_training=True)")
    print("\n# With leaky column:")
    print("fe_with_leak = FeatureEngineer(include_leaky_column=True)")
    print("df_processed = fe_with_leak.process_features(df, is_training=True)")


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("=" * 50)
    print("\nExample usage:")
    print("from build_features import process_training_data")
    print("df_train = process_training_data('data.xlsx', filter_working=True)")
    print("\nThis will filter the data to only include working alumni.")