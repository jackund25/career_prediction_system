"""
Module untuk memastikan kompatibilitas data antara 2016 dan 2024
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import paths
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import RAW_DATA_DIR, PROCESSED_DATA_DIR


class DataCompatibilityChecker:
    """Class untuk mengecek dan memastikan kompatibilitas data"""
    
    def __init__(self):
        """Initialize compatibility checker"""
        # Define mappings for different years
        self.mappings_2016_to_2024 = {
            'relationship': {
                'item1': 'sangat erat',
                'item2': 'erat', 
                'item3': 'cukup erat',
                'item4': 'kurang erat',
                'item5': 'tidak sama sekali'
            },
            # Add other mappings if needed
        }
        
    def check_data_format(self, df: pd.DataFrame, year: str = 'unknown') -> Dict:
        """
        Check data format and identify which year's format it follows
        
        Args:
            df: DataFrame to check
            year: Expected year ('2016' or '2024')
            
        Returns:
            Dictionary with format information
        """
        format_info = {
            'year': year,
            'issues': [],
            'format_differences': {}
        }
        
        # Check relationship column
        relationship_cols = [col for col in df.columns if 'hubungan bidang studi' in col.lower()]
        if relationship_cols:
            rel_col = relationship_cols[0]
            unique_values = df[rel_col].dropna().unique()
            
            # Check if it has item format (2016)
            has_items = any('item' in str(val).lower() for val in unique_values)
            # Check if it has text format (2024)
            has_text = any(any(cat in str(val).lower() for cat in ['erat', 'kurang', 'cukup']) 
                          for val in unique_values)
            
            if has_items:
                format_info['format_differences']['relationship'] = '2016_format'
                logger.info(f"Detected 2016 format for relationship column: {unique_values[:5]}")
            elif has_text:
                format_info['format_differences']['relationship'] = '2024_format'
                logger.info(f"Detected 2024 format for relationship column: {unique_values[:5]}")
            else:
                format_info['issues'].append(f"Unknown format for relationship column: {unique_values[:5]}")
        
        # Check other potential format differences
        # Add more checks as needed
        
        return format_info
    
    def standardize_data(self, df: pd.DataFrame, source_year: str = 'auto') -> pd.DataFrame:
        """
        Standardize data to 2024 format
        
        Args:
            df: DataFrame to standardize
            source_year: '2016', '2024', or 'auto' to detect
            
        Returns:
            Standardized DataFrame
        """
        df = df.copy()
        
        # Auto-detect format if needed
        if source_year == 'auto':
            format_info = self.check_data_format(df)
            detected_formats = format_info.get('format_differences', {})
        else:
            detected_formats = {'relationship': f'{source_year}_format'}
        
        # Standardize relationship column
        relationship_cols = [col for col in df.columns if 'hubungan bidang studi' in col.lower()]
        if relationship_cols and detected_formats.get('relationship') == '2016_format':
            rel_col = relationship_cols[0]
            logger.info(f"Standardizing relationship column from 2016 to 2024 format")
            
            # Apply mapping
            df[rel_col] = df[rel_col].map(self.mappings_2016_to_2024['relationship']).fillna(df[rel_col])
            
            # Log changes
            logger.info(f"After standardization: {df[rel_col].value_counts()}")
        
        return df
    
    def compare_datasets(self, df_2016: pd.DataFrame, df_2024: pd.DataFrame) -> Dict:
        """
        Compare two datasets and identify all differences
        
        Args:
            df_2016: 2016 dataset
            df_2024: 2024 dataset
            
        Returns:
            Dictionary with comparison results
        """
        comparison = {
            'common_columns': [],
            'unique_2016': [],
            'unique_2024': [],
            'format_differences': {},
            'value_differences': {}
        }
        
        # Column comparison
        cols_2016 = set(df_2016.columns)
        cols_2024 = set(df_2024.columns)
        
        comparison['common_columns'] = list(cols_2016.intersection(cols_2024))
        comparison['unique_2016'] = list(cols_2016 - cols_2024)
        comparison['unique_2024'] = list(cols_2024 - cols_2016)
        
        # For common columns, check value differences
        for col in comparison['common_columns']:
            values_2016 = set(df_2016[col].dropna().unique())
            values_2024 = set(df_2024[col].dropna().unique())
            
            if len(values_2016) < 20 and len(values_2024) < 20:  # Only for categorical
                if values_2016 != values_2024:
                    comparison['value_differences'][col] = {
                        'only_2016': list(values_2016 - values_2024)[:10],
                        'only_2024': list(values_2024 - values_2016)[:10],
                        'sample_2016': list(values_2016)[:5],
                        'sample_2024': list(values_2024)[:5]
                    }
        
        return comparison
    
    def create_compatibility_report(self, save_path: str = None) -> Dict:
        """
        Create a comprehensive compatibility report
        
        Args:
            save_path: Path to save report
            
        Returns:
            Compatibility report dictionary
        """
        # Load both datasets
        df_2016 = pd.read_csv(RAW_DATA_DIR / 'data 2016 - daffari_raw.csv')
        df_2024 = pd.read_excel(RAW_DATA_DIR / 'DATA TS SARJANA 2024.xlsx')
        
        logger.info(f"Loaded 2016 data: {df_2016.shape}")
        logger.info(f"Loaded 2024 data: {df_2024.shape}")
        
        # Check formats
        format_2016 = self.check_data_format(df_2016, '2016')
        format_2024 = self.check_data_format(df_2024, '2024')
        
        # Compare datasets
        comparison = self.compare_datasets(df_2016, df_2024)
        
        # Create report
        report = {
            'data_shapes': {
                '2016': df_2016.shape,
                '2024': df_2024.shape
            },
            'format_checks': {
                '2016': format_2016,
                '2024': format_2024
            },
            'comparison': comparison,
            'recommendations': []
        }
        
        # Add recommendations
        if comparison['value_differences']:
            report['recommendations'].append(
                f"Found {len(comparison['value_differences'])} columns with different values between datasets"
            )
        
        if format_2016['format_differences'] or format_2024['format_differences']:
            report['recommendations'].append(
                "Use standardize_data() function to convert 2016 format to 2024 format"
            )
        
        # Save report if path provided
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Compatibility report saved to: {save_path}")
        
        return report


def ensure_compatibility():
    """
    Main function to ensure data compatibility
    """
    checker = DataCompatibilityChecker()
    
    # Create compatibility report
    report_path = PROCESSED_DATA_DIR / 'data_compatibility_report.json'
    report = checker.create_compatibility_report(str(report_path))
    
    print("\n=== Data Compatibility Report ===")
    print(f"2016 shape: {report['data_shapes']['2016']}")
    print(f"2024 shape: {report['data_shapes']['2024']}")
    print(f"Common columns: {len(report['comparison']['common_columns'])}")
    
    if report['comparison']['value_differences']:
        print(f"\nColumns with value differences: {len(report['comparison']['value_differences'])}")
        for col, diff in list(report['comparison']['value_differences'].items())[:5]:
            print(f"\n{col}:")
            print(f"  2016 sample: {diff['sample_2016']}")
            print(f"  2024 sample: {diff['sample_2024']}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"- {rec}")
    
    return report


if __name__ == "__main__":
    ensure_compatibility()