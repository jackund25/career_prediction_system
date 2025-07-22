import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

class DataLeakageChecker:
    """Check for potential data leakage in tracer study datasets"""
    
    def __init__(self):
        self.leakage_keywords = [
            # Direct alignment keywords
            'sesuai', 'cocok', 'relevan', 'terkait', 'berhubungan',
            'selaras', 'match', 'fit', 'align', 'related',
            'sama', 'mirip', 'dekat', 'erat', 'hubungan',
            # Benefit/usefulness keywords
            'manfaat', 'bermanfaat', 'berguna', 'membantu', 'benefit',
            'useful', 'helpful', 'advantage'
        ]
        
    def check_column_names_for_leakage(self, df, dataset_name):
        """Check column names for potential leakage indicators"""
        print(f"\n{'='*80}")
        print(f"CHECKING DATASET: {dataset_name}")
        print(f"{'='*80}")
        print(f"Total columns: {len(df.columns)}")
        print(f"Total rows: {len(df)}")
        
        suspicious_cols = {
            'critical': [],
            'high': [],
            'moderate': [],
            'low': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            risk_score = 0
            risk_factors = []
            
            # Check for direct alignment questions
            if any(word in col_lower for word in ['sesuai', 'erat', 'hubungan', 'cocok', 'relevan']):
                if 'pekerjaan' in col_lower and ('bidang' in col_lower or 'program' in col_lower or 'studi' in col_lower):
                    risk_score += 90
                    risk_factors.append("Directly asks about job-study alignment")
            
            # Check for benefit/usefulness questions
            if any(word in col_lower for word in ['manfaat', 'bermanfaat', 'berguna']):
                if 'pekerjaan' in col_lower or 'kerja' in col_lower or 'karir' in col_lower:
                    risk_score += 70
                    risk_factors.append("Asks about study benefits to work")
            
            # Check for job-related information
            if 'pekerjaan' in col_lower or 'kerja' in col_lower or 'job' in col_lower:
                if risk_score == 0:  # Only if not already flagged
                    risk_score += 30
                    risk_factors.append("Contains job information")
            
            # Check for skill/competency alignment
            if any(word in col_lower for word in ['kompetensi', 'skill', 'kemampuan']):
                if 'pekerjaan' in col_lower or 'kerja' in col_lower:
                    risk_score += 40
                    risk_factors.append("Links skills to work")
            
            # Categorize by risk level
            if risk_score >= 80:
                suspicious_cols['critical'].append({
                    'column': col,
                    'score': risk_score,
                    'factors': risk_factors
                })
            elif risk_score >= 60:
                suspicious_cols['high'].append({
                    'column': col,
                    'score': risk_score,
                    'factors': risk_factors
                })
            elif risk_score >= 30:
                suspicious_cols['moderate'].append({
                    'column': col,
                    'score': risk_score,
                    'factors': risk_factors
                })
            elif risk_score > 0:
                suspicious_cols['low'].append({
                    'column': col,
                    'score': risk_score,
                    'factors': risk_factors
                })
        
        return suspicious_cols
    
    def analyze_column_content(self, df, column_name, sample_size=10):
        """Analyze actual content of a column"""
        print(f"\n--- Analyzing: {column_name} ---")
        
        # Basic statistics
        print(f"Non-null values: {df[column_name].notna().sum()} ({df[column_name].notna().sum()/len(df)*100:.1f}%)")
        print(f"Unique values: {df[column_name].nunique()}")
        
        # Value distribution
        value_counts = df[column_name].value_counts()
        print(f"\nTop {min(10, len(value_counts))} values:")
        for idx, (val, count) in enumerate(value_counts.head(10).items()):
            print(f"  {idx+1}. '{val}': {count} ({count/len(df)*100:.1f}%)")
        
        # For text columns, check for leakage keywords
        if df[column_name].dtype == 'object':
            text_values = df[column_name].dropna().astype(str)
            if len(text_values) > 0:
                keyword_found = False
                for keyword in self.leakage_keywords:
                    count = text_values.str.lower().str.contains(keyword).sum()
                    if count > 0:
                        if not keyword_found:
                            print("\nLeakage keywords found in content:")
                            keyword_found = True
                        print(f"  - '{keyword}': {count} occurrences ({count/len(text_values)*100:.1f}%)")
        
        # Show random samples
        non_null_values = df[column_name].dropna()
        if len(non_null_values) > 0:
            print(f"\nRandom samples:")
            samples = non_null_values.sample(min(sample_size, len(non_null_values)))
            for idx, val in enumerate(samples, 1):
                print(f"  {idx}. {str(val)[:100]}{'...' if len(str(val)) > 100 else ''}")
    
    def calculate_correlation_matrix(self, df, suspicious_columns):
        """Calculate correlations between suspicious columns"""
        print("\n=== CORRELATION ANALYSIS ===")
        
        # Filter to only numeric or encodable columns
        correlation_data = {}
        
        for col in suspicious_columns:
            if col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    correlation_data[col] = df[col]
                else:
                    # Try to encode categorical
                    try:
                        le = LabelEncoder()
                        encoded = le.fit_transform(df[col].fillna('MISSING'))
                        correlation_data[col] = encoded
                    except:
                        pass
        
        if len(correlation_data) > 1:
            corr_df = pd.DataFrame(correlation_data)
            corr_matrix = corr_df.corr()
            
            print("\nHigh correlations (>0.5):")
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        col1 = corr_matrix.columns[i]
                        col2 = corr_matrix.columns[j]
                        print(f"  {col1[:40]}... <-> {col2[:40]}...: {corr_val:.3f}")
    
    def generate_report(self, df, dataset_name):
        """Generate comprehensive leakage report"""
        print(f"\n{'='*80}")
        print(f"DATA LEAKAGE REPORT: {dataset_name}")
        print(f"{'='*80}")
        
        # Check column names
        suspicious = self.check_column_names_for_leakage(df, dataset_name)
        
        # Report findings
        if suspicious['critical']:
            print(f"\n🚨 CRITICAL RISK ({len(suspicious['critical'])} columns):")
            for item in suspicious['critical']:
                print(f"\n  Column: {item['column']}")
                print(f"  Score: {item['score']}/100")
                print(f"  Reasons: {', '.join(item['factors'])}")
                # Analyze content
                self.analyze_column_content(df, item['column'])
        
        if suspicious['high']:
            print(f"\n⚠️  HIGH RISK ({len(suspicious['high'])} columns):")
            for item in suspicious['high']:
                print(f"\n  Column: {item['column']}")
                print(f"  Score: {item['score']}/100")
                print(f"  Reasons: {', '.join(item['factors'])}")
                # Analyze content for first 2
                if suspicious['high'].index(item) < 2:
                    self.analyze_column_content(df, item['column'])
        
        if suspicious['moderate']:
            print(f"\n⚡ MODERATE RISK ({len(suspicious['moderate'])} columns):")
            for item in suspicious['moderate']:
                print(f"\n  Column: {item['column'][:80]}...")
                print(f"  Score: {item['score']}/100")
                print(f"  Reasons: {', '.join(item['factors'])}")
        
        # Check correlations
        all_suspicious_cols = []
        for risk_level in ['critical', 'high']:
            all_suspicious_cols.extend([item['column'] for item in suspicious[risk_level]])
        
        if len(all_suspicious_cols) > 1:
            self.calculate_correlation_matrix(df, all_suspicious_cols)
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        total_suspicious = sum(len(suspicious[level]) for level in suspicious)
        print(f"Total columns: {len(df.columns)}")
        print(f"Suspicious columns: {total_suspicious}")
        print(f"  - Critical: {len(suspicious['critical'])}")
        print(f"  - High: {len(suspicious['high'])}")
        print(f"  - Moderate: {len(suspicious['moderate'])}")
        print(f"  - Low: {len(suspicious['low'])}")
        print(f"Clean columns: {len(df.columns) - total_suspicious}")
        
        return suspicious


# Main execution
if __name__ == "__main__":
    checker = DataLeakageChecker()
    
    # Check Dataset 2016
    print("\n" + "="*80)
    print("ANALYZING DATASET 2016")
    print("="*80)
    
    try:
        # Try different possible filenames
        try:
            df_2016 = pd.read_csv('data_2016_daffari_raw.csv')
        except:
            try:
                df_2016 = pd.read_csv('data 2016  daffari_raw.csv')
            except:
                df_2016 = pd.read_csv('data/01_raw/data 2016 - daffari_raw.csv')
        
        suspicious_2016 = checker.generate_report(df_2016, "Dataset 2016")
        
    except Exception as e:
        print(f"Error loading 2016 dataset: {e}")
        print("Please check the file path and try again.")
    
    # Check Dataset 2017
    print("\n\n" + "="*80)
    print("ANALYZING DATASET 2017 (Excel)")
    print("="*80)
    
    try:
        # Try to read Excel file
        try:
            df_2017 = pd.read_excel('DATA TS SARJANA 2024.xlsx')
        except:
            try:
                df_2017 = pd.read_excel('data/01_raw/DATA TS SARJANA 2024.xlsx')
            except:
                # Try with different engines
                df_2017 = pd.read_excel('DATA TS SARJANA 2024.xlsx', engine='openpyxl')
        
        suspicious_2017 = checker.generate_report(df_2017, "Dataset 2017")
        
    except Exception as e:
        print(f"Error loading 2017 dataset: {e}")
        print("Please check the file path and try again.")
        print("\nNote: Make sure you have openpyxl installed: pip install openpyxl")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Remove all CRITICAL risk columns before modeling")
    print("2. Carefully evaluate HIGH risk columns")
    print("3. Consider transforming MODERATE risk columns")
    print("4. Use strong regularization in your models")
    print("5. Implement proper train/test split without leakage")