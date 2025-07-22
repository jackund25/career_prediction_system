"""
Module untuk melakukan prediksi menggunakan model yang sudah dilatih
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from typing import Union, Dict, List, Tuple
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import paths and modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import MODELS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features.build_features import FeatureEngineer, load_column_info


class ModelPredictor:
    """Class for making predictions with trained models"""
    
    def __init__(self, model_path: str = None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to model file. If None, uses final_model.pkl
        """
        self.fe = FeatureEngineer()
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.metadata = None
        
        # Load model and related artifacts
        self._load_artifacts(model_path)
        
    def _load_artifacts(self, model_path: str = None):
        """Load model and related artifacts"""
        try:
            # Load model
            if model_path is None:
                model_path = MODELS_DIR / 'final_model.pkl'
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
            
            # Load preprocessor
            preprocessor_path = MODELS_DIR / 'preprocessor.pkl'
            self.preprocessor = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor")
            
            # Load label encoder if exists
            label_encoder_path = MODELS_DIR / 'label_encoder.pkl'
            if label_encoder_path.exists():
                self.label_encoder = joblib.load(label_encoder_path)
                logger.info("Loaded label encoder")
            
            # Load metadata
            metadata_path = MODELS_DIR / 'model_metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
                logger.info(f"Loaded metadata - Best model: {self.metadata.get('best_model')}")
            
        except FileNotFoundError as e:
            logger.error(f"Required file not found: {e}")
            raise
    
    def prepare_data(self, data: Union[pd.DataFrame, str], 
                    use_common_columns: bool = True,
                    source_year: str = 'auto') -> pd.DataFrame:
        """
        Prepare data for prediction with year-specific handling
        
        Args:
            data: DataFrame or path to data file
            use_common_columns: Whether to use only common columns
            source_year: '2016', '2024', or 'auto' to detect format
            
        Returns:
            Prepared DataFrame
        """
        # Load data if path is provided
        if isinstance(data, str):
            original_path = data
            data = self._load_data(data)
            # Auto-detect year from filename if possible
            if source_year == 'auto' and '2016' in original_path:
                source_year = '2016'
            elif source_year == 'auto' and '2024' in original_path:
                source_year = '2024'
        
        # Standardize data format if needed
        if source_year == '2016' or source_year == 'auto':
            try:
                from src.data.ensure_data_compatibility import DataCompatibilityChecker
                checker = DataCompatibilityChecker()
                data = checker.standardize_data(data, source_year)
                logger.info(f"Standardized data from {source_year} format")
            except ImportError:
                logger.warning("DataCompatibilityChecker not found, proceeding without standardization")
                # Manual handling for 2016 relationship column
                rel_cols = [col for col in data.columns if 'hubungan bidang studi' in col.lower()]
                if rel_cols:
                    col = rel_cols[0]
                    mapping = {
                        'item1': 'sangat erat',
                        'item2': 'erat',
                        'item3': 'cukup erat',
                        'item4': 'kurang erat',
                        'item5': 'tidak sama sekali'
                    }
                    data[col] = data[col].map(mapping).fillna(data[col])
        
        # Load column info if using common columns
        common_columns = None
        if use_common_columns:
            column_info = load_column_info()
            if column_info:
                common_columns = column_info.get('common_columns', None)
                logger.info(f"Using {len(common_columns)} common columns")
        
        # Process features
        df_processed = self.fe.process_features(data, common_columns=common_columns, is_training=False)
        
        # Remove target column if it exists
        target_candidates = ['Lulus_label', 'lulus', 'label', 'target']
        for col in target_candidates:
            if col in df_processed.columns:
                df_processed = df_processed.drop(columns=[col])
                logger.info(f"Removed target column: {col}")
        
        # Handle missing columns that preprocessor expects
        if hasattr(self.preprocessor, 'feature_names_in_'):
            expected_cols = self.preprocessor.feature_names_in_
            missing_cols = set(expected_cols) - set(df_processed.columns)
            
            if missing_cols:
                logger.warning(f"Missing {len(missing_cols)} columns expected by preprocessor")
                # Add missing columns with default values
                for col in missing_cols:
                    if '_numeric' in col:
                        # For numeric columns from ordinal features, use median value (3)
                        df_processed[col] = 3
                    elif any(cat in col for cat in ['_encoded', '_']):
                        # For encoded categorical columns, use 0
                        df_processed[col] = 0
                    else:
                        # For other columns, use appropriate default
                        df_processed[col] = 'unknown'
                
                logger.info(f"Added {len(missing_cols)} missing columns with default values")
        
        # Ensure column order matches what preprocessor expects
        if hasattr(self.preprocessor, 'feature_names_in_'):
            df_processed = df_processed[self.preprocessor.feature_names_in_]
        
        return df_processed
    
    def _load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"Loaded data from {filepath}: {df.shape}")
        return df
    
    def predict(self, data: Union[pd.DataFrame, str], 
               return_proba: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions
        
        Args:
            data: DataFrame or path to data file
            return_proba: Whether to return probabilities
            
        Returns:
            Predictions (and probabilities if requested)
        """
        # Prepare data
        X = self.prepare_data(data)
        
        # Transform using preprocessor
        X_transformed = self.preprocessor.transform(X)
        logger.info(f"Transformed data shape: {X_transformed.shape}")
        
        # Make predictions
        y_pred = self.model.predict(X_transformed)
        
        # Decode labels if label encoder exists
        if self.label_encoder:
            y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
        else:
            y_pred_decoded = y_pred
        
        if return_proba:
            y_proba = self.model.predict_proba(X_transformed)
            return y_pred_decoded, y_proba
        
        return y_pred_decoded
    
    def predict_and_save(self, data: Union[pd.DataFrame, str], 
                        output_path: str = None) -> pd.DataFrame:
        """
        Make predictions and save results
        
        Args:
            data: DataFrame or path to data file
            output_path: Path to save results. If None, saves to processed data dir
            
        Returns:
            DataFrame with predictions
        """
        # Load original data for reference
        if isinstance(data, str):
            df_original = self._load_data(data)
        else:
            df_original = data.copy()
        
        # Make predictions
        predictions, probabilities = self.predict(data, return_proba=True)
        
        # Create results dataframe
        results = df_original.copy()
        results['prediction'] = predictions
        
        # Add probabilities
        if probabilities.shape[1] == 2:
            # Binary classification
            results['probability_class_0'] = probabilities[:, 0]
            results['probability_class_1'] = probabilities[:, 1]
            results['confidence'] = np.max(probabilities, axis=1)
        else:
            # Multi-class
            for i in range(probabilities.shape[1]):
                results[f'probability_class_{i}'] = probabilities[:, i]
            results['confidence'] = np.max(probabilities, axis=1)
        
        # Save results
        if output_path is None:
            output_path = PROCESSED_DATA_DIR / 'predictions.csv'
        else:
            output_path = Path(output_path)
        
        results.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Print summary
        print("\n=== Prediction Summary ===")
        print(f"Total predictions: {len(predictions)}")
        print(f"\nPrediction distribution:")
        print(pd.Series(predictions).value_counts())
        print(f"\nAverage confidence: {results['confidence'].mean():.3f}")
        
        return results
    
    def evaluate_on_test_data(self, test_data: Union[pd.DataFrame, str]) -> Dict:
        """
        Evaluate model on test data with known labels
        
        Args:
            test_data: DataFrame or path to test data with labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
        
        # Load data
        if isinstance(test_data, str):
            df_test = self._load_data(test_data)
        else:
            df_test = test_data.copy()
        
        # Find target column
        target_col = None
        target_candidates = ['Lulus_label', 'lulus', 'label', 'target']
        for col in target_candidates:
            if col in df_test.columns:
                target_col = col
                break
        
        if not target_col:
            raise ValueError("Target column not found in test data")
        
        # Separate features and target
        y_true = df_test[target_col]
        
        # Encode target if necessary
        if y_true.dtype == 'object' and self.label_encoder:
            y_true_encoded = self.label_encoder.transform(y_true)
        else:
            y_true_encoded = y_true
        
        # Make predictions
        y_pred, y_proba = self.predict(df_test, return_proba=True)
        
        # Encode predictions for comparison if necessary
        if y_pred.dtype == 'object' and self.label_encoder:
            y_pred_encoded = self.label_encoder.transform(y_pred)
        else:
            y_pred_encoded = y_pred
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_true_encoded, y_pred_encoded),
            'f1_score': f1_score(y_true_encoded, y_pred_encoded, average='weighted'),
            'roc_auc': roc_auc_score(y_true_encoded, y_proba[:, 1]) if y_proba.shape[1] == 2 else None
        }
        
        # Print evaluation results
        print("\n=== Model Evaluation Results ===")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        return metrics


def make_prediction(dataframe: pd.DataFrame, 
                   model_path: str = None,
                   use_common_columns: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for making predictions
    
    Args:
        dataframe: Input dataframe
        model_path: Path to model file
        use_common_columns: Whether to use only common columns
        
    Returns:
        Tuple of (predictions, probabilities)
    """
    predictor = ModelPredictor(model_path)
    predictions, probabilities = predictor.predict(dataframe, return_proba=True)
    return predictions, probabilities


def predict_2016_data():
    """Specific function to predict on 2016 data"""
    logger.info("Predicting on 2016 data...")
    
    # Initialize predictor
    predictor = ModelPredictor()
    
    # Load 2016 data
    data_2016_path = RAW_DATA_DIR / 'data 2016 - daffari_raw.csv'
    
    # Make predictions and save
    results = predictor.predict_and_save(
        str(data_2016_path),
        output_path=PROCESSED_DATA_DIR / 'predictions_2016.csv'
    )
    
    # If 2016 data has labels, evaluate
    try:
        metrics = predictor.evaluate_on_test_data(str(data_2016_path))
        
        # Save evaluation metrics
        with open(PROCESSED_DATA_DIR / 'evaluation_2016.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Evaluation completed and saved")
    except ValueError as e:
        logger.warning(f"Could not evaluate: {e}")
    
    return results


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--input', type=str, help='Path to input data file')
    parser.add_argument('--output', type=str, help='Path to save predictions')
    parser.add_argument('--model', type=str, help='Path to model file (default: final_model.pkl)')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate if test data has labels')
    parser.add_argument('--predict-2016', action='store_true', help='Predict on 2016 data')
    
    args = parser.parse_args()
    
    if args.predict_2016:
        # Predict on 2016 data
        results = predict_2016_data()
        print(f"\nPredictions saved for {len(results)} samples")
        
    elif args.input:
        # Initialize predictor
        predictor = ModelPredictor(args.model)
        
        if args.evaluate:
            # Evaluate on test data
            metrics = predictor.evaluate_on_test_data(args.input)
            print("\nEvaluation completed")
        else:
            # Make predictions
            results = predictor.predict_and_save(args.input, args.output)
            print(f"\nPredictions saved for {len(results)} samples")
    else:
        print("Usage examples:")
        print("  python predict_model.py --predict-2016")
        print("  python predict_model.py --input data.csv --output predictions.csv")
        print("  python predict_model.py --input test_data.csv --evaluate")


if __name__ == "__main__":
    main()