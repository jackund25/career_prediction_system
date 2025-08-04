"""
Prediction module for ensemble models
Clean implementation for predicting alumni career alignment
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional
import argparse
import sys
from pathlib import Path
# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Then import features
from features.build_features import TARGET_VARIABLE, TARGET_COLUMNS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CareerPredictor:
    """Main class for career alignment prediction"""
    
    def __init__(self, model_scenario: str = 'without_leaky'):
        """
        Initialize predictor with specified model scenario
        
        Args:
            model_scenario: 'without_leaky' or 'with_leaky'
        """
        self.model_scenario = model_scenario
        self.models = {}
        self.preprocessing = {}
        
        # Set paths
        project_root = Path(__file__).parent.parent.parent
        self.model_dir = project_root / 'models' / 'advanced' / model_scenario
        self.results_dir = project_root / 'results' / 'predictions'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load models and preprocessing
        self._load_models()
        
        logger.info(f"CareerPredictor initialized with scenario: {model_scenario}")

    def _load_models(self):
        """Load all required models and preprocessing objects"""
        print(f"Looking for models in: {self.model_dir}")
        print(f"Directory exists: {self.model_dir.exists()}")
        if self.model_dir.exists():
            print(f"Files found: {list(self.model_dir.glob('*.pkl'))}")
        try:
            # Load ensemble model (primary)
            self.models['ensemble'] = joblib.load(self.model_dir / 'ensemble.pkl')
            
            # Load individual models for comparison
            self.models['xgboost'] = joblib.load(self.model_dir / 'xgboost.pkl')
            self.models['catboost'] = joblib.load(self.model_dir / 'catboost.pkl')
            
            # Load preprocessing
            self.preprocessing['feature_engineer'] = joblib.load(self.model_dir / 'feature_engineer.pkl')
            self.preprocessing['scaler'] = joblib.load(self.model_dir / 'scaler.pkl')
            self.preprocessing['label_encoder'] = joblib.load(self.model_dir / 'label_encoder.pkl')
            
            # Load feature info
            feature_info_path = self.model_dir / 'feature_info.json'
            if feature_info_path.exists():
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_names = feature_info.get('feature_names', [])
            else:
                self.feature_names = []
                logger.warning("Feature names not found")
            
            logger.info(f"Successfully loaded {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def prepare_data(self, df: pd.DataFrame, is_2017_data: bool = True) -> Tuple[np.ndarray, pd.DataFrame, Optional[pd.Series], pd.DataFrame]:
        """
        Prepare data for prediction
        
        Args:
            df: Input DataFrame
            is_2017_data: Whether this is 2017 format data
            
        Returns:
            Tuple of (X_scaled, df_processed, original_target, df_input_with_target)
        """
        
        # Save original dataframe with target before any processing
        df_input_with_target = df.copy() if TARGET_VARIABLE in df.columns else pd.DataFrame()
        
        # Process features FIRST (this includes filtering)
        df_processed = self.preprocessing['feature_engineer'].process_features(
            df,
            common_columns=TARGET_COLUMNS,
            is_training=False,
            is_2017_data=is_2017_data
        )
        
        # Check for target column AFTER processing/filtering
        original_target = None
        if 'target' in df_processed.columns:
            original_target = df_processed['target'].copy()
            logger.info(f"Found target column after processing. Distribution:\n{original_target.value_counts(dropna=False)}")
        
        # Get numeric features
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['target', 'NIM']]
        
        # Align with training features
        if self.feature_names:
            missing_features = set(self.feature_names) - set(feature_cols)
            if missing_features:
                logger.warning(f"Adding {len(missing_features)} missing features with zeros")
                for feat in missing_features:
                    df_processed[feat] = 0
            
            X = df_processed[self.feature_names]
        else:
            X = df_processed[feature_cols]
        
        # Scale features
        X_scaled = self.preprocessing['scaler'].transform(X.values)
        
        logger.info(f"Prepared {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features")
        
        return X_scaled, df_processed, original_target, df_input_with_target

    def predict(self, df: pd.DataFrame, is_2017_data: bool = True) -> pd.DataFrame:
        """
        Make predictions on input data
        
        Args:
            df: Input DataFrame
            is_2017_data: Whether this is 2017 format data
            
        Returns:
            DataFrame with predictions and metadata
        """
        # Prepare data
        X_scaled, df_processed, original_target, df_input_with_target = self.prepare_data(df, is_2017_data)
        
        # Make predictions
        predictions = pd.DataFrame()
        
        # Add NIM if available
        if 'NIM' in df_processed.columns:
            predictions['NIM'] = df_processed['NIM'].values
        
        # Ensemble predictions (primary)
        ensemble_pred = self.models['ensemble'].predict(X_scaled)
        ensemble_proba = self.models['ensemble'].predict_proba(X_scaled)
        
        # Get probabilities for both classes
        predictions['prediction'] = self.preprocessing['label_encoder'].inverse_transform(ensemble_pred)
        predictions['probability_tidak'] = ensemble_proba[:, 0]  # Probability for class "tidak"
        predictions['probability_ya'] = ensemble_proba[:, 1]     # Probability for class "ya"
        predictions['confidence'] = np.maximum(ensemble_proba[:, 0], ensemble_proba[:, 1])
        
        # Individual model predictions for comparison
        xgb_pred = self.models['xgboost'].predict(X_scaled)
        cat_pred = self.models['catboost'].predict(X_scaled)
        
        predictions['xgboost_agrees'] = (xgb_pred == ensemble_pred).astype(int)
        predictions['catboost_agrees'] = (cat_pred == ensemble_pred).astype(int)
        predictions['model_agreement'] = predictions[['xgboost_agrees', 'catboost_agrees']].sum(axis=1) + 1  # +1 for ensemble itself
        
        # Add metadata
        predictions['model_scenario'] = self.model_scenario
        predictions['prediction_date'] = datetime.now().strftime('%m/%d/%Y %H:%M')
        
        # Add original target if available (for evaluation) - this is the filtered target
        if original_target is not None:
            predictions['actual_target'] = original_target.values
            
            # Calculate metrics for valid targets only
            mask_valid = predictions['actual_target'].notna()
            if mask_valid.sum() > 0:
                valid_actual = predictions.loc[mask_valid, 'actual_target']
                valid_pred = predictions.loc[mask_valid, 'prediction']
                
                accuracy = (valid_actual == valid_pred).mean()
                logger.info(f"Accuracy on {mask_valid.sum()} valid samples: {accuracy:.4f}")
        
        # Add original target column from input data using NIM mapping
        if not df_input_with_target.empty and 'NIM' in predictions.columns and 'NIM' in df_input_with_target.columns:
            # Create mapping from NIM to original target
            nim_to_target = df_input_with_target.set_index('NIM')[TARGET_VARIABLE].to_dict()
            
            # Map target values
            predictions['target'] = predictions['NIM'].map(nim_to_target)
            
            logger.info(f"Added original target column for {predictions['target'].notna().sum()} samples")
        
        # Summary statistics
        pred_dist = predictions['prediction'].value_counts()
        logger.info(f"Prediction distribution:\n{pred_dist}")
        logger.info(f"Average confidence: {predictions['confidence'].mean():.4f}")
        
        return predictions
    
    def evaluate_predictions(self, predictions: pd.DataFrame) -> Dict:
        """
        Evaluate predictions if actual target is available
        
        Args:
            predictions: DataFrame with predictions and actual_target column
            
        Returns:
            Dictionary with evaluation metrics
        """
        if 'actual_target' not in predictions.columns:
            logger.warning("No actual target found for evaluation")
            return {}
        
        # Filter valid targets
        mask_valid = predictions['actual_target'].notna()
        valid_data = predictions[mask_valid].copy()
        
        if len(valid_data) == 0:
            logger.warning("No valid targets for evaluation")
            return {}
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Encode for metrics
        y_true = self.preprocessing['label_encoder'].transform(valid_data['actual_target'])
        y_pred = self.preprocessing['label_encoder'].transform(valid_data['prediction'])
        
        metrics = {
            'n_evaluated': int(len(valid_data)),
            'n_total': int(len(predictions)),
            'n_missing': int((~mask_valid).sum()),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, average='binary')),
            'recall': float(recall_score(y_true, y_pred, average='binary')),
            'f1_score': float(f1_score(y_true, y_pred, average='binary')),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add class-wise metrics
        cm = confusion_matrix(y_true, y_pred)
        metrics['true_negatives'] = int(cm[0, 0])
        metrics['false_positives'] = int(cm[0, 1])
        metrics['false_negatives'] = int(cm[1, 0])
        metrics['true_positives'] = int(cm[1, 1])
        
        logger.info(f"Evaluation complete: F1={metrics['f1_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
        
        return metrics
    
    def save_predictions(self, predictions: pd.DataFrame, filename: Optional[str] = None):
        """Save predictions to file and always create JSON metrics"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'predictions_{self.model_scenario}_{timestamp}.csv'
        
        filepath = self.results_dir / filename
        predictions.to_csv(filepath, index=False)
        logger.info(f"Predictions saved to {filepath}")
        
        # Always create JSON file with prediction summary
        metrics = {}
        
        # Check if we have actual target for evaluation
        if 'actual_target' in predictions.columns:
            # Full evaluation metrics
            metrics = self.evaluate_predictions(predictions)
        else:
            # Basic prediction summary when no target available
            metrics = {
                'n_total': int(len(predictions)),
                'n_evaluated': 0,
                'n_missing': int(len(predictions)),
                'model_scenario': self.model_scenario,
                'prediction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'predictions_distribution': predictions['prediction'].value_counts().to_dict(),
                'average_confidence': float(predictions['confidence'].mean()),
                'confidence_stats': {
                    'mean': float(predictions['confidence'].mean()),
                    'std': float(predictions['confidence'].std()),
                    'min': float(predictions['confidence'].min()),
                    'max': float(predictions['confidence'].max()),
                    'median': float(predictions['confidence'].median())
                },
                'model_agreement_stats': {
                    'all_models_agree': int((predictions['model_agreement'] == 3).sum()),
                    'two_models_agree': int((predictions['model_agreement'] == 2).sum()),
                    'only_ensemble': int((predictions['model_agreement'] == 1).sum()),
                    'agreement_percentage': float((predictions['model_agreement'] == 3).sum() / len(predictions) * 100)
                }
            }
            
            # Add probability distribution stats
            if 'probability_ya' in predictions.columns:
                metrics['probability_ya_stats'] = {
                    'mean': float(predictions['probability_ya'].mean()),
                    'std': float(predictions['probability_ya'].std()),
                    'min': float(predictions['probability_ya'].min()),
                    'max': float(predictions['probability_ya'].max()),
                    'median': float(predictions['probability_ya'].median())
                }
            
            logger.info("No actual target found - creating summary metrics only")
        
        # Always save JSON file
        metrics_file = filepath.with_suffix('.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_file}")
        
        # Print summary
        print(f"\nMetrics saved to: {metrics_file}")
        if 'accuracy' in metrics:
            print(f"Evaluation metrics available (accuracy: {metrics['accuracy']:.2%})")
        else:
            print("Prediction summary saved (no target data for evaluation)")

def predict_alumni_batch(data_path: str, model_scenario: str = 'without_leaky', 
                        output_file: Optional[str] = None):
    """
    Convenience function to predict on a batch of alumni data
    
    Args:
        data_path: Path to data file (CSV or Excel)
        model_scenario: Which model to use ('without_leaky' or 'with_leaky')
        output_file: Optional output filename
    """
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith(('.xlsx', '.xls')):
        df = pd.read_excel(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Loaded data: {df.shape}")
    
    # Initialize predictor
    predictor = CareerPredictor(model_scenario=model_scenario)
    
    # Make predictions
    predictions = predictor.predict(df)
    
    # Save results
    predictor.save_predictions(predictions, output_file)
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Model scenario: {model_scenario}")
    print(f"Total samples: {len(predictions)}")
    print(f"\nPrediction distribution:")
    print(predictions['prediction'].value_counts())
    print(f"\nAverage confidence: {predictions['confidence'].mean():.2%}")
    
    if 'actual_target' in predictions.columns:
        metrics = predictor.evaluate_predictions(predictions)
        if metrics:
            print(f"\nEvaluation metrics (on {metrics['n_evaluated']} valid samples):")
            print(f"  Accuracy: {metrics['accuracy']:.2%}")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
    
    return predictions


def compare_model_scenarios(data_path: str):
    """
    Compare predictions between with/without leaky column scenarios
    
    Args:
        data_path: Path to data file
    """
    print("\n" + "="*80)
    print("COMPARING MODEL SCENARIOS")
    print("="*80)
    
    results = {}
    
    # Run predictions for both scenarios
    for scenario in ['without_leaky', 'with_leaky']:
        print(f"\n--- Scenario: {scenario} ---")
        predictor = CareerPredictor(model_scenario=scenario)
        
        # Load data fresh for each scenario
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            df = pd.read_excel(data_path)
        
        predictions = predictor.predict(df)
        results[scenario] = predictions
        
        # Save scenario results
        predictor.save_predictions(predictions, f'comparison_{scenario}.csv')
    
    # Compare predictions
    comparison = pd.DataFrame({
        'NIM': results['without_leaky']['NIM'] if 'NIM' in results['without_leaky'].columns else range(len(results['without_leaky'])),
        'pred_without_leaky': results['without_leaky']['prediction'],
        'prob_without_leaky': results['without_leaky']['probability'],
        'pred_with_leaky': results['with_leaky']['prediction'],
        'prob_with_leaky': results['with_leaky']['probability'],
        'same_prediction': results['without_leaky']['prediction'] == results['with_leaky']['prediction']
    })
    
    # Add actual target if available
    if 'actual_target' in results['without_leaky'].columns:
        comparison['actual_target'] = results['without_leaky']['actual_target']
    
    # Calculate agreement
    agreement_rate = comparison['same_prediction'].mean()
    
    print(f"\n--- Comparison Results ---")
    print(f"Agreement rate: {agreement_rate:.2%}")
    print(f"Disagreements: {(~comparison['same_prediction']).sum()} samples")
    
    # Save comparison
    project_root = Path(__file__).parent.parent.parent
    comparison_path = project_root / 'results' / 'predictions' / 'scenario_comparison.csv'
    comparison.to_csv(comparison_path, index=False)
    
    # Show some disagreement cases
    if (~comparison['same_prediction']).sum() > 0:
        print("\nSample disagreement cases:")
        disagreements = comparison[~comparison['same_prediction']].head(10)
        print(disagreements[['pred_without_leaky', 'pred_with_leaky', 'prob_without_leaky', 'prob_with_leaky']])
    
    return comparison


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description='Predict alumni career alignment')
    parser.add_argument('data_path', type=str, help='Path to alumni data file')
    parser.add_argument('--scenario', type=str, default='without_leaky',
                       choices=['without_leaky', 'with_leaky'],
                       help='Model scenario to use (default: without_leaky)')
    parser.add_argument('--compare', action='store_true',
                       help='Compare both model scenarios')
    parser.add_argument('--output', type=str, default=None,
                       help='Output filename for predictions')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison
        compare_model_scenarios(args.data_path)
    else:
        # Single prediction
        predict_alumni_batch(args.data_path, args.scenario, args.output)


if __name__ == "__main__":
    
    # If running directly, use example
    if len(sys.argv) > 1:
        main()
    else:
        # Example usage
        print("Example usage:")
        print("python predict_models.py data/01_raw/DATA TS SARJANA 2024.csv")
        print("python predict_models.py data/01_raw/DATA TS SARJANA 2024.csv --compare")
        print("\nOr use in Python:")
        print("from predict_models import predict_alumni_batch")
        print("predictions = predict_alumni_batch('path/to/data.csv')")