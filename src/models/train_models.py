"""
Advanced model training with XGBoost, CatBoost, Optuna optimization, and Soft Voting Ensemble
Following the flowchart: Optuna -> XGBoost/CatBoost -> Ensemble -> SHAP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, confusion_matrix)

# Advanced ML libraries
import optuna
from optuna.samplers import TPESampler
import xgboost as xgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import shap

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Custom imports
import sys
sys.path.append(str(Path(__file__).parent.parent))
from features.build_features import FeatureEngineer, TARGET_COLUMNS, LEAKY_COLUMN

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdvancedModelTrainer:
    """Train XGBoost and CatBoost with Optuna optimization and create ensemble"""
    
    def __init__(self, include_leaky_column: bool = False, n_trials: int = 50):
        """
        Initialize trainer
        
        Args:
            include_leaky_column: Whether to include potentially leaky column
            n_trials: Number of Optuna trials for hyperparameter optimization
        """
        self.include_leaky_column = include_leaky_column
        self.n_trials = n_trials
        self.feature_engineer = FeatureEngineer(include_leaky_column=include_leaky_column)
        
        # Model storage
        self.models = {}
        self.ensemble = None
        self.best_params = {}
        self.results = {}
        
        # Create directories
        scenario = 'with_leaky' if include_leaky_column else 'without_leaky'
        self.model_dir = Path('models') / 'advanced' / scenario
        self.results_dir = Path('results') / 'advanced' / scenario
        self.plots_dir = Path('plots') / 'advanced' / scenario
        
        for dir_path in [self.model_dir, self.results_dir, self.plots_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"AdvancedModelTrainer initialized:")
        logger.info(f"- Include leaky column: {include_leaky_column}")
        logger.info(f"- Optuna trials: {n_trials}")
        logger.info(f"- Output directory: {scenario}")
    
    def load_and_prepare_data(self, train_path: str, test_path: str = None):
        """Load and prepare data following the flowchart"""
        logger.info("Loading and preparing data...")
        
        # Load training data (2016)
        df_train = pd.read_csv(train_path)
        logger.info(f"Loaded training data: {df_train.shape}")
        
        # Process features
        df_processed = self.feature_engineer.process_features(
            df_train,
            common_columns=TARGET_COLUMNS,
            is_training=True,
            is_2017_data=False
        )
        
        # Get numeric features only
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in ['target', 'NIM']]
        
        X = df_processed[feature_cols]
        y = df_processed['target']
        
        # Encode target
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split 80/20 as per flowchart
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Store feature names
        self.feature_names = feature_cols
        
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Validation set: {X_val_scaled.shape}")
        
        # Prepare test data if provided
        X_test_scaled, y_test = None, None
        if test_path:
            df_test = pd.read_csv(test_path)
            logger.info(f"Loaded test data: {df_test.shape}")
            
            df_test_processed = self.feature_engineer.process_features(
                df_test,
                common_columns=TARGET_COLUMNS,
                is_training=False,
                is_2017_data=True
            )
            
            # Ensure same features
            X_test = df_test_processed[feature_cols]
            
            if 'target' in df_test_processed.columns:
                y_test = self.label_encoder.transform(df_test_processed['target'])
                X_test_scaled = self.scaler.transform(X_test)
                logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, y_train, y_val, X_test_scaled, y_test
    
    def objective_xgboost(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for XGBoost"""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='binary')
        
        return f1
    
    def objective_catboost(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for CatBoost"""
        params = {
            'iterations': trial.suggest_int('iterations', 100, 500),
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 10),
            'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': False
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average='binary')
        
        return f1
    
    def optimize_with_optuna(self, X_train, y_train, X_val, y_val):
        """Run Optuna optimization for both models"""
        logger.info("Starting Optuna hyperparameter optimization...")
        
        # Optimize XGBoost
        logger.info(f"Optimizing XGBoost with {self.n_trials} trials...")
        study_xgb = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study_xgb.optimize(
            lambda trial: self.objective_xgboost(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials
        )
        
        self.best_params['xgboost'] = study_xgb.best_params
        logger.info(f"Best XGBoost params: {study_xgb.best_params}")
        logger.info(f"Best XGBoost F1: {study_xgb.best_value:.4f}")
        
        # Optimize CatBoost
        logger.info(f"Optimizing CatBoost with {self.n_trials} trials...")
        study_cat = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study_cat.optimize(
            lambda trial: self.objective_catboost(trial, X_train, y_train, X_val, y_val),
            n_trials=self.n_trials
        )
        
        self.best_params['catboost'] = study_cat.best_params
        logger.info(f"Best CatBoost params: {study_cat.best_params}")
        logger.info(f"Best CatBoost F1: {study_cat.best_value:.4f}")
        
        # Save Optuna studies
        joblib.dump(study_xgb, self.results_dir / 'optuna_study_xgboost.pkl')
        joblib.dump(study_cat, self.results_dir / 'optuna_study_catboost.pkl')
        
        return study_xgb, study_cat
    
    def train_final_models(self, X_train, y_train, X_val, y_val):
        """Train final models with best parameters"""
        logger.info("Training final models with best parameters...")
        
        # Train XGBoost
        xgb_params = self.best_params['xgboost'].copy()
        xgb_params.update({
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        })
        
        self.models['xgboost'] = XGBClassifier(**xgb_params)
        self.models['xgboost'].fit(X_train, y_train)
        
        # Train CatBoost
        cat_params = self.best_params['catboost'].copy()
        cat_params.update({
            'random_state': 42,
            'verbose': False
        })
        
        self.models['catboost'] = CatBoostClassifier(**cat_params)
        self.models['catboost'].fit(X_train, y_train)
        
        # Evaluate individual models
        for model_name, model in self.models.items():
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            self.results[model_name] = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred, average='binary'),
                'recall': recall_score(y_val, y_pred, average='binary'),
                'f1': f1_score(y_val, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_val, y_proba),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
            
            logger.info(f"{model_name} - Val F1: {self.results[model_name]['f1']:.4f}")
    
    def create_ensemble(self, X_train, y_train, X_val, y_val):
        """Create soft voting ensemble as per flowchart"""
        logger.info("Creating Soft Voting Ensemble...")
        
        # Create ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgboost', self.models['xgboost']),
                ('catboost', self.models['catboost'])
            ],
            voting='soft'
        )
        
        # Train ensemble
        self.ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        y_pred = self.ensemble.predict(X_val)
        y_proba = self.ensemble.predict_proba(X_val)[:, 1]
        
        self.results['ensemble'] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='binary'),
            'recall': recall_score(y_val, y_pred, average='binary'),
            'f1': f1_score(y_val, y_pred, average='binary'),
            'roc_auc': roc_auc_score(y_val, y_proba),
            'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
        }
        
        logger.info(f"Ensemble - Val F1: {self.results['ensemble']['f1']:.4f}")
        
        # Compare with individual models
        improvement_xgb = self.results['ensemble']['f1'] - self.results['xgboost']['f1']
        improvement_cat = self.results['ensemble']['f1'] - self.results['catboost']['f1']
        
        logger.info(f"Ensemble improvement over XGBoost: {improvement_xgb:+.4f}")
        logger.info(f"Ensemble improvement over CatBoost: {improvement_cat:+.4f}")
    
    def run_shap_analysis(self, X_val, save_plots=True):
        """Run SHAP analysis as per flowchart"""
        logger.info("Running SHAP analysis...")
        
        # Use ensemble for SHAP analysis
        explainer = shap.Explainer(self.ensemble.predict, X_val)
        shap_values = explainer(X_val)
        
        if save_plots:
            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_val, feature_names=self.feature_names, 
                            show=False)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'shap_summary.png', dpi=300)
            plt.close()
            
            # Feature importance bar plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X_val, feature_names=self.feature_names,
                            plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(self.plots_dir / 'shap_importance.png', dpi=300)
            plt.close()
            
            logger.info(f"SHAP plots saved to {self.plots_dir}")
        
        # Get feature importance
        shap_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(shap_values.values).mean(0)
        }).sort_values('importance', ascending=False)
        
        # Save importance
        shap_importance.to_csv(self.results_dir / 'shap_feature_importance.csv', index=False)
        
        return shap_values, shap_importance
    
    def evaluate_on_test(self, X_test, y_test):
        """Evaluate all models on test set"""
        logger.info("Evaluating on test set (2017 data)...")
        
        test_results = {}
        
        # Evaluate individual models and ensemble
        all_models = {**self.models, 'ensemble': self.ensemble}
        
        for model_name, model in all_models.items():
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            test_results[model_name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='binary'),
                'recall': recall_score(y_test, y_pred, average='binary'),
                'f1': f1_score(y_test, y_pred, average='binary'),
                'roc_auc': roc_auc_score(y_test, y_proba)
            }
            
            logger.info(f"{model_name} - Test F1: {test_results[model_name]['f1']:.4f}")
            
            # Temporal degradation
            val_f1 = self.results[model_name]['f1']
            test_f1 = test_results[model_name]['f1']
            degradation = val_f1 - test_f1
            logger.info(f"{model_name} - Temporal degradation: {degradation:.4f}")
        
        return test_results
    
    def save_models_and_results(self):
        """Save all models and results"""
        logger.info("Saving models and results...")
        
        # Save models
        joblib.dump(self.models['xgboost'], self.model_dir / 'xgboost.pkl')
        joblib.dump(self.models['catboost'], self.model_dir / 'catboost.pkl')
        joblib.dump(self.ensemble, self.model_dir / 'ensemble.pkl')
        
        # Save preprocessing
        joblib.dump(self.feature_engineer, self.model_dir / 'feature_engineer.pkl')
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')
        joblib.dump(self.label_encoder, self.model_dir / 'label_encoder.pkl')
        
        # Save parameters
        with open(self.results_dir / 'best_params.json', 'w') as f:
            json.dump(self.best_params, f, indent=4)
        
        # Save results
        with open(self.results_dir / 'validation_results.json', 'w') as f:
            json.dump(self.results, f, indent=4)
        
        logger.info(f"Models saved to {self.model_dir}")
        logger.info(f"Results saved to {self.results_dir}")
    
    def run_complete_pipeline(self, train_path: str, test_path: str = None):
        """Run complete training pipeline following the flowchart"""
        logger.info("="*80)
        logger.info(f"Starting Advanced Model Training Pipeline")
        logger.info(f"Scenario: {'With' if self.include_leaky_column else 'Without'} leaky column")
        logger.info("="*80)
        
        # 1. Load and prepare data
        X_train, X_val, y_train, y_val, X_test, y_test = self.load_and_prepare_data(
            train_path, test_path
        )
        
        # 2. Optuna optimization
        self.optimize_with_optuna(X_train, y_train, X_val, y_val)
        
        # 3. Train final models
        self.train_final_models(X_train, y_train, X_val, y_val)
        
        # 4. Create ensemble
        self.create_ensemble(X_train, y_train, X_val, y_val)
        
        # 5. SHAP analysis
        shap_values, shap_importance = self.run_shap_analysis(X_val)
        
        # 6. Test evaluation if available
        test_results = None
        if X_test is not None and y_test is not None:
            test_results = self.evaluate_on_test(X_test, y_test)
            
            # Save test results
            with open(self.results_dir / 'test_results.json', 'w') as f:
                json.dump(test_results, f, indent=4)
        
        # 7. Save everything
        self.save_models_and_results()
        
        # 8. Generate summary report
        self.generate_summary_report(test_results)
        
        return self.results, test_results
    
    def generate_summary_report(self, test_results=None):
        """Generate summary report"""
        report = []
        report.append("="*80)
        report.append("ADVANCED MODEL TRAINING SUMMARY")
        report.append("="*80)
        report.append(f"Scenario: {'With' if self.include_leaky_column else 'Without'} leaky column")
        report.append(f"Optuna trials: {self.n_trials}")
        report.append("")
        
        report.append("Validation Results:")
        for model_name in ['xgboost', 'catboost', 'ensemble']:
            f1 = self.results[model_name]['f1']
            report.append(f"  {model_name}: F1={f1:.4f}")
        
        if test_results:
            report.append("\nTest Results (2017 data):")
            for model_name in ['xgboost', 'catboost', 'ensemble']:
                f1 = test_results[model_name]['f1']
                report.append(f"  {model_name}: F1={f1:.4f}")
        
        report.append("\nBest Model: Ensemble (Soft Voting)")
        report.append("="*80)
        
        report_text = "\n".join(report)
        
        # Save report
        with open(self.results_dir / 'summary_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)


def run_experiment_comparison(train_path: str, test_path: str = None):
    """Run experiments with and without leaky column"""
    
    results_comparison = {}
    
    # Run without leaky column
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 1: WITHOUT LEAKY COLUMN")
    logger.info("="*80)
    
    trainer_no_leak = AdvancedModelTrainer(include_leaky_column=False, n_trials=50)
    val_results_no_leak, test_results_no_leak = trainer_no_leak.run_complete_pipeline(
        train_path, test_path
    )
    
    results_comparison['without_leaky'] = {
        'validation': val_results_no_leak,
        'test': test_results_no_leak
    }
    
    # Run with leaky column
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT 2: WITH LEAKY COLUMN")
    logger.info("="*80)
    
    trainer_with_leak = AdvancedModelTrainer(include_leaky_column=True, n_trials=50)
    val_results_with_leak, test_results_with_leak = trainer_with_leak.run_complete_pipeline(
        train_path, test_path
    )
    
    results_comparison['with_leaky'] = {
        'validation': val_results_with_leak,
        'test': test_results_with_leak
    }
    
    # Save comparison
    with open('results/advanced/experiment_comparison.json', 'w') as f:
        json.dump(results_comparison, f, indent=4)
    
    # Print comparison
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON")
    print("="*80)
    
    for scenario in ['without_leaky', 'with_leaky']:
        print(f"\n{scenario.upper()}:")
        ensemble_val_f1 = results_comparison[scenario]['validation']['ensemble']['f1']
        print(f"  Validation F1: {ensemble_val_f1:.4f}")
        
        if results_comparison[scenario]['test']:
            ensemble_test_f1 = results_comparison[scenario]['test']['ensemble']['f1']
            print(f"  Test F1: {ensemble_test_f1:.4f}")
    
    # Calculate impact
    val_impact = (results_comparison['with_leaky']['validation']['ensemble']['f1'] - 
                  results_comparison['without_leaky']['validation']['ensemble']['f1'])
    print(f"\nLeaky column impact (validation): {val_impact:+.4f}")
    
    if test_results_no_leak and test_results_with_leak:
        test_impact = (results_comparison['with_leaky']['test']['ensemble']['f1'] - 
                      results_comparison['without_leaky']['test']['ensemble']['f1'])
        print(f"Leaky column impact (test): {test_impact:+.4f}")


def main():
    """Main function"""
    # Paths
    train_path = 'data/01_raw/DATA TS SARJANA 2023.csv'
    test_path = 'data/01_raw/DATA TS SARJANA 2024.csv'
    
    # Check paths
    from pathlib import Path
    if not Path(train_path).exists():
        train_path = '../../' + train_path
    if not Path(test_path).exists():
        test_path = '../../' + test_path
    
    # Run comparison experiments
    run_experiment_comparison(train_path, test_path)


if __name__ == "__main__":
    main()