"""
Module untuk training model utama (XGBoost/CatBoost) dengan feature selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib
import logging
from typing import Dict, List, Tuple, Any

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.ensemble import VotingClassifier

# Advanced models
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb

# Feature selection and interpretation
import shap
from sklearn.feature_selection import mutual_info_classif, SelectKBest

# Hyperparameter optimization
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import paths and modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src import RAW_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.features.build_features import FeatureEngineer, load_column_info


class AdvancedModelTrainer:
    """Class for training advanced models with feature selection"""
    
    def __init__(self, random_state: int = 42):
        """Initialize trainer"""
        self.random_state = random_state
        self.fe = FeatureEngineer()
        self.models = {}
        self.best_features = None
        self.preprocessor = None
        
    def load_and_prepare_data(self, use_common_columns: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and prepare data for training
        
        Args:
            use_common_columns: Whether to use only common columns
            
        Returns:
            X, y tuple
        """
        # Load data
        data_path = RAW_DATA_DIR / 'DATA TS SARJANA 2024.xlsx'
        df = pd.read_excel(data_path)
        logger.info(f"Loaded data: {df.shape}")
        
        # Load column info if using common columns
        common_columns = None
        if use_common_columns:
            column_info = load_column_info()
            if column_info:
                common_columns = column_info.get('common_columns', None)
                logger.info(f"Using {len(common_columns)} common columns")
        
        # Process features
        df_processed = self.fe.process_features(df, common_columns=common_columns)
        
        # Find and separate target
        target_col = self._find_target_column(df_processed)
        if not target_col:
            raise ValueError("Target column not found!")
        
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Encode target if necessary
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            # Save label encoder
            joblib.dump(le, MODELS_DIR / 'label_encoder.pkl')
            logger.info(f"Encoded target classes: {le.classes_}")
        
        return X, y
    
    def _find_target_column(self, df: pd.DataFrame) -> str:
        """Find target column in dataframe"""
        target_candidates = ['Lulus_label', 'lulus', 'label', 'target']
        for col in target_candidates:
            if col in df.columns:
                return col
        return None
    
    def create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create preprocessing pipeline
        
        Args:
            X: Feature dataframe
            
        Returns:
            ColumnTransformer preprocessor
        """
        # Identify column types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")
        
        # Create transformers
        numeric_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def feature_selection_with_shap(self, X: pd.DataFrame, y: pd.Series, 
                                   n_features: int = 30) -> List[str]:
        """
        Select top features using SHAP values
        
        Args:
            X: Features
            y: Target
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger.info("Starting SHAP-based feature selection...")
        
        # Create a simple XGBoost model for SHAP analysis
        X_temp = self.preprocessor.fit_transform(X)
        
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_temp, y)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_temp)
        
        # Get feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            shap_importance = np.abs(shap_values).mean(axis=1).mean(axis=0)
        else:
            # Binary case
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Get feature names after preprocessing
        feature_names = []
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out().tolist()
        else:
            # Fallback for older sklearn versions
            feature_names = [f"feature_{i}" for i in range(X_temp.shape[1])]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': shap_importance
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_features = importance_df.head(n_features)['feature'].tolist()
        
        logger.info(f"Selected top {len(top_features)} features")
        
        # Save SHAP summary plot
        plt = shap.summary_plot(shap_values, X_temp, feature_names=feature_names, 
                               show=False, max_display=20)
        import matplotlib.pyplot as plt
        plt.savefig(PROCESSED_DATA_DIR / 'shap_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return top_features, importance_df
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val, n_trials: int = 50):
        """
        Optimize XGBoost hyperparameters using Optuna
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGBoost params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_catboost(self, X_train, y_train, X_val, y_val, n_trials: int = 50):
        """
        Optimize CatBoost hyperparameters using Optuna
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            n_trials: Number of optimization trials
            
        Returns:
            Best parameters
        """
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_state': self.random_state,
                'verbose': False
            }
            
            model = CatBoostClassifier(**params)
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            
            y_pred = model.predict(X_val)
            score = f1_score(y_val, y_pred, average='weighted')
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best CatBoost params: {study.best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return study.best_params
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, 
                    feature_selection: bool = True,
                    optimize_hyperparams: bool = True) -> Dict[str, Any]:
        """
        Train all models with optional feature selection and hyperparameter optimization
        
        Args:
            X: Features
            y: Target
            feature_selection: Whether to perform feature selection
            optimize_hyperparams: Whether to optimize hyperparameters
            
        Returns:
            Dictionary with trained models and metrics
        """
        # Create preprocessor
        self.create_preprocessor(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Further split train into train/val for optimization
        X_train_opt, X_val_opt, y_train_opt, y_val_opt = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state, stratify=y_train
        )
        
        # Transform data
        X_train_transformed = self.preprocessor.fit_transform(X_train)
        X_test_transformed = self.preprocessor.transform(X_test)
        X_train_opt_transformed = self.preprocessor.transform(X_train_opt)
        X_val_opt_transformed = self.preprocessor.transform(X_val_opt)
        
        # Feature selection
        if feature_selection:
            selected_features, importance_df = self.feature_selection_with_shap(X_train, y_train)
            self.best_features = selected_features
            
            # Save feature importance
            importance_df.to_csv(PROCESSED_DATA_DIR / 'feature_importance.csv', index=False)
        
        # Initialize results
        results = {}
        
        # Train XGBoost
        logger.info("Training XGBoost...")
        if optimize_hyperparams:
            xgb_params = self.optimize_xgboost(
                X_train_opt_transformed, y_train_opt,
                X_val_opt_transformed, y_val_opt
            )
        else:
            xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train_transformed, y_train)
        self.models['xgboost'] = xgb_model
        
        # Evaluate XGBoost
        y_pred_xgb = xgb_model.predict(X_test_transformed)
        y_proba_xgb = xgb_model.predict_proba(X_test_transformed)[:, 1]
        
        results['xgboost'] = {
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'f1_score': f1_score(y_test, y_pred_xgb, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba_xgb)
        }
        
        # Train CatBoost
        logger.info("Training CatBoost...")
        if optimize_hyperparams:
            cat_params = self.optimize_catboost(
                X_train_opt_transformed, y_train_opt,
                X_val_opt_transformed, y_val_opt
            )
        else:
            cat_params = {
                'iterations': 200,
                'depth': 6,
                'learning_rate': 0.1,
                'random_state': self.random_state,
                'verbose': False
            }
        
        cat_model = CatBoostClassifier(**cat_params)
        cat_model.fit(X_train_transformed, y_train, verbose=False)
        self.models['catboost'] = cat_model
        
        # Evaluate CatBoost
        y_pred_cat = cat_model.predict(X_test_transformed)
        y_proba_cat = cat_model.predict_proba(X_test_transformed)[:, 1]
        
        results['catboost'] = {
            'accuracy': accuracy_score(y_test, y_pred_cat),
            'f1_score': f1_score(y_test, y_pred_cat, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba_cat)
        }
        
        # Create ensemble model
        logger.info("Creating ensemble model...")
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('cat', cat_model)
            ],
            voting='soft'
        )
        ensemble.fit(X_train_transformed, y_train)
        self.models['ensemble'] = ensemble
        
        # Evaluate ensemble
        y_pred_ens = ensemble.predict(X_test_transformed)
        y_proba_ens = ensemble.predict_proba(X_test_transformed)[:, 1]
        
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, y_pred_ens),
            'f1_score': f1_score(y_test, y_pred_ens, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_proba_ens)
        }
        
        # Log results
        logger.info("\n=== Model Performance ===")
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name.upper()}:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
            logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(PROCESSED_DATA_DIR / 'model_results.csv')
        
        return results
    
    def save_models(self):
        """Save all trained models and preprocessor"""
        # Save preprocessor
        joblib.dump(self.preprocessor, MODELS_DIR / 'preprocessor.pkl')
        logger.info("Saved preprocessor")
        
        # Save models
        for name, model in self.models.items():
            filename = f"{name}_model.pkl"
            joblib.dump(model, MODELS_DIR / filename)
            logger.info(f"Saved {name} model")
        
        # Save feature list if available
        if self.best_features:
            with open(MODELS_DIR / 'selected_features.json', 'w') as f:
                json.dump(self.best_features, f, indent=2)
            logger.info("Saved selected features")
        
        # Determine best model based on F1-score
        results_df = pd.read_csv(PROCESSED_DATA_DIR / 'model_results.csv', index_col=0)
        best_model_name = results_df['f1_score'].idxmax()
        best_model = self.models[best_model_name]
        
        # Save best model as final model
        joblib.dump(best_model, MODELS_DIR / 'final_model.pkl')
        logger.info(f"Saved {best_model_name} as final model")
        
        # Save metadata
        metadata = {
            'best_model': best_model_name,
            'best_f1_score': float(results_df.loc[best_model_name, 'f1_score']),
            'feature_selection_used': self.best_features is not None,
            'n_features': len(self.best_features) if self.best_features else 'all'
        }
        
        with open(MODELS_DIR / 'model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Model training completed successfully!")


def main():
    """Main function to train models"""
    # Initialize trainer
    trainer = AdvancedModelTrainer()
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    X, y = trainer.load_and_prepare_data(use_common_columns=True)
    
    # Train models with feature selection and hyperparameter optimization
    logger.info("Starting model training...")
    results = trainer.train_models(
        X, y,
        feature_selection=True,
        optimize_hyperparams=True
    )
    
    # Save models
    trainer.save_models()
    
    # Load benchmark metrics for comparison
    try:
        with open(PROCESSED_DATA_DIR / 'benchmark_metrics.json', 'r') as f:
            benchmark = json.load(f)
        
        logger.info("\n=== Comparison with Baseline ===")
        logger.info(f"Baseline RF F1-Score: {benchmark.get('rf_f1', 'N/A'):.4f}")
        logger.info(f"Baseline MLP F1-Score: {benchmark.get('mlp_f1', 'N/A'):.4f}")
        
        results_df = pd.read_csv(PROCESSED_DATA_DIR / 'model_results.csv', index_col=0)
        logger.info(f"\nBest Advanced Model F1-Score: {results_df['f1_score'].max():.4f}")
        
    except FileNotFoundError:
        logger.warning("Benchmark metrics not found. Run baseline models first.")
    
    return results


if __name__ == "__main__":
    main()