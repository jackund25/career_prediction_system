"""
Generate project directory structure
"""
import os
from pathlib import Path

def generate_tree_structure(start_path, prefix="", max_depth=None, current_depth=0):
    """Generate tree structure of directory"""
    if max_depth is not None and current_depth >= max_depth:
        return ""
    
    start_path = Path(start_path)
    if not start_path.exists():
        return f"{prefix}[Directory not found: {start_path}]\n"
    
    items = []
    try:
        # Get all items and sort them
        all_items = list(start_path.iterdir())
        # Separate directories and files
        dirs = [item for item in all_items if item.is_dir()]
        files = [item for item in all_items if item.is_file()]
        
        # Sort each group
        dirs.sort(key=lambda x: x.name.lower())
        files.sort(key=lambda x: x.name.lower())
        
        # Combine: directories first, then files
        items = dirs + files
        
    except PermissionError:
        return f"{prefix}[Permission Denied]\n"
    
    # Skip certain directories
    skip_dirs = {
        '__pycache__', '.git', '.vscode', 'node_modules', 
        '.pytest_cache', '.mypy_cache', 'venv', 'env',
        'career-pred-sys'  # Virtual environment
    }
    
    # Skip certain file extensions
    skip_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll'}
    
    tree_str = ""
    
    for i, item in enumerate(items):
        # Skip unwanted items
        if item.name in skip_dirs:
            continue
        if item.suffix in skip_extensions:
            continue
        if item.name.startswith('.') and item.name not in {'.gitignore', '.gitkeep'}:
            continue
            
        is_last = i == len(items) - 1
        
        # Find if this is actually the last item after filtering
        remaining_items = items[i+1:]
        has_more_items = any(
            not (
                remaining_item.name in skip_dirs or
                remaining_item.suffix in skip_extensions or
                (remaining_item.name.startswith('.') and remaining_item.name not in {'.gitignore', '.gitkeep'})
            )
            for remaining_item in remaining_items
        )
        
        current_prefix = "└── " if not has_more_items else "├── "
        next_prefix = "    " if not has_more_items else "│   "
        
        tree_str += f"{prefix}{current_prefix}{item.name}\n"
        
        if item.is_dir():
            tree_str += generate_tree_structure(
                item, 
                prefix + next_prefix, 
                max_depth, 
                current_depth + 1
            )
    
    return tree_str

def create_project_structure():
    """Create comprehensive project structure documentation"""
    
    # Get project root
    project_root = Path(__file__).parent
    project_name = project_root.name
    
    # Generate tree structure
    print("Generating project directory structure...")
    tree_structure = generate_tree_structure(project_root, max_depth=4)
    
    # Create detailed structure with descriptions
    structure_with_descriptions = f"""
# {project_name.upper().replace('-', ' ')} - PROJECT STRUCTURE

## Directory Tree
```
{project_name}/
{tree_structure}
```

## Directory Descriptions

### Root Level
- **README.md**: Project documentation and setup instructions
- **requirements.txt**: Python package dependencies
- **career-pred-sys/**: Virtual environment directory (excluded from tree)

### `/data/`
Data storage organized by processing stage:
- **01_raw/**: Original, unprocessed datasets
  - `data 2016 - daffari_raw.csv`: Training data from 2016
  - `DATA TS SARJANA 2024.xlsx`: Prediction target data from 2024
- **02_processed/**: Cleaned and processed datasets
  - Feature-engineered data
  - Column analysis results
  - Preprocessed datasets ready for modeling

### `/models/`
Trained model artifacts organized by approach:
- **advanced/**: Advanced models (XGBoost, CatBoost, Ensemble)
  - `without_leaky/`: Models trained without leaky features
  - `with_leaky/`: Models trained with potentially leaky features
- **baseline/**: Baseline models (Random Forest, MLP)
- Individual model files (.pkl), preprocessors, and metadata

### `/notebooks/`
Jupyter notebooks for analysis and experimentation:
- **01_eda_dan_validasi.ipynb**: Exploratory Data Analysis and validation
- **02_baseline_model_development.ipynb**: Baseline model development
- **03_analisis_hasil_akhir.ipynb**: Final results analysis

### `/src/`
Source code organized by functionality:
- **data/**: Data processing and compatibility modules
  - `make_dataset.py`: Dataset creation utilities
  - `ensure_data_compatibility.py`: Cross-year data standardization
- **features/**: Feature engineering and selection
  - `build_features.py`: Main feature engineering pipeline
- **models/**: Model training and prediction
  - `train_models.py`: Advanced model training with optimization
  - `predict_model.py`: Prediction pipeline for new data
  - `visualize_results.py`: Results visualization and analysis

### `/results/`
Experimental results and outputs:
- **predictions/**: Model predictions on test data
  - CSV files with predictions and confidence scores
  - JSON files with evaluation metrics
- **advanced/**: Advanced model results by scenario
  - Validation results
  - Performance metrics
  - Feature importance data
- **baseline/**: Baseline model results

### `/reports/`
Documentation and visualizations:
- **figures/**: Generated plots and charts
  - Model comparison plots
  - Feature importance visualizations
  - Performance analysis charts

## Key Features

### Data Processing
- Cross-temporal data standardization (2016 ↔ 2024)
- Feature engineering with interaction terms
- Missing value handling strategies

### Model Development
- Baseline models: Random Forest, MLP
- Advanced models: XGBoost, CatBoost, Ensemble
- Hyperparameter optimization with Optuna
- Feature selection using SHAP values

### Evaluation Framework
- Leaky vs non-leaky feature comparison
- Temporal validation (train on 2016, test on 2024)
- Comprehensive metrics and visualizations

### Prediction Pipeline
- Clean API for batch predictions
- Confidence scoring and uncertainty quantification
- Model agreement analysis

## Usage Workflow

1. **Data Preparation**: Process raw data using `/src/data/` modules
2. **Feature Engineering**: Create features using `/src/features/build_features.py`
3. **Model Training**: Train models using `/src/models/train_models.py`
4. **Prediction**: Make predictions using `/src/models/predict_model.py`
5. **Analysis**: Analyze results using notebooks and visualization tools

## Technology Stack
- **Python 3.12+**
- **Machine Learning**: scikit-learn, XGBoost, CatBoost, LightGBM
- **Optimization**: Optuna
- **Interpretation**: SHAP
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Development**: Jupyter, pytest

---
Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Project: Career Prediction System for ITB Alumni
"""
    
    # Save to file
    output_file = project_root / 'PROJECT_STRUCTURE.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(structure_with_descriptions)
    
    print(f"✅ Project structure saved to: {output_file}")
    print(f"📂 Total lines: {len(structure_with_descriptions.split(chr(10)))}")
    
    return output_file

if __name__ == "__main__":
    create_project_structure()
