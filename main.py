import sys
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.exceptions import ConvergenceWarning
import warnings

# Suppress minor convergence warnings during hyperparameter search
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_selected_dataset():
    """Prompts user via terminal to select or load a dataset."""
    print("\n[Select Dataset]")
    print("1. Breast Cancer Diagnostic Dataset (Binary Classification)")
    print("2. Wine Recognition Dataset (Multi-class Classification)")
    print("3. Load Custom CSV File")
    
    choice = input("Enter your choice (1-3) [Default: 1]: ").strip()
    if not choice:
        choice = "1"
        
    try:
        if choice == "1":
            data = load_breast_cancer()
            X, y = data.data, data.target
            print("-> Loaded Breast Cancer dataset successfully.")
        elif choice == "2":
            data = load_wine()
            X, y = data.data, data.target
            print("-> Loaded Wine dataset successfully.")
        elif choice == "3":
            path = input("Enter path to CSV file: ").strip()
            target_col = input("Enter target column name: ").strip()
            df = pd.read_csv(path)
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in CSV.")
            X = df.drop(columns=[target_col]).values
            y = df[target_col].values
            print(f"-> Loaded custom dataset from '{path}' successfully.")
        else:
            print("Invalid choice. Defaulting to Breast Cancer dataset.")
            data = load_breast_cancer()
            X, y = data.data, data.target
            
        return X, y
    except Exception as e:
        print(f"\n[Error] Failed to load dataset: {e}", file=sys.stderr)
        sys.exit(1)


def build_automl_pipeline():
    """
    Constructs the base Pipeline structure and the multi-estimator 
    parameter grids for RandomizedSearchCV without parameter collision.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('selector', SelectKBest(score_func=f_classif)),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    param_distributions = [
        # --- Logistic Regression ---
        {
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'selector': [SelectKBest(score_func=f_classif)],
            'selector__k': [5, 10, 'all'],
            'classifier': [LogisticRegression(max_iter=2000, random_state=42)],
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__solver': ['lbfgs', 'saga']
        },
        {
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'selector': ['passthrough'],
            'classifier': [LogisticRegression(max_iter=2000, random_state=42)],
            'classifier__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'classifier__solver': ['lbfgs', 'saga']
        },
        
        # --- Random Forest ---
        {
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'selector': [SelectKBest(score_func=f_classif)],
            'selector__k': [5, 10, 'all'],
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        {
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'selector': ['passthrough'],
            'classifier': [RandomForestClassifier(random_state=42)],
            'classifier__n_estimators': [50, 100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 20],
            'classifier__min_samples_split': [2, 5, 10]
        },
        
        # --- Support Vector Classifier ---
        {
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'selector': [SelectKBest(score_func=f_classif)],
            'selector__k': [5, 10, 'all'],
            'classifier': [SVC(random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        },
        {
            'scaler': [StandardScaler(), MinMaxScaler(), None],
            'selector': ['passthrough'],
            'classifier': [SVC(random_state=42)],
            'classifier__C': [0.1, 1.0, 10.0, 100.0],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    ]
    
    return pipeline, param_distributions


def run_automl():
    print("==================================================")
    print("      CUSTOM AUTOML PIPELINE ENGINE (SKLEARN)     ")
    print("==================================================")
    
    # 1. Load Data
    X, y = load_selected_dataset()
    
    # 2. Terminal Configuration Inputs
    try:
        n_iter = int(input("Enter number of search iterations for RandomizedSearch [Default: 10]: ") or 10)
        outer_folds = int(input("Enter number of outer folds for Nested CV [Default: 3]: ") or 3)
        inner_folds = int(input("Enter number of inner folds for Hyperparameter Tuning [Default: 3]: ") or 3)
    except ValueError:
        print("[Warning] Invalid numeric input detected. Using default values (iter=10, outer=3, inner=3).")
        n_iter, outer_folds, inner_folds = 10, 3, 3

    print("\n[Status] Initializing pipeline components and parameter spaces...")
    pipeline, param_distributions = build_automl_pipeline()
    
    # 3. Setup Nested Cross-Validation Structure
    inner_cv = KFold(n_splits=inner_folds, shuffle=True, random_state=42)
    outer_cv = KFold(n_splits=outer_folds, shuffle=True, random_state=42)
    
    search_engine = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=inner_cv,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\n[Status] Running Nested Cross-Validation ({outer_folds} outer folds x {inner_folds} inner folds)...")
    try:
        nested_scores = cross_val_score(search_engine, X, y, cv=outer_cv, scoring='accuracy', n_jobs=-1)
    except Exception as e:
        print(f"\n[Error] Failed during execution of model search: {e}", file=sys.stderr)
        sys.exit(1)
        
    print("\n--------------------------------------------------")
    print("               NESTED CV RESULTS                  ")
    print("--------------------------------------------------")
    print(f"Outer Fold Accuracy Scores : {np.round(nested_scores, 4)}")
    print(f"Mean Generalization Accuracy: {nested_scores.mean():.4f} (+/- {nested_scores.std() * 2:.4f})")
    
    # 4. Fit Final Optimal Pipeline on Full Dataset
    print("\n[Status] Fitting final optimal estimator on full dataset...")
    search_engine.fit(X, y)
    best_estimator = search_engine.best_estimator_
    
    print("\n==================================================")
    print("              OPTIMAL PIPELINE FOUND              ")
    print("==================================================")
    print(f"Best Estimator Class: {type(best_estimator.named_steps['classifier']).__name__}")
    print("\nOptimal Hyperparameters & Configuration:")
    for step_name, step_obj in best_estimator.named_steps.items():
        print(f"  - Step '{step_name}': {step_obj}")
        
    print(f"\nBest Cross-Validation Score (Inner Search): {search_engine.best_score_:.4f}")
    print("==================================================\n")


if __name__ == "__main__":
    run_automl()