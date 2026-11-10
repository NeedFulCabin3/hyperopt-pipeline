# Hyperopt Pipeline

Automated Model Selection & Nested Cross-Validation Engine

## Overview

Manual hyperparameter tuning across disparate machine learning models frequently introduces accidental data leakage and optimization bias. Data scientists often struggle with tightly coupled preprocessing steps, feature selection parameters, and algorithm-specific hyperparameters during experimental iterations. **sklearn-hyperopt-pipeline** solves this operational friction by encapsulating the entire model exploration workflow inside compound scikit-learn pipelines, executing rigorous nested cross-validation to guarantee unbiased performance evaluation and optimal estimator extraction.

## How It Works

The engine operates through a structured execution pipeline managed by scikit-learn:

1. **Dataset Ingestion & Validation:** Prompts the user via the terminal to select between benchmark datasets (Breast Cancer Diagnostic, Wine Recognition) or ingest an arbitrary local CSV file, performing schema verification and target column extraction.
2. **Compound Pipeline Composition:** Chains preprocessing transformers (`StandardScaler`, `MinMaxScaler`, or `None`), feature selection modules (`SelectKBest` with ANOVA F-value scoring or `'passthrough'`), and diverse estimators (`LogisticRegression`, `RandomForestClassifier`, `SVC`).
3. **Partitioned Parameter Distribution:** Utilizes mutually exclusive parameter dictionaries mapped directly to each estimator to prevent parameter collision errors during search iterations.
4. **Nested Cross-Validation:** Deploys an inner `RandomizedSearchCV` loop over stratified K-folds to identify optimal hyperparameter configurations, wrapped within an outer `KFold` cross-validation loop to compute unbiased generalization accuracy.
5. **Final Estimation:** Fits the winning estimator configuration across the complete dataset and exposes internal parameters and performance metrics.

## Key Features

* **Multi-Algorithm Search Space:** Simultaneously explores hyperparameter grids across linear models, ensemble methods, and support vector machines.
* **Dynamic Feature Selection Tuning:** Evaluates varying feature counts (`k=5`, `10`, `'all`) or bypasses selection entirely depending on performance gains.
* **Interactive Terminal CLI:** Simple interactive prompt supporting built-in datasets and custom CSV files without requiring code modifications.
* **Parallelized Execution:** Leverages multi-core processing (`n_jobs=-1`) across cross-validation folds and randomized searches to minimize execution time.

## Tech Stack & Core Dependencies

* **Python 3.9+:** Core runtime environment.
* **scikit-learn:** Provides `Pipeline`, `RandomizedSearchCV`, cross-validation splitters, scalers, feature selectors, and classifiers.
* **numpy:** Handles numerical array operations and score aggregations.
* **pandas:** Manages tabular data ingestion and CSV parsing.

## Environment & Web-Based Quick Start

### GitHub Codespaces Setup
Launch a cloud development environment directly from the repository homepage:
1. Click the **Code** dropdown on GitHub and select the **Codespaces** tab.
2. Click **Create codespace on main**.
3. Once the container initializes, execute the setup commands in the terminal.

### Local or Virtual Environment Setup

```bash
# Clone the repository
git clone [https://github.com/username/sklearn-hyperopt-pipeline.git](https://github.com/username/sklearn-hyperopt-pipeline.git)
cd sklearn-hyperopt-pipeline

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run the pipeline engine
python main.py
```

## Repository Structure

```bash
sklearn-hyperopt-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .gitignore
├── README.md
├── requirements.txt
└── main.py
```

## Roadmap

**[ ] Type Stubs & Static Analysis:** Introduce strict type annotations (mypy) across all data loading and pipeline construction functions.

**[ ] YAML/JSON Configuration Loading:** Externalize search spaces and hyperparameter grids into configuration files to support headless execution.

**[ ] Structured Logging Migration:** Replace standard terminal print statements with Python's native logging module for better observability and log level management.