# Churn Prediction & Outreach Targeting

## Overview
This project predicts member churn using behavioral, engagement, and claims data, and uses the model outputs to guide outreach decisions.

The objective is not only to estimate churn risk, but to prioritize **who should receive outreach**, based on the expected reduction in churn probability.

---

## Data
The pipeline uses four member-level data sources:
- Churn labels
- Web visit activity
- App usage
- Insurance claims

Expected paths for train and test data are defined under config.json. Current setup: 
- data/train/
- data/test/

All datasets are aggregated at the member level and joined using `member_id`.

---

### Evaluation
- Standard classification metrics (precision, recall, ROC-AUC)
- Segment-level comparisons between outreach and non-outreach groups
- Visualizations designed for non-technical stakeholders

The solution intentionally stops short of full causal or uplift modeling, which would be a natural next step in a production setting.

---

## Project Structure
```text
.
├── data/                 # Raw and processed member datasets
├── docs/                 # Documentation and methodology notes
├── figures/              # Generated EDA visualizations and plots
├── src/                  # Source code directory
│   ├── churn_classification_model.py  # Model training and evaluation
│   ├── churn_outreach_selection.py     # Outreach optimization logic
│   ├── create_feat_df.py               # Feature engineering pipeline
│   ├── data_exploration.py             # Script for EDA and visualization
│   └── utils.py                        # Helper functions (loading, etc.)
├── .gitignore            # Git exclusion rules
├── config.json           # Paths and hyperparameter configurations
├── poetry.lock           # Dependency lock file (Poetry)
├── pyproject.toml        # Build system and dependency management
├── README.md             # Project overview and instructions
└── requirements.txt      # Dependency list for pip
````

---

## Setup

### Requirements
- Python >= 3.11, < 3.15
- Poetry

Dependencies are managed with Poetry.
A `requirements.txt` file is provided for convenience and is generated from `pyproject.toml`.


### Install Poetry
If Poetry is not installed:
```bash
pip install poetry
````

### Install Dependencies

From the project root:

```bash
poetry install
```

---

## Configuration

Data file paths are defined in:

```
config.json
```

Update this file if directory locations change.

---

## Running the Pipeline

All commands should be run inside the Poetry environment.


### 1. run exploration script for visualiations and embedding clustering output

```bash
poetry run python src/data_exploration.py
```

---

### 2. Train and evaluate Model

```bash
poetry run python src/churn_classification_model.py -t
```

Models can run with feature selection based on feature importance (default) or without (-m xgb)  
To evaluate existing model without retraining, remove -t  

---


### 3. Generate Outreach Targeting Decisions

```bash
poetry run python src/churn_outreach_selection.py 
```


---

## Approach

### Feature Engineering
- Behavioral features: recency, frequency, trends
- App usage intensity and variability
- Claims-based features (e.g. ICD code usage patterns)
- Outreach indicator included as a model feature

Where linear assumptions break (e.g. time-of-day effects), circular statistics are used.  
High-dimensional and sparse features are handled directly by the model.

---

### Modeling
- **XGBoost classifier** is used due to:
  - Nonlinear interactions
  - Mixed feature types
  - Robustness to sparsity and missing values
  - Explainability
- Regularization and early stopping are applied to limit overfitting
- Cross-validation is used for evaluation
- Feature selection is included to reduce noisy features

The model outputs a churn probability per member.

---

### Outreach Targeting Logic
Rather than targeting purely by churn risk, members are selected based on:

- The difference between:
  - Predicted churn probability **with outreach**
  - Predicted churn probability **without outreach**
- Predicted churn probability **without outreach**  


Members are targeted when:
- Baseline churn risk exceeds a threshold (`p0`)
- Expected reduction in churn probability exceeds a minimum effect size (`Δp`)

This balances outreach cost with expected retention lift.

The current implementation uses p0 = 0.5 and Δp = 0.01; both parameters should be calibrated using cost–benefit analysis.

---

## Notes on Scope & Assumptions

- Outreach cost was not provided. If outreach cost is high relative to churn cost, the churn-risk classification threshold should be increased to limit outreach volume.


