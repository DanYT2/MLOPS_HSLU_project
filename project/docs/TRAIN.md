# `train.py` — Training Pipeline Documentation

## Overview

`train.py` is the end-to-end ML training pipeline for the Customer Churn Prediction system. It orchestrates data loading, feature engineering, Optuna-based hyperparameter optimization, 5-fold stratified cross-validation ensemble training (LightGBM + XGBoost), comprehensive MLflow experiment tracking, and model registration. Running the script produces a registered MLflow model (`CustomerChurnEnsemble@champion`) and a `submission.csv` of test-set predictions.

```bash
python project/train.py
```

---

## Module-Level Constants

| Constant | Value | Purpose |
|---|---|---|
| `N_SPLITS` | `5` | Number of folds for stratified K-fold cross-validation. |
| `RANDOM_STATE` | `42` | Seed for all random operations ensuring full reproducibility. |
| `SERVICE_COLUMNS` | 8 columns | Columns representing Yes/No telecom services. Used to derive the `TotalServices` feature by counting how many services a customer subscribes to. |
| `ONE_HOT_COLUMNS` | 10 columns | Categorical columns that undergo one-hot encoding via `pd.get_dummies`. Includes the service columns plus `InternetService`, `Contract`, and `PaymentMethod`. |
| `BINARY_MAP` | `{"Yes": 1, "No": 0}` | Lookup table for mapping binary string columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`) to integer values. |
| `REGISTERED_MODEL_NAME` | `"CustomerChurnEnsemble"` | Name under which the final ensemble is registered in MLflow Model Registry. |

---

## Pipeline Functions

### 1. `load_data(data_dir: Path) -> tuple[DataFrame, DataFrame, Series]`

**Purpose:** Reads the raw CSV files and performs minimal cleanup.

**Inputs:**
- `data_dir` — path to the directory containing `train.csv` and `test.csv`.

**Steps:**
1. Reads `train.csv` and `test.csv` via `pd.read_csv`.
2. Drops the `id` column from the training set (not a feature).
3. Pops `id` from the test set into a separate Series (needed later for the submission file).

**Returns:** `(train_df, test_df, test_id)` — the training DataFrame (with target), the test DataFrame (without id), and the test IDs.

---

### 2. `engineer_features(train, test) -> tuple[DataFrame, Series, DataFrame]`

**Purpose:** Applies identical feature engineering to both train and test sets, then splits the training data into features `X` and target `y`.

**Feature Engineering Steps:**

| Step | Operation | Details |
|---|---|---|
| Derived features | `AvgMonthlyCharge = TotalCharges / (tenure + 1)` | Estimates the customer's average monthly spend. The `+1` prevents division by zero for brand-new customers (tenure=0). |
| Derived features | `TotalServices = count of "Yes" in SERVICE_COLUMNS` | Integer count (0–8) of how many services the customer subscribes to. |
| One-hot encoding | `pd.get_dummies(columns=ONE_HOT_COLUMNS, drop_first=True)` | Converts 10 categorical columns into binary indicator columns. `drop_first=True` avoids the dummy variable trap (multicollinearity). |
| Binary encoding | Map `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | Converts Yes/No strings to 1/0 integers via `BINARY_MAP`. |
| Drop `gender` | `df.drop(columns=["gender"])` | Gender was determined to be non-predictive and is removed. |
| Encode target | `train["Churn"].map(BINARY_MAP)` | Converts the target from "Yes"/"No" strings to 1/0 integers. Only applied to the training set. |

**Returns:** `(X, y, test)` — feature matrix, target series, and the transformed test DataFrame.

**Critical Constraint:** The `preprocess()` function in `web_service.py` must mirror these exact transformations. Any changes here require synchronized updates there.

---

### 3. Hyperparameter Tuning

#### `tune_hyperparams(X, y, *, storage_path, n_trials) -> tuple[dict, dict]`

**Purpose:** Runs Optuna Bayesian hyperparameter search for both LightGBM and XGBoost, returning the best parameter dictionaries.

**Mechanism:**
- Creates (or resumes) two named Optuna studies backed by a SQLite database (`project/optuna_studies.db`), allowing incremental tuning across runs.
- Each trial evaluates a candidate parameter set using 5-fold stratified cross-validation with ROC-AUC as the scoring metric.
- Optimization direction is `maximize` (higher AUC = better).

#### `_objective_lgbm(trial, X, y) -> float`

**Search space for LightGBM:**

| Parameter | Range | Scale |
|---|---|---|
| `n_estimators` | 300 – 2000 | Linear |
| `learning_rate` | 0.01 – 0.3 | Log-uniform |
| `num_leaves` | 10 – 60 | Linear |
| `max_depth` | 3 – 10 | Linear |
| `min_child_samples` | 5 – 50 | Linear |
| `subsample` | 0.5 – 1.0 | Linear |
| `colsample_bytree` | 0.4 – 1.0 | Linear |
| `reg_alpha` | 0.001 – 10.0 | Log-uniform |
| `reg_lambda` | 0.001 – 10.0 | Log-uniform |

Creates an `LGBMClassifier` with `verbosity=-1` (silent) and evaluates it via `cross_val_score` using all CPU cores (`n_jobs=-1`).

#### `_objective_xgb(trial, X, y) -> float`

**Search space for XGBoost:**

| Parameter | Range | Scale |
|---|---|---|
| `n_estimators` | 300 – 2000 | Linear |
| `max_depth` | 3 – 10 | Linear |
| `learning_rate` | 0.01 – 0.3 | Log-uniform |
| `subsample` | 0.5 – 1.0 | Linear |
| `colsample_bytree` | 0.4 – 1.0 | Linear |
| `gamma` | 0.0 – 5.0 | Linear |
| `reg_alpha` | 0.001 – 10.0 | Log-uniform |
| `reg_lambda` | 0.001 – 10.0 | Log-uniform |

Additionally pins `random_state=42` and `eval_metric="auc"` on every trial.

**Returns:** `(best_lgbm_params, best_xgb_params)` — the best parameter dict for each model type, ready to pass into training.

---

### 4. `train_ensemble(X, y, test, best_lgbm_params, best_xgb_params) -> tuple[list, list, ndarray, list[str]]`

**Purpose:** Performs the core training loop — 5-fold stratified cross-validation with both LightGBM and XGBoost — while logging every detail to MLflow.

**MLflow Run Structure:**

```
Customer_Churn_Ensemble          (parent run)
├── LightGBM_Fold_1              (nested child run)
├── XGBoost_Fold_1               (nested child run)
├── LightGBM_Fold_2              (nested child run)
├── XGBoost_Fold_2               (nested child run)
│   ... (10 nested runs total)
├── LightGBM_Fold_5              (nested child run)
└── XGBoost_Fold_5               (nested child run)
```

**What gets logged to the parent run:**

| Category | Items |
|---|---|
| Parameters | Dataset dimensions, target positive rate, CV config, all hyperparameters (prefixed `lgbm_`/`xgb_`) |
| Tags | Task type, ensemble method, models used, target column |
| Text artifacts | `feature_list.json` — ordered list of feature column names |
| Aggregated metrics | Mean and std of AUC, mean accuracy, precision, recall, F1, log loss for each model type |
| Ensemble metrics | Prediction distribution stats (mean, std, median, min, max) |
| Figures | `cv_auc_comparison.png` — per-fold AUC line chart; `ensemble_prediction_distribution.png` — histogram of test predictions |
| CSV artifact | `cv_fold_metrics.csv` — tabular per-fold metrics |

**What gets logged to each nested child run:**

| Category | Items |
|---|---|
| Parameters | Model type, fold number, fold sizes, all hyperparameters |
| Metrics | `val_auc`, `val_accuracy`, `val_precision`, `val_recall`, `val_f1`, `val_log_loss` |
| Figures | Feature importance bar plot (top 15 features), confusion matrix heatmap |
| Model artifact | Native model (via `mlflow.lightgbm.log_model` or `mlflow.xgboost.log_model`) |

**Test Predictions — Averaging Strategy:**

For both model types, test predictions are accumulated across folds by averaging:

```python
lgb_test_preds += lgb_model.predict_proba(test.values)[:, 1] / N_SPLITS
```

Each fold contributes `1/N_SPLITS` of the final probability. The final ensemble prediction is a simple average of the LightGBM and XGBoost averaged predictions:

```python
final_test_preds = (lgb_test_preds + xgb_test_preds) / 2
```

**Returns:** `(lgb_models, xgb_models, final_test_preds, feature_list)` — the 5 LightGBM models, 5 XGBoost models, averaged test probabilities, and the ordered feature column names.

---

### 5. Model Registration

#### `ChurnEnsembleModel(mlflow.pyfunc.PythonModel)`

A custom MLflow `pyfunc` wrapper that bundles all 10 fold models (5 LightGBM + 5 XGBoost) into a single servable artifact.

**`load_context(self, context)`**
- Deserializes 5 LightGBM and 5 XGBoost models from joblib files referenced in `context.artifacts`.
- Loads the `feature_list.json` to know the expected column order.

**`predict(self, context, model_input, params=None) -> ndarray`**
- Reindexes the input DataFrame to match the training feature order, filling missing columns with 0.
- Computes the mean predicted probability across all 5 LightGBM models.
- Computes the mean predicted probability across all 5 XGBoost models.
- Returns the average of both means — the final ensemble churn probability per row.

#### `register_model(lgb_models, xgb_models, feature_list, best_params) -> None`

**Purpose:** Serializes all artifacts and registers the ensemble in MLflow Model Registry.

**Steps:**
1. Creates a temporary directory and serializes each of the 10 models to `.joblib` files.
2. Writes `feature_list.json` and `best_params.json` as JSON artifacts.
3. Calls `mlflow.pyfunc.log_model` with the `ChurnEnsembleModel` class and the artifact dictionary.
4. Registers the logged model under the name `"CustomerChurnEnsemble"` in the MLflow Model Registry.
5. Sets the alias `"champion"` on the newly registered version, so the web service always loads the latest best model.

---

### `main() -> None`

The entry point that chains all pipeline steps together.

**MLflow Connection Logic:**
- Uses a hardcoded tracking URI of `http://localhost:5001` (remote MLflow server, typically the Docker container).
- If that line were commented out and the environment variable `MLFLOW_TRACKING_URI` were used instead, it would fall back to a local SQLite backend at `project/mlflow.db`.

**Execution Order:**
1. `load_data()` — reads CSVs
2. `engineer_features()` — transforms data
3. `tune_hyperparams()` — Optuna HPO (incremental, persisted to SQLite)
4. `train_ensemble()` — 5-fold CV training + MLflow tracking
5. Write `submission.csv` and log it as an MLflow artifact
6. `register_model()` — persist and register the pyfunc ensemble
7. Print the MLflow run ID

---

## Dependencies

| Library | Role |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `lightgbm` | LightGBM gradient boosting classifier |
| `xgboost` | XGBoost gradient boosting classifier |
| `scikit-learn` | Metrics (AUC, F1, etc.), StratifiedKFold, cross_val_score |
| `optuna` | Bayesian hyperparameter optimization |
| `mlflow` | Experiment tracking, model logging, model registry |
| `joblib` | Model serialization for the pyfunc wrapper |
| `matplotlib`, `seaborn` | Visualization (feature importance, confusion matrices, AUC charts) |

---

## Output Artifacts

| Artifact | Location | Description |
|---|---|---|
| `submission.csv` | `project/submission.csv` | Test set predictions (id, churn probability) |
| Optuna DB | `project/optuna_studies.db` | Persistent HPO study results (SQLite) |
| MLflow experiment | `Customer_Churn_Prediction` | All runs, metrics, params, figures |
| Registered model | `CustomerChurnEnsemble@champion` | Servable pyfunc ensemble in MLflow Registry |
