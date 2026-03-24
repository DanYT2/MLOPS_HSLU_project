"""Training pipeline for the Customer Churn Prediction ensemble.

Extracts the notebook pipeline into composable functions so it can be
run as a script:  ``python train.py``

Steps:
  1. load_data        – read CSVs
  2. engineer_features – feature eng + encoding
  3. tune_hyperparams  – Optuna HPO
  4. train_ensemble    – 5-fold CV + MLflow tracking
  5. register_model    – pyfunc ensemble → MLflow model registry
"""

# Enables PEP 604 union syntax (X | Y) and PEP 585 generics (list[str])
# in type hints at runtime for Python <3.10 compatibility. Without this,
# annotations like `dict | None` would raise a TypeError at class definition time.
from __future__ import annotations

import json
import tempfile
from pathlib import Path

# joblib is used for serializing trained sklearn-compatible model objects
# to disk. It handles numpy arrays more efficiently than pickle.
import joblib
import matplotlib.pyplot as plt

# MLflow is the experiment tracking and model registry platform.
# We import submodules explicitly because mlflow uses lazy-loading —
# `mlflow.lightgbm` and `mlflow.xgboost` provide model-flavor-specific
# logging functions (log_model) that serialize models in their native format.
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np

# Optuna is a Bayesian hyperparameter optimization framework. It uses
# strategies like TPE (Tree-structured Parzen Estimator) to efficiently
# search the parameter space, unlike grid search which is exhaustive.
import optuna
import pandas as pd

# Seaborn wraps matplotlib and provides higher-level statistical plotting
# functions (e.g. barplot, heatmap) with better default aesthetics.
import seaborn as sns
from lightgbm import LGBMClassifier

# Importing specific metrics from sklearn rather than the whole module.
# Each metric measures a different aspect of classification performance:
#   - accuracy_score: fraction of correct predictions (can be misleading on imbalanced data)
#   - confusion_matrix: 2x2 matrix of TP/TN/FP/FN counts
#   - f1_score: harmonic mean of precision and recall (balances both)
#   - log_loss: cross-entropy loss; penalizes confident wrong predictions heavily
#   - precision_score: of all predicted positives, how many are truly positive
#   - recall_score: of all actual positives, how many did we catch
#   - roc_auc_score: area under the ROC curve; measures ranking quality across all thresholds
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

# StratifiedKFold: splits data into K folds while preserving the class
# distribution (ratio of churn/non-churn) in each fold. This is critical
# for imbalanced datasets to avoid folds with very few positive samples.
# cross_val_score: convenience function that runs K-fold CV and returns
# the scoring metric for each fold as an array.
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ── Constants ────────────────────────────────────────────────────

# Number of cross-validation folds. 5 is a standard choice that balances
# bias (too few folds → high bias) vs. variance (too many folds → high
# variance and slow training). Each fold uses 80% train / 20% validation.
N_SPLITS = 5

# Fixed seed for all random number generators to ensure reproducibility.
# Setting this across StratifiedKFold, LightGBM, and XGBoost means that
# re-running the script produces identical models and metrics.
RANDOM_STATE = 42

# The 8 telecom service columns that can take "Yes"/"No" values. Used to
# derive the TotalServices feature (count of subscribed services per customer).
# These represent add-on services a telecom customer can subscribe to.
SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Categorical columns to be one-hot encoded. One-hot encoding creates a
# binary (0/1) column for each unique value in the original column. For
# example, Contract with values ["Month-to-month", "One year", "Two year"]
# becomes two columns (drop_first=True removes one to avoid multicollinearity
# in the resulting feature matrix — the dropped category is implicitly
# represented when all indicator columns are 0).
ONE_HOT_COLUMNS = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod",
]

# Lookup dictionary for converting binary "Yes"/"No" strings to integers.
# Used for columns that have exactly two values (as opposed to the multi-
# category columns handled by one-hot encoding above).
BINARY_MAP = {"Yes": 1, "No": 0}

# The name used to register the final ensemble model in the MLflow Model
# Registry. The web service loads the model by this name + the "champion" alias.
REGISTERED_MODEL_NAME = "CustomerChurnEnsemble"


# ── 1. Data loading ──────────────────────────────────────────────


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Load the train and test CSVs and separate identifiers from features.

    The 'id' column is a row identifier with no predictive value, so it is
    removed from training data. For the test set, we preserve the IDs
    separately because they're needed for the final submission CSV.

    Args:
        data_dir: Path to the directory containing train.csv and test.csv.

    Returns:
        A tuple of (train_df, test_df, test_id) where:
        - train_df includes the Churn target column
        - test_df has no id column
        - test_id is a Series of test row identifiers
    """
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    # Drop id from train — it's just a row identifier, not a feature
    train = train.drop(columns=["id"])

    # Pop removes the column from the DataFrame and returns it as a Series,
    # so test_id captures the IDs while test_df no longer contains them
    test_id = test.pop("id")

    return train, test, test_id


# ── 2. Feature engineering ───────────────────────────────────────


def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Apply feature engineering identically to train and test.

    It is essential that the exact same transformations are applied to both
    datasets. If the test set is transformed differently (e.g. different
    column ordering, missing a derived feature), the model will produce
    nonsensical predictions.

    Returns (X, y, test) where X has no target column and the Churn
    target has been label-encoded to int.
    """

    def _feature_eng(df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing columns.

        Defined as an inner function to guarantee identical logic is applied
        to both the train and test DataFrames without code duplication.
        """
        # Copy to avoid modifying the original DataFrame in-place, which
        # could cause subtle bugs if the caller still references it
        df = df.copy()

        # AvgMonthlyCharge: estimates the customer's average spend per month.
        # We add 1 to tenure to avoid division-by-zero for new customers
        # (tenure=0 means they just joined). This is a "smoothed" average —
        # for a customer with tenure=0, it equals their TotalCharges.
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

        # TotalServices: counts how many of the 8 service columns have
        # value "Yes". This is a single numeric feature that captures
        # overall engagement — customers with more services may be less
        # likely to churn (they're more invested in the platform).
        # The expression (df[SERVICE_COLUMNS] == "Yes") produces a boolean
        # DataFrame, and .sum(axis=1) counts True values per row.
        df["TotalServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
        return df

    train = _feature_eng(train)
    test = _feature_eng(test)

    # One-hot encoding: converts categorical string columns into numeric
    # binary columns. For example, InternetService with values
    # ["DSL", "Fiber optic", "No"] becomes two columns:
    #   InternetService_Fiber optic (1 if fiber, else 0)
    #   InternetService_No          (1 if no internet, else 0)
    # The "DSL" category is dropped (drop_first=True) and is the implicit
    # baseline — when both indicator columns are 0, the customer has DSL.
    # dtype=int converts the boolean indicators to 0/1 integers.
    train = pd.get_dummies(
        train, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int
    )
    test = pd.get_dummies(
        test, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int
    )

    # Binary encoding: for columns with exactly two values ("Yes"/"No"),
    # a simple map to 1/0 is cleaner than one-hot encoding (which would
    # produce a single column anyway after drop_first). These four columns
    # are not included in ONE_HOT_COLUMNS because they're handled here.
    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        train[col] = train[col].map(BINARY_MAP)
        test[col] = test[col].map(BINARY_MAP)

    # Gender was determined to have negligible predictive power in EDA
    # (exploratory data analysis) and is dropped. errors="ignore" prevents
    # a KeyError if the column was already removed in a previous step.
    train = train.drop(columns=["gender"], errors="ignore")
    test = test.drop(columns=["gender"], errors="ignore")

    # Convert the target variable from "Yes"/"No" strings to 1/0 integers,
    # which is the format expected by LightGBM and XGBoost for binary
    # classification (they interpret 1 as the positive class).
    train["Churn"] = train["Churn"].map(BINARY_MAP)

    # Separate features (X) from target (y). The model should never see
    # the target column as an input feature — that would be data leakage.
    X = train.drop(columns=["Churn"])
    y = train["Churn"]

    return X, y, test


# ── 3. Hyperparameter tuning ────────────────────────────────────


def _objective_lgbm(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function for LightGBM hyperparameter search.

    Optuna calls this function once per trial. Each trial proposes a set of
    hyperparameter values from the defined search space using Bayesian
    optimization (TPE sampler by default). The function returns a single
    scalar (mean ROC-AUC) that Optuna maximizes.

    Args:
        trial: Optuna Trial object that suggests parameter values.
        X: Feature matrix.
        y: Target series.

    Returns:
        Mean cross-validated ROC-AUC score across K folds.
    """
    params = {
        # n_estimators: number of boosting rounds (trees). More trees can
        # capture more complex patterns but risk overfitting and increase
        # training time. Range 300-2000 covers conservative to aggressive.
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),

        # learning_rate: shrinks each tree's contribution. Lower values
        # need more trees but generalize better. Log scale is used because
        # the impact of changes is proportional (0.01→0.02 matters more
        # than 0.2→0.21). Typical sweet spot is 0.01-0.1.
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),

        # num_leaves: max number of leaves per tree. LightGBM grows trees
        # leaf-wise (unlike XGBoost's level-wise), so this directly controls
        # model complexity. More leaves = more expressive but risk overfitting.
        "num_leaves": trial.suggest_int("num_leaves", 10, 60),

        # max_depth: maximum depth of each tree. Limits how many sequential
        # decisions the tree can make. Prevents overly complex decision paths.
        "max_depth": trial.suggest_int("max_depth", 3, 10),

        # min_child_samples: minimum data points required in a leaf node.
        # Higher values prevent the model from learning patterns specific
        # to very small groups of samples (regularization against overfitting).
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),

        # subsample: fraction of training data used per tree (row sampling).
        # Values <1.0 introduce randomness that reduces overfitting — each
        # tree sees a different random subset of the data (similar to bagging).
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),

        # colsample_bytree: fraction of features used per tree (column sampling).
        # Forces trees to work with different feature subsets, reducing
        # correlation between trees and improving ensemble diversity.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # reg_alpha: L1 regularization (Lasso). Penalizes absolute magnitude
        # of leaf weights. Can drive some weights to exactly zero, effectively
        # performing feature selection within each tree. Log scale because
        # regularization strength has multiplicative effects.
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),

        # reg_lambda: L2 regularization (Ridge). Penalizes squared magnitude
        # of leaf weights. Smooths the model by preventing any single leaf
        # from having an extreme weight. Generally more stable than L1.
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }

    # verbosity=-1 suppresses LightGBM's training output (warnings, info).
    # random_state ensures reproducibility within the trial.
    model = LGBMClassifier(**params, verbosity=-1, random_state=RANDOM_STATE)

    # StratifiedKFold ensures each fold has approximately the same
    # proportion of churned/non-churned customers as the full dataset.
    # shuffle=True randomizes which samples go into which fold.
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # cross_val_score handles the full train/validate loop: for each fold,
    # it trains the model on the training portion and scores on the validation
    # portion. scoring="roc_auc" uses probability predictions, not hard labels.
    # n_jobs=-1 parallelizes across all available CPU cores.
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)

    # Return the mean AUC across all folds as the objective value.
    # Optuna will try to maximize this number across trials.
    return scores.mean()


def _objective_xgb(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    """Optuna objective function for XGBoost hyperparameter search.

    Same structure as _objective_lgbm but with XGBoost-specific parameters.
    XGBoost grows trees level-wise (breadth-first) unlike LightGBM's leaf-wise
    approach, so the parameter names and their effects differ slightly.

    Args:
        trial: Optuna Trial object that suggests parameter values.
        X: Feature matrix.
        y: Target series.

    Returns:
        Mean cross-validated ROC-AUC score across K folds.
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),

        # gamma: minimum loss reduction required to make a further partition
        # on a leaf node. Acts as a pruning threshold — a split only happens
        # if it reduces the loss by at least gamma. Higher values = more
        # conservative tree growth. This parameter is XGBoost-specific and
        # has no direct equivalent in LightGBM.
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),

        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),

        # These are fixed (not tuned) because they don't affect model quality
        # in a way that benefits from search — they're just configuration.
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",  # Internal evaluation metric used by XGBoost
    }

    # verbosity=0 is XGBoost's equivalent of LightGBM's verbosity=-1
    model = XGBClassifier(**params, verbosity=0)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def tune_hyperparams(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    storage_path: str = "sqlite:///project/optuna_studies.db",
    n_trials: int = 5,
) -> tuple[dict, dict]:
    """Run Bayesian hyperparameter optimization for both models.

    Uses Optuna's create_study with SQLite persistence, so that studies
    accumulate trials across multiple script runs (load_if_exists=True).
    This means running the script 3 times with n_trials=5 yields 15 total
    trials, progressively refining the search.

    Args:
        X: Feature matrix for cross-validation.
        y: Target series for cross-validation.
        storage_path: SQLite URI for persisting Optuna study state.
        n_trials: Number of new trials to run per study in this invocation.

    Returns:
        A tuple of (best_lgbm_params, best_xgb_params) dictionaries containing
        the optimal hyperparameters found across all trials (not just this run).
    """
    # Reduce Optuna's console output — only show warnings, not per-trial info
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # create_study initializes a new study or loads an existing one from the
    # SQLite database. direction="maximize" tells Optuna that higher AUC = better.
    # The study_name acts as a unique key in the database.
    study_lgbm = optuna.create_study(
        direction="maximize",
        study_name="lgbm_tuning",
        storage=storage_path,
        load_if_exists=True,  # Resume from previous runs instead of starting fresh
    )

    # optimize() runs n_trials new trials. The lambda wraps _objective_lgbm
    # to pass X and y alongside the trial object (Optuna only passes the trial).
    study_lgbm.optimize(
        lambda trial: _objective_lgbm(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(
        f"LightGBM — best trial #{study_lgbm.best_trial.number}  "
        f"AUC={study_lgbm.best_value:.6f}"
    )

    study_xgb = optuna.create_study(
        direction="maximize",
        study_name="xgb_tuning",
        storage=storage_path,
        load_if_exists=True,
    )
    study_xgb.optimize(
        lambda trial: _objective_xgb(trial, X, y),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(
        f"XGBoost  — best trial #{study_xgb.best_trial.number}  "
        f"AUC={study_xgb.best_value:.6f}"
    )

    # .best_params returns only the Optuna-suggested params (no fixed ones).
    # For XGBoost, we must add back the fixed params that weren't part of the
    # search space but are required for model construction.
    best_lgbm_params = study_lgbm.best_params
    best_xgb_params = {
        **study_xgb.best_params,       # Unpack the tuned parameters
        "random_state": RANDOM_STATE,   # Add fixed params back
        "eval_metric": "auc",
    }

    return best_lgbm_params, best_xgb_params


# ── 4. Training ──────────────────────────────────────────────────


def train_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    best_lgbm_params: dict,
    best_xgb_params: dict,
) -> tuple[list, list, np.ndarray, list[str], float]:
    """Train a LightGBM + XGBoost ensemble using stratified K-fold CV.

    For each of the K folds, both a LightGBM and an XGBoost model are trained
    on the training portion and evaluated on the validation portion. Test set
    predictions are accumulated across folds by averaging (each fold contributes
    1/K of the final prediction). The final ensemble prediction is the simple
    average of the LightGBM and XGBoost averaged predictions.

    Everything is tracked in MLflow: one parent run for the overall ensemble,
    with nested child runs for each (model, fold) combination — 10 total.

    Args:
        X: Training feature matrix.
        y: Training target series.
        test: Test feature matrix (predictions are generated for this).
        best_lgbm_params: Optimized LightGBM hyperparameters from Optuna.
        best_xgb_params: Optimized XGBoost hyperparameters from Optuna.

    Returns:
        A tuple of (lgb_models, xgb_models, final_test_preds, feature_list, ensemble_mean_cv_auc):
        - lgb_models: list of 5 trained LightGBM models (one per fold)
        - xgb_models: list of 5 trained XGBoost models (one per fold)
        - final_test_preds: numpy array of averaged ensemble churn probabilities
        - feature_list: ordered list of feature column names
        - ensemble_mean_cv_auc: combined mean CV AUC of the ensemble (used for champion comparison)
    """
    # Create the stratified K-fold splitter. "Stratified" means each fold
    # preserves the percentage of samples for each class (churn vs no churn).
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    # Accumulators for test predictions. Initialized to zeros and each fold
    # adds its contribution (prediction / N_SPLITS). After all folds, this
    # contains the average prediction across all fold models.
    lgb_test_preds = np.zeros(len(test))
    xgb_test_preds = np.zeros(len(test))

    # Lists to collect the trained model objects (needed for registration)
    lgb_models: list = []
    xgb_models: list = []

    # Per-fold validation metric accumulators. After the loop, these are
    # aggregated (mean, std) and logged to the MLflow parent run to give
    # an overall picture of model performance and its variance across folds.
    lgb_val_aucs: list[float] = []
    xgb_val_aucs: list[float] = []
    lgb_val_accs: list[float] = []
    xgb_val_accs: list[float] = []
    lgb_val_f1s: list[float] = []
    xgb_val_f1s: list[float] = []
    lgb_val_precisions: list[float] = []
    xgb_val_precisions: list[float] = []
    lgb_val_recalls: list[float] = []
    xgb_val_recalls: list[float] = []
    lgb_val_logloss: list[float] = []
    xgb_val_logloss: list[float] = []

    # Defensively end any active MLflow run (e.g. from a previous failed execution)
    # before starting a fresh parent run for this training session.
    mlflow.end_run()
    mlflow.start_run(run_name="Customer_Churn_Ensemble")

    # Log dataset metadata as MLflow parameters. Parameters are key-value pairs
    # that describe the configuration of the run (not metrics that change over time).
    mlflow.log_param("dataset_train_rows", X.shape[0])
    mlflow.log_param("dataset_train_cols", X.shape[1])
    mlflow.log_param("dataset_test_rows", len(test))
    # Target positive rate: what fraction of customers actually churned. This
    # tells us about class imbalance — a rate of 0.27 means 27% churned.
    mlflow.log_param("target_positive_rate", round(float(y.mean()), 4))
    mlflow.log_param("cv_n_splits", N_SPLITS)
    mlflow.log_param("cv_random_state", RANDOM_STATE)

    # Tags are free-form metadata for organizing and searching runs in the
    # MLflow UI. Unlike params, they're not expected to affect the model.
    mlflow.set_tag("task_type", "binary_classification")
    mlflow.set_tag("ensemble_method", "simple_average")
    mlflow.set_tag("models_used", "LightGBM, XGBoost")
    mlflow.set_tag("target_column", "Churn")

    # Log every hyperparameter with a model-type prefix so they're easily
    # distinguishable in the MLflow UI (e.g. "lgbm_learning_rate" vs "xgb_learning_rate")
    for k, v in best_lgbm_params.items():
        mlflow.log_param(f"lgbm_{k}", v)
    for k, v in best_xgb_params.items():
        mlflow.log_param(f"xgb_{k}", v)

    # Save the feature column names as a JSON artifact. The pyfunc model
    # uses this at inference time to reindex input DataFrames to the correct
    # column order (important because tree models are column-order-dependent).
    mlflow.log_text(json.dumps(list(X.columns), indent=2), "feature_list.json")

    # Main cross-validation loop. skf.split(X, y) yields (train_indices, val_indices)
    # for each fold. The stratification ensures each fold's y distribution
    # matches the overall dataset's class distribution.
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1} / {N_SPLITS}")
        print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        print(f"{'='*60}")

        # Split the data using integer-location indexing (.iloc) with the
        # indices provided by StratifiedKFold
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # ── LightGBM ────────────────────────────────────────────
        # nested=True creates a child run under the parent "Customer_Churn_Ensemble"
        # run. This organizes the MLflow UI hierarchically — you can expand the
        # parent run to see all fold-level runs underneath it.
        with mlflow.start_run(run_name=f"LightGBM_Fold_{fold+1}", nested=True):
            # Log fold-specific metadata for this child run
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("fold", fold + 1)
            mlflow.log_param("train_fold_size", len(train_idx))
            mlflow.log_param("val_fold_size", len(val_idx))
            mlflow.log_params({f"lgbm_{k}": v for k, v in best_lgbm_params.items()})

            # Instantiate a fresh LightGBM model with the Optuna-optimized params.
            # objective="binary" configures the loss function for binary classification
            # (binary cross-entropy / log loss).
            lgb_model = LGBMClassifier(
                **best_lgbm_params,
                objective="binary",
                random_state=RANDOM_STATE,
                verbosity=-1,
            )
            lgb_model.fit(X_train_fold, y_train_fold)

            # predict_proba returns a 2D array of shape (n_samples, 2) where
            # column 0 = P(no churn) and column 1 = P(churn). We take [:, 1]
            # because AUC and log_loss need the positive class probability.
            lgb_val_proba = lgb_model.predict_proba(X_val_fold)[:, 1]
            # predict returns hard class labels (0 or 1) using the default
            # 0.5 threshold. Needed for accuracy, precision, recall, F1.
            lgb_val_pred = lgb_model.predict(X_val_fold)

            # Compute all validation metrics for this fold:
            # ROC-AUC: measures ranking quality — how well the model separates
            # the two classes at ALL thresholds (not just 0.5). Score of 1.0
            # means perfect separation; 0.5 means random guessing.
            fold_lgb_auc = roc_auc_score(y_val_fold, lgb_val_proba)
            fold_lgb_acc = accuracy_score(y_val_fold, lgb_val_pred)
            # Precision: of customers predicted as churning, what % actually churned.
            # High precision = few false alarms.
            fold_lgb_prec = precision_score(y_val_fold, lgb_val_pred)
            # Recall: of customers who actually churned, what % did we catch.
            # High recall = few missed churners.
            fold_lgb_rec = recall_score(y_val_fold, lgb_val_pred)
            # F1: harmonic mean of precision and recall. A balanced metric when
            # both false positives and false negatives matter.
            fold_lgb_f1 = f1_score(y_val_fold, lgb_val_pred)
            # Log loss: measures the quality of probability estimates (not just
            # rankings). Heavily penalizes confident wrong predictions.
            fold_lgb_ll = log_loss(y_val_fold, lgb_val_proba)

            # Log each metric to the nested MLflow child run
            mlflow.log_metric("val_auc", fold_lgb_auc)
            mlflow.log_metric("val_accuracy", fold_lgb_acc)
            mlflow.log_metric("val_precision", fold_lgb_prec)
            mlflow.log_metric("val_recall", fold_lgb_rec)
            mlflow.log_metric("val_f1", fold_lgb_f1)
            mlflow.log_metric("val_log_loss", fold_lgb_ll)

            # Feature importance: LightGBM tracks how many times each feature
            # was used in a split (split-based importance). Higher importance
            # means the feature was more useful for making predictions.
            # We create a sorted bar plot of the top 15 most important features.
            feat_imp = pd.DataFrame(
                {"feature": X.columns, "importance": lgb_model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feat_imp.head(15), x="importance", y="feature", ax=ax)
            ax.set_title(f"LightGBM Feature Importance — Fold {fold+1}")
            plt.tight_layout()
            mlflow.log_figure(fig, f"lgbm_feature_importance_fold_{fold+1}.png")
            plt.close(fig)  # Close to free memory (important in loops)

            # Confusion matrix: a 2x2 table showing:
            #   [[True Negatives, False Positives],
            #    [False Negatives, True Positives]]
            # Visualized as a heatmap where darker cells = higher counts.
            cm = confusion_matrix(y_val_fold, lgb_val_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
            )
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"LightGBM Confusion Matrix — Fold {fold+1}")
            plt.tight_layout()
            mlflow.log_figure(fig, f"lgbm_confusion_matrix_fold_{fold+1}.png")
            plt.close(fig)

            # Log the model in LightGBM's native format. This preserves the
            # full model structure and can be loaded back with mlflow.lightgbm.load_model.
            mlflow.lightgbm.log_model(lgb_model, artifact_path=f"lgbm_model_fold_{fold+1}")

            # Accumulate metrics for later aggregation on the parent run
            lgb_val_aucs.append(fold_lgb_auc)
            lgb_val_accs.append(fold_lgb_acc)
            lgb_val_f1s.append(fold_lgb_f1)
            lgb_val_precisions.append(fold_lgb_prec)
            lgb_val_recalls.append(fold_lgb_rec)
            lgb_val_logloss.append(fold_lgb_ll)
            print(f"  LightGBM — AUC: {fold_lgb_auc:.6f} | Acc: {fold_lgb_acc:.6f}")

        # Accumulate this fold's test predictions. Each fold's model predicts
        # on the FULL test set, and we average by dividing by N_SPLITS. This
        # is a form of model averaging that reduces variance in predictions —
        # each fold's model was trained on slightly different data, so their
        # predictions capture different aspects of the training distribution.
        # .values converts the DataFrame to a raw numpy array (required by
        # predict_proba when columns might not match exactly).
        lgb_test_preds += lgb_model.predict_proba(test.values)[:, 1] / N_SPLITS
        lgb_models.append(lgb_model)

        # ── XGBoost ─────────────────────────────────────────────
        # Same structure as LightGBM above: train → evaluate → log → accumulate
        with mlflow.start_run(run_name=f"XGBoost_Fold_{fold+1}", nested=True):
            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("fold", fold + 1)
            mlflow.log_param("train_fold_size", len(train_idx))
            mlflow.log_param("val_fold_size", len(val_idx))
            mlflow.log_params({f"xgb_{k}": v for k, v in best_xgb_params.items()})

            xgb_model = XGBClassifier(**best_xgb_params)
            xgb_model.fit(X_train_fold, y_train_fold)

            xgb_val_proba = xgb_model.predict_proba(X_val_fold)[:, 1]
            xgb_val_pred = xgb_model.predict(X_val_fold)

            fold_xgb_auc = roc_auc_score(y_val_fold, xgb_val_proba)
            fold_xgb_acc = accuracy_score(y_val_fold, xgb_val_pred)
            fold_xgb_prec = precision_score(y_val_fold, xgb_val_pred)
            fold_xgb_rec = recall_score(y_val_fold, xgb_val_pred)
            fold_xgb_f1 = f1_score(y_val_fold, xgb_val_pred)
            fold_xgb_ll = log_loss(y_val_fold, xgb_val_proba)

            mlflow.log_metric("val_auc", fold_xgb_auc)
            mlflow.log_metric("val_accuracy", fold_xgb_acc)
            mlflow.log_metric("val_precision", fold_xgb_prec)
            mlflow.log_metric("val_recall", fold_xgb_rec)
            mlflow.log_metric("val_f1", fold_xgb_f1)
            mlflow.log_metric("val_log_loss", fold_xgb_ll)

            feat_imp_xgb = pd.DataFrame(
                {"feature": X.columns, "importance": xgb_model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feat_imp_xgb.head(15), x="importance", y="feature", ax=ax)
            ax.set_title(f"XGBoost Feature Importance — Fold {fold+1}")
            plt.tight_layout()
            mlflow.log_figure(fig, f"xgb_feature_importance_fold_{fold+1}.png")
            plt.close(fig)

            cm_xgb = confusion_matrix(y_val_fold, xgb_val_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(
                cm_xgb, annot=True, fmt="d", cmap="Oranges", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"],
            )
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"XGBoost Confusion Matrix — Fold {fold+1}")
            plt.tight_layout()
            mlflow.log_figure(fig, f"xgb_confusion_matrix_fold_{fold+1}.png")
            plt.close(fig)

            mlflow.xgboost.log_model(xgb_model, name=f"xgb_model_fold_{fold+1}")

            xgb_val_aucs.append(fold_xgb_auc)
            xgb_val_accs.append(fold_xgb_acc)
            xgb_val_f1s.append(fold_xgb_f1)
            xgb_val_precisions.append(fold_xgb_prec)
            xgb_val_recalls.append(fold_xgb_rec)
            xgb_val_logloss.append(fold_xgb_ll)
            print(f"  XGBoost  — AUC: {fold_xgb_auc:.6f} | Acc: {fold_xgb_acc:.6f}")

        xgb_test_preds += xgb_model.predict_proba(test.values)[:, 1] / N_SPLITS
        xgb_models.append(xgb_model)

    # ── Aggregate CV metrics on parent run ───────────────────────
    # Now that all folds are complete, compute the mean and standard deviation
    # of each metric across folds. The mean tells us overall performance;
    # the std tells us how stable the model is across different data splits
    # (low std = consistent performance, high std = model is sensitive to
    # which data it trains on).
    mlflow.log_metric("lgbm_mean_cv_auc", np.mean(lgb_val_aucs))
    mlflow.log_metric("lgbm_std_cv_auc", np.std(lgb_val_aucs))
    mlflow.log_metric("lgbm_mean_cv_accuracy", np.mean(lgb_val_accs))
    mlflow.log_metric("lgbm_mean_cv_precision", np.mean(lgb_val_precisions))
    mlflow.log_metric("lgbm_mean_cv_recall", np.mean(lgb_val_recalls))
    mlflow.log_metric("lgbm_mean_cv_f1", np.mean(lgb_val_f1s))
    mlflow.log_metric("lgbm_mean_cv_log_loss", np.mean(lgb_val_logloss))

    mlflow.log_metric("xgb_mean_cv_auc", np.mean(xgb_val_aucs))
    mlflow.log_metric("xgb_std_cv_auc", np.std(xgb_val_aucs))
    mlflow.log_metric("xgb_mean_cv_accuracy", np.mean(xgb_val_accs))
    mlflow.log_metric("xgb_mean_cv_precision", np.mean(xgb_val_precisions))
    mlflow.log_metric("xgb_mean_cv_recall", np.mean(xgb_val_recalls))
    mlflow.log_metric("xgb_mean_cv_f1", np.mean(xgb_val_f1s))
    mlflow.log_metric("xgb_mean_cv_log_loss", np.mean(xgb_val_logloss))

    # Visualization: per-fold AUC comparison chart. This lets you quickly
    # see if one model consistently outperforms the other, and whether
    # any particular fold was an outlier. Dashed horizontal lines show the
    # mean AUC for each model type across all folds.
    fig, ax = plt.subplots(figsize=(10, 6))
    fold_numbers = list(range(1, N_SPLITS + 1))
    ax.plot(fold_numbers, lgb_val_aucs, marker="o", label="LightGBM", linewidth=2)
    ax.plot(fold_numbers, xgb_val_aucs, marker="s", label="XGBoost", linewidth=2)
    ax.axhline(np.mean(lgb_val_aucs), ls="--", color="tab:blue", alpha=0.5,
               label=f"LGBM Mean: {np.mean(lgb_val_aucs):.6f}")
    ax.axhline(np.mean(xgb_val_aucs), ls="--", color="tab:orange", alpha=0.5,
               label=f"XGB Mean: {np.mean(xgb_val_aucs):.6f}")
    ax.set_xlabel("Fold"); ax.set_ylabel("Validation AUC")
    ax.set_title("Per-Fold Validation AUC Comparison")
    ax.set_xticks(fold_numbers); ax.legend()
    plt.tight_layout()
    mlflow.log_figure(fig, "cv_auc_comparison.png")
    plt.close(fig)

    # Save a CSV of per-fold metrics as an MLflow text artifact for easy
    # tabular review without needing to parse individual child runs
    metrics_df = pd.DataFrame({
        "fold": fold_numbers,
        "lgbm_auc": lgb_val_aucs, "xgb_auc": xgb_val_aucs,
        "lgbm_acc": lgb_val_accs, "xgb_acc": xgb_val_accs,
        "lgbm_f1": lgb_val_f1s, "xgb_f1": xgb_val_f1s,
    })
    mlflow.log_text(metrics_df.to_csv(index=False), "cv_fold_metrics.csv")

    # Compute the ensemble's combined mean CV AUC as the average of both model
    # types' mean AUCs. This single number represents the overall quality of the
    # ensemble and is used by register_model() to decide whether this run should
    # replace the current champion in the Model Registry.
    ensemble_mean_cv_auc = (np.mean(lgb_val_aucs) + np.mean(xgb_val_aucs)) / 2
    mlflow.log_metric("ensemble_mean_cv_auc", ensemble_mean_cv_auc)

    # Final ensemble prediction: average the LightGBM and XGBoost fold-averaged
    # predictions. This is a "simple average" ensemble — each model type gets
    # equal weight. Ensembling two different algorithms (LightGBM is leaf-wise,
    # XGBoost is level-wise) tends to produce better generalization because
    # they make different types of errors that partially cancel out.
    final_test_preds = (lgb_test_preds + xgb_test_preds) / 2

    # Log summary statistics of the ensemble's test predictions. These help
    # detect anomalies: e.g. if pred_mean is very high, the model might be
    # systematically over-predicting churn; if pred_std is very low, the model
    # might not be discriminating well between customers.
    mlflow.log_metric("ensemble_pred_mean", float(np.mean(final_test_preds)))
    mlflow.log_metric("ensemble_pred_std", float(np.std(final_test_preds)))
    mlflow.log_metric("ensemble_pred_median", float(np.median(final_test_preds)))
    mlflow.log_metric("ensemble_pred_min", float(np.min(final_test_preds)))
    mlflow.log_metric("ensemble_pred_max", float(np.max(final_test_preds)))

    # Histogram of predicted churn probabilities. A well-calibrated model
    # should show a bimodal distribution (clusters near 0 and 1), not a
    # unimodal blob near 0.5. The red dashed line marks the decision boundary.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(final_test_preds, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("Predicted Churn Probability"); ax.set_ylabel("Count")
    ax.set_title("Distribution of Ensemble Test Predictions")
    ax.axvline(0.5, color="red", ls="--", linewidth=1.5, label="Decision Boundary (0.5)")
    ax.legend(); plt.tight_layout()
    mlflow.log_figure(fig, "ensemble_prediction_distribution.png")
    plt.close(fig)

    print(f"\n{'='*60}")
    print("  CV Training Completed")
    print(f"{'='*60}")
    print(f"  LightGBM — Mean AUC: {np.mean(lgb_val_aucs):.6f} "
          f"(+/- {np.std(lgb_val_aucs):.6f})")
    print(f"  XGBoost  — Mean AUC: {np.mean(xgb_val_aucs):.6f} "
          f"(+/- {np.std(xgb_val_aucs):.6f})")
    print(f"{'='*60}")

    feature_list = list(X.columns)
    return lgb_models, xgb_models, final_test_preds, feature_list, ensemble_mean_cv_auc


# ── 5. MLflow pyfunc wrapper & model registration ───────────────


class ChurnEnsembleModel(mlflow.pyfunc.PythonModel):
    """Custom MLflow pyfunc wrapper that bundles all 10 fold models into one artifact.

    MLflow's pyfunc (Python function) flavor is a generic model interface
    that lets you define custom predict logic. This is necessary here because
    our "model" is actually 10 separate models (5 LightGBM + 5 XGBoost)
    whose predictions need to be averaged. None of the built-in MLflow
    flavors (lightgbm, xgboost, sklearn) support multi-model ensembles
    natively, so we use pyfunc as a flexible wrapper.

    The class contract requires implementing:
    - load_context(): deserialize artifacts when the model is loaded
    - predict(): generate predictions from input data
    """

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Deserialize all fold models and metadata when the model is loaded.

        MLflow calls this automatically when you do mlflow.pyfunc.load_model().
        The context.artifacts dict maps artifact keys to their file paths on disk.

        Args:
            context: MLflow context containing the artifact file paths.
        """
        # Load all 5 LightGBM models from their joblib-serialized files.
        # joblib.load reconstructs the full LGBMClassifier object including
        # its trained tree structure, feature importances, and predict methods.
        self.lgb_models = [
            joblib.load(context.artifacts[f"lgbm_fold_{i}"]) for i in range(1, 6)
        ]
        self.xgb_models = [
            joblib.load(context.artifacts[f"xgb_fold_{i}"]) for i in range(1, 6)
        ]

        # Load the ordered list of feature column names. This is critical
        # for aligning incoming prediction data to the exact column order
        # the models were trained on (tree models are sensitive to column order).
        with open(context.artifacts["feature_list"]) as f:
            self.feature_list: list[str] = json.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        """Generate churn probability predictions by averaging all fold models.

        Args:
            context: MLflow context (not used here but required by the interface).
            model_input: DataFrame of preprocessed customer features.
            params: Optional prediction parameters (not used).

        Returns:
            1D numpy array of churn probabilities (one per input row).
        """
        # Reindex ensures the input DataFrame has exactly the same columns
        # in exactly the same order as the training data. fill_value=0 handles
        # missing columns (e.g. a one-hot category not present in the input
        # data). Extra columns in the input are silently dropped.
        df = model_input.reindex(columns=self.feature_list, fill_value=0)

        # Convert to raw numpy array for predict_proba (faster than DataFrame)
        arr = df.values

        # Average the positive-class probabilities across all 5 LightGBM folds.
        # np.mean(..., axis=0) averages element-wise across the list of arrays,
        # producing one probability per input row.
        lgb_probs = np.mean(
            [m.predict_proba(arr)[:, 1] for m in self.lgb_models], axis=0
        )
        xgb_probs = np.mean(
            [m.predict_proba(arr)[:, 1] for m in self.xgb_models], axis=0
        )

        # Final ensemble: simple average of LightGBM and XGBoost averaged probs
        return (lgb_probs + xgb_probs) / 2


def _get_champion_auc() -> float | None:
    """Query the current champion model's ensemble_mean_cv_auc from MLflow.

    Looks up the "champion" alias in the Model Registry, finds the MLflow run
    that produced it, and reads its ensemble_mean_cv_auc metric. This is the
    metric the new run must beat to be promoted.

    Returns:
        The champion's ensemble_mean_cv_auc, or None if:
        - No model is registered yet (first run)
        - The champion alias doesn't exist
        - The champion's run has no ensemble_mean_cv_auc metric (legacy run
          from before this comparison logic was added)
    """
    try:
        client = mlflow.tracking.MlflowClient()

        # get_model_version_by_alias looks up which version number the
        # "champion" alias currently points to, and returns its metadata
        # including the run_id that produced it.
        champion_version = client.get_model_version_by_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
        )

        # Fetch the full run data for the champion's producing run.
        # run.data.metrics is a dict of {metric_name: latest_value}.
        champion_run = client.get_run(champion_version.run_id)
        champion_auc = champion_run.data.metrics.get("ensemble_mean_cv_auc")

        return champion_auc

    except Exception:
        # Any failure (model not registered, alias not set, run deleted, etc.)
        # means there's no valid champion to compare against.
        return None


def register_model(
    lgb_models: list,
    xgb_models: list,
    feature_list: list[str],
    best_params: dict,
    new_auc: float,
) -> None:
    """Serialize all fold models, register in MLflow, and conditionally promote.

    The model is always registered as a new version in the Model Registry
    (for auditability — every training run is preserved). However, the
    "champion" alias is only reassigned if the new model's ensemble_mean_cv_auc
    exceeds the current champion's. This prevents a regression from reaching
    production.

    If there is no existing champion (first run, or the champion alias was
    removed), the new model is promoted unconditionally.

    Args:
        lgb_models: List of 5 trained LightGBM models.
        xgb_models: List of 5 trained XGBoost models.
        feature_list: Ordered list of feature column names.
        best_params: Dictionary containing the best parameters for both model types.
        new_auc: The new run's ensemble_mean_cv_auc to compare against the champion.
    """
    # ── Query the current champion's AUC before registering ──────
    # We do this first so the comparison is against the champion that existed
    # before this run, not against our own newly registered version.
    champion_auc = _get_champion_auc()

    # Use a temporary directory that's automatically cleaned up when done.
    # We need a staging area to write files that MLflow will then copy into
    # its artifact store.
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifacts: dict[str, str] = {}  # Maps artifact keys → file paths

        # Serialize each LightGBM fold model with joblib. Joblib is preferred
        # over pickle for sklearn-compatible models because it efficiently
        # handles numpy arrays (uses memory-mapped files for large arrays).
        for i, m in enumerate(lgb_models, start=1):
            p = tmp_path / f"lgbm_fold_{i}.joblib"
            joblib.dump(m, p)
            artifacts[f"lgbm_fold_{i}"] = str(p)

        for i, m in enumerate(xgb_models, start=1):
            p = tmp_path / f"xgb_fold_{i}.joblib"
            joblib.dump(m, p)
            artifacts[f"xgb_fold_{i}"] = str(p)

        # Write the feature list as JSON — needed by ChurnEnsembleModel.load_context()
        # to reindex incoming DataFrames to the correct column order
        fl_path = tmp_path / "feature_list.json"
        fl_path.write_text(json.dumps(feature_list, indent=2))
        artifacts["feature_list"] = str(fl_path)

        # Also save the best hyperparameters for reproducibility documentation
        bp_path = tmp_path / "best_params.json"
        bp_path.write_text(json.dumps(best_params, indent=2))
        artifacts["best_params"] = str(bp_path)

        # log_model packages the ChurnEnsembleModel class + all artifact files
        # into a single MLflow model artifact. When loaded, MLflow instantiates
        # ChurnEnsembleModel and calls load_context() with paths to the artifacts.
        model_info = mlflow.pyfunc.log_model(
            name="ensemble_model",
            python_model=ChurnEnsembleModel(),
            artifacts=artifacts,
        )

    # register_model promotes the logged model artifact into the Model Registry.
    # This always creates a new version (version numbers auto-increment) regardless
    # of whether the model becomes champion — every run is preserved for auditability.
    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print(f"\nRegistered {REGISTERED_MODEL_NAME} v{registered.version}")

    # ── Champion promotion decision ──────────────────────────────
    # Compare the new model's AUC against the existing champion. The alias
    # is only moved if the new model is strictly better, preventing performance
    # regressions from reaching production. Three cases:
    #
    #   1. No existing champion (first run or legacy) → promote unconditionally
    #   2. New AUC > champion AUC                     → promote (improvement)
    #   3. New AUC <= champion AUC                     → keep existing champion
    client = mlflow.tracking.MlflowClient()

    if champion_auc is None:
        # Case 1: no existing champion to compare against. This happens on
        # the very first training run, or if the champion was from a legacy
        # run that didn't log ensemble_mean_cv_auc. Promote unconditionally.
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=registered.version,
        )
        print(f"  → No existing champion found. "
              f"Promoted v{registered.version} as champion (AUC={new_auc:.6f})")

    elif new_auc > champion_auc:
        # Case 2: the new model outperforms the current champion. Move the
        # "champion" alias to point to the new version. The web service will
        # pick up this new model on its next restart.
        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=registered.version,
        )
        print(f"  → New model BEATS champion! "
              f"AUC: {new_auc:.6f} > {champion_auc:.6f}")
        print(f"  → Promoted v{registered.version} as new champion")

    else:
        # Case 3: the new model is equal or worse. Keep the existing champion.
        # The new version is still in the registry for inspection, but it
        # won't be served by the web service.
        print(f"  → New model did NOT beat champion. "
              f"AUC: {new_auc:.6f} <= {champion_auc:.6f}")
        print(f"  → Champion unchanged. v{registered.version} registered "
              f"but NOT promoted.")


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    """Entry point: orchestrates the full training pipeline end-to-end.

    Resolves paths, configures the MLflow backend, and calls each pipeline
    step in sequence. The MLflow tracking URI determines where experiments,
    metrics, and model artifacts are stored.
    """
    # Resolve the project directory relative to this script's location.
    # Path(__file__) is the path to train.py; .resolve() makes it absolute;
    # .parent gives the containing directory (project/).
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "dataset"

    # MLflow tracking URI: points to the MLflow server that stores all
    # experiment data. The remote server (Docker container) runs on port 5001.
    # When commented out and using the env var, it falls back to a local
    # SQLite database for standalone/offline usage.
    # tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    tracking_uri = "http://localhost:5001"
    if tracking_uri:
        # set_tracking_uri: where MLflow logs metrics, params, and artifacts
        # set_registry_uri: where MLflow stores the Model Registry (names, versions, aliases)
        # Both point to the same server in this setup.
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        print(f"Using remote MLflow tracking URI: {tracking_uri}")
    else:
        # Fallback: use a local SQLite file as the MLflow backend.
        # This works without a running MLflow server but has limitations
        # (no concurrent access, no web UI unless you start mlflow ui separately).
        (project_dir / "mlruns").mkdir(parents=True, exist_ok=True)
        db_path = project_dir / "mlflow.db"
        local_tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(local_tracking_uri)
        mlflow.set_registry_uri(local_tracking_uri)
        print(f"Using local MLflow tracking URI: {local_tracking_uri}")

    # set_experiment creates the experiment if it doesn't exist, or sets it
    # as the active experiment for subsequent runs. All runs created after
    # this call will be filed under "Customer_Churn_Prediction".
    mlflow.set_experiment("Customer_Churn_Prediction")

    # ── Pipeline execution ───────────────────────────────────────
    print("Loading data …")
    train_df, test_df, test_id = load_data(data_dir)

    print("Engineering features …")
    X, y, test = engineer_features(train_df, test_df)

    print("Tuning hyperparameters …")
    best_lgbm_params, best_xgb_params = tune_hyperparams(
        X, y, storage_path=f"sqlite:///{project_dir / 'optuna_studies.db'}"
    )

    print("Training ensemble …")
    lgb_models, xgb_models, final_test_preds, feature_list, ensemble_auc = train_ensemble(
        X, y, test, best_lgbm_params, best_xgb_params
    )

    # Build the submission CSV with test IDs and the ensemble's predicted
    # churn probabilities. This file can be submitted to a competition
    # platform or used for downstream analysis.
    submission_df = pd.DataFrame({"id": test_id, "Churn": final_test_preds})
    submission_path = project_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    # Also log the submission as an MLflow artifact so it's versioned
    # alongside the model and metrics for full reproducibility
    mlflow.log_artifact(str(submission_path))
    print(f"Submission saved to {submission_path}")

    # Package both model types' best params into a single dict for
    # the model registry artifact (documentation/reproducibility)
    best_params = {"lgbm": best_lgbm_params, "xgb": best_xgb_params}
    register_model(lgb_models, xgb_models, feature_list, best_params, ensemble_auc)

    # Capture the run ID before ending the run (after end_run, active_run is None)
    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Run ID : {run_id}")
    print("  mlflow ui --port 5000")
    print(f"{'='*60}")


# Standard Python idiom: only run main() when the script is executed directly
# (python train.py), not when imported as a module (import train).
if __name__ == "__main__":
    main()
