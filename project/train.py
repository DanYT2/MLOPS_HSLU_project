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

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ── Constants ────────────────────────────────────────────────────

N_SPLITS = 5
RANDOM_STATE = 42

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

BINARY_MAP = {"Yes": 1, "No": 0}

REGISTERED_MODEL_NAME = "CustomerChurnEnsemble"


# ── 1. Data loading ──────────────────────────────────────────────


def load_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    train = train.drop(columns=["id"])
    test_id = test.pop("id")

    return train, test, test_id


# ── 2. Feature engineering ───────────────────────────────────────


def engineer_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Apply feature engineering identically to train and test.

    Returns (X, y, test) where X has no target column and the Churn
    target has been label-encoded to int.
    """

    def _feature_eng(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
        df["TotalServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)
        return df

    train = _feature_eng(train)
    test = _feature_eng(test)

    # One-hot encoding
    train = pd.get_dummies(
        train, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int
    )
    test = pd.get_dummies(
        test, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int
    )

    # Binary encoding
    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        train[col] = train[col].map(BINARY_MAP)
        test[col] = test[col].map(BINARY_MAP)

    # Drop gender
    train = train.drop(columns=["gender"], errors="ignore")
    test = test.drop(columns=["gender"], errors="ignore")

    # Encode target
    train["Churn"] = train["Churn"].map(BINARY_MAP)

    X = train.drop(columns=["Churn"])
    y = train["Churn"]

    return X, y, test


# ── 3. Hyperparameter tuning ────────────────────────────────────


def _objective_lgbm(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 10, 60),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
    model = LGBMClassifier(**params, verbosity=-1, random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_xgb(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "eval_metric": "auc",
    }
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
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study_lgbm = optuna.create_study(
        direction="maximize",
        study_name="lgbm_tuning",
        storage=storage_path,
        load_if_exists=True,
    )
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

    best_lgbm_params = study_lgbm.best_params
    best_xgb_params = {
        **study_xgb.best_params,
        "random_state": RANDOM_STATE,
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
) -> tuple[list, list, np.ndarray, list[str]]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    lgb_test_preds = np.zeros(len(test))
    xgb_test_preds = np.zeros(len(test))
    lgb_models: list = []
    xgb_models: list = []

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

    mlflow.end_run()
    mlflow.start_run(run_name="Customer_Churn_Ensemble")

    mlflow.log_param("dataset_train_rows", X.shape[0])
    mlflow.log_param("dataset_train_cols", X.shape[1])
    mlflow.log_param("dataset_test_rows", len(test))
    mlflow.log_param("target_positive_rate", round(float(y.mean()), 4))
    mlflow.log_param("cv_n_splits", N_SPLITS)
    mlflow.log_param("cv_random_state", RANDOM_STATE)

    mlflow.set_tag("task_type", "binary_classification")
    mlflow.set_tag("ensemble_method", "simple_average")
    mlflow.set_tag("models_used", "LightGBM, XGBoost")
    mlflow.set_tag("target_column", "Churn")

    for k, v in best_lgbm_params.items():
        mlflow.log_param(f"lgbm_{k}", v)
    for k, v in best_xgb_params.items():
        mlflow.log_param(f"xgb_{k}", v)

    mlflow.log_text(json.dumps(list(X.columns), indent=2), "feature_list.json")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n{'='*60}")
        print(f"  FOLD {fold + 1} / {N_SPLITS}")
        print(f"  Train: {len(train_idx):,} | Val: {len(val_idx):,}")
        print(f"{'='*60}")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y.iloc[val_idx]

        # ── LightGBM ────────────────────────────────────────────
        with mlflow.start_run(run_name=f"LightGBM_Fold_{fold+1}", nested=True):
            mlflow.log_param("model_type", "LightGBM")
            mlflow.log_param("fold", fold + 1)
            mlflow.log_param("train_fold_size", len(train_idx))
            mlflow.log_param("val_fold_size", len(val_idx))
            mlflow.log_params({f"lgbm_{k}": v for k, v in best_lgbm_params.items()})

            lgb_model = LGBMClassifier(
                **best_lgbm_params,
                objective="binary",
                random_state=RANDOM_STATE,
                verbosity=-1,
            )
            lgb_model.fit(X_train_fold, y_train_fold)

            lgb_val_proba = lgb_model.predict_proba(X_val_fold)[:, 1]
            lgb_val_pred = lgb_model.predict(X_val_fold)

            fold_lgb_auc = roc_auc_score(y_val_fold, lgb_val_proba)
            fold_lgb_acc = accuracy_score(y_val_fold, lgb_val_pred)
            fold_lgb_prec = precision_score(y_val_fold, lgb_val_pred)
            fold_lgb_rec = recall_score(y_val_fold, lgb_val_pred)
            fold_lgb_f1 = f1_score(y_val_fold, lgb_val_pred)
            fold_lgb_ll = log_loss(y_val_fold, lgb_val_proba)

            mlflow.log_metric("val_auc", fold_lgb_auc)
            mlflow.log_metric("val_accuracy", fold_lgb_acc)
            mlflow.log_metric("val_precision", fold_lgb_prec)
            mlflow.log_metric("val_recall", fold_lgb_rec)
            mlflow.log_metric("val_f1", fold_lgb_f1)
            mlflow.log_metric("val_log_loss", fold_lgb_ll)

            feat_imp = pd.DataFrame(
                {"feature": X.columns, "importance": lgb_model.feature_importances_}
            ).sort_values("importance", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.barplot(data=feat_imp.head(15), x="importance", y="feature", ax=ax)
            ax.set_title(f"LightGBM Feature Importance — Fold {fold+1}")
            plt.tight_layout()
            mlflow.log_figure(fig, f"lgbm_feature_importance_fold_{fold+1}.png")
            plt.close(fig)

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

            mlflow.lightgbm.log_model(lgb_model, artifact_path=f"lgbm_model_fold_{fold+1}")

            lgb_val_aucs.append(fold_lgb_auc)
            lgb_val_accs.append(fold_lgb_acc)
            lgb_val_f1s.append(fold_lgb_f1)
            lgb_val_precisions.append(fold_lgb_prec)
            lgb_val_recalls.append(fold_lgb_rec)
            lgb_val_logloss.append(fold_lgb_ll)
            print(f"  LightGBM — AUC: {fold_lgb_auc:.6f} | Acc: {fold_lgb_acc:.6f}")

        lgb_test_preds += lgb_model.predict_proba(test.values)[:, 1] / N_SPLITS
        lgb_models.append(lgb_model)

        # ── XGBoost ─────────────────────────────────────────────
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

    # Per-fold AUC comparison chart
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

    metrics_df = pd.DataFrame({
        "fold": fold_numbers,
        "lgbm_auc": lgb_val_aucs, "xgb_auc": xgb_val_aucs,
        "lgbm_acc": lgb_val_accs, "xgb_acc": xgb_val_accs,
        "lgbm_f1": lgb_val_f1s, "xgb_f1": xgb_val_f1s,
    })
    mlflow.log_text(metrics_df.to_csv(index=False), "cv_fold_metrics.csv")

    final_test_preds = (lgb_test_preds + xgb_test_preds) / 2

    mlflow.log_metric("ensemble_pred_mean", float(np.mean(final_test_preds)))
    mlflow.log_metric("ensemble_pred_std", float(np.std(final_test_preds)))
    mlflow.log_metric("ensemble_pred_median", float(np.median(final_test_preds)))
    mlflow.log_metric("ensemble_pred_min", float(np.min(final_test_preds)))
    mlflow.log_metric("ensemble_pred_max", float(np.max(final_test_preds)))

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
    return lgb_models, xgb_models, final_test_preds, feature_list


# ── 5. MLflow pyfunc wrapper & model registration ───────────────


class ChurnEnsembleModel(mlflow.pyfunc.PythonModel):
    """Wraps all fold models into a single MLflow-servable artefact."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.lgb_models = [
            joblib.load(context.artifacts[f"lgbm_fold_{i}"]) for i in range(1, 6)
        ]
        self.xgb_models = [
            joblib.load(context.artifacts[f"xgb_fold_{i}"]) for i in range(1, 6)
        ]
        with open(context.artifacts["feature_list"]) as f:
            self.feature_list: list[str] = json.load(f)

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
        params: dict | None = None,
    ) -> np.ndarray:
        df = model_input.reindex(columns=self.feature_list, fill_value=0)
        arr = df.values
        lgb_probs = np.mean(
            [m.predict_proba(arr)[:, 1] for m in self.lgb_models], axis=0
        )
        xgb_probs = np.mean(
            [m.predict_proba(arr)[:, 1] for m in self.xgb_models], axis=0
        )
        return (lgb_probs + xgb_probs) / 2


def register_model(
    lgb_models: list,
    xgb_models: list,
    feature_list: list[str],
    best_params: dict,
) -> None:
    """Persist the ensemble as a pyfunc model and register it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        artifacts: dict[str, str] = {}

        for i, m in enumerate(lgb_models, start=1):
            p = tmp_path / f"lgbm_fold_{i}.joblib"
            joblib.dump(m, p)
            artifacts[f"lgbm_fold_{i}"] = str(p)

        for i, m in enumerate(xgb_models, start=1):
            p = tmp_path / f"xgb_fold_{i}.joblib"
            joblib.dump(m, p)
            artifacts[f"xgb_fold_{i}"] = str(p)

        fl_path = tmp_path / "feature_list.json"
        fl_path.write_text(json.dumps(feature_list, indent=2))
        artifacts["feature_list"] = str(fl_path)

        bp_path = tmp_path / "best_params.json"
        bp_path.write_text(json.dumps(best_params, indent=2))
        artifacts["best_params"] = str(bp_path)

        model_info = mlflow.pyfunc.log_model(
            name="ensemble_model",
            python_model=ChurnEnsembleModel(),
            artifacts=artifacts,
        )

    registered = mlflow.register_model(
        model_uri=model_info.model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered.version,
    )

    print(f"\nRegistered {REGISTERED_MODEL_NAME} v{registered.version} "
          f"with alias 'champion'")


# ── main ─────────────────────────────────────────────────────────


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "dataset"

    # tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    tracking_uri = "http://localhost:5001"
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)
        print(f"Using remote MLflow tracking URI: {tracking_uri}")
    else:
        (project_dir / "mlruns").mkdir(parents=True, exist_ok=True)
        db_path = project_dir / "mlflow.db"
        local_tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(local_tracking_uri)
        mlflow.set_registry_uri(local_tracking_uri)
        print(f"Using local MLflow tracking URI: {local_tracking_uri}")

    mlflow.set_experiment("Customer_Churn_Prediction")

    print("Loading data …")
    train_df, test_df, test_id = load_data(data_dir)

    print("Engineering features …")
    X, y, test = engineer_features(train_df, test_df)

    print("Tuning hyperparameters …")
    best_lgbm_params, best_xgb_params = tune_hyperparams(
        X, y, storage_path=f"sqlite:///{project_dir / 'optuna_studies.db'}"
    )

    print("Training ensemble …")
    lgb_models, xgb_models, final_test_preds, feature_list = train_ensemble(
        X, y, test, best_lgbm_params, best_xgb_params
    )

    # Submission CSV
    submission_df = pd.DataFrame({"id": test_id, "Churn": final_test_preds})
    submission_path = project_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    mlflow.log_artifact(str(submission_path))
    print(f"Submission saved to {submission_path}")

    # Register model
    best_params = {"lgbm": best_lgbm_params, "xgb": best_xgb_params}
    register_model(lgb_models, xgb_models, feature_list, best_params)

    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    print(f"\n{'='*60}")
    print("  Pipeline complete!")
    print(f"  Run ID : {run_id}")
    print(f"  mlflow ui --port 5000")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
