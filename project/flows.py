"""Prefect flows for scheduled training and batch prediction.

Wraps the existing train.py pipeline functions as Prefect tasks and
composes them into two flows:
  - training_flow:        end-to-end retraining with Optuna HPO + MLflow tracking
  - batch_prediction_flow: load champion model, predict on a CSV, save results
"""

from __future__ import annotations

import os
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from train import (
    BINARY_MAP,
    ONE_HOT_COLUMNS,
    REGISTERED_MODEL_NAME,
    SERVICE_COLUMNS,
    engineer_features,
    load_data,
    register_model,
    train_ensemble,
    tune_hyperparams,
)


# ── Training tasks ───────────────────────────────────────────────


@task(name="load-data", retries=1)
def load_data_task(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return load_data(data_dir)


@task(name="engineer-features")
def engineer_features_task(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    return engineer_features(train_df, test_df)


@task(name="tune-hyperparams", retries=1, timeout_seconds=3600)
def tune_hyperparams_task(
    X: pd.DataFrame,
    y: pd.Series,
    storage_path: str,
    n_trials: int,
) -> tuple[dict, dict]:
    return tune_hyperparams(X, y, storage_path=storage_path, n_trials=n_trials)


@task(name="train-ensemble", timeout_seconds=3600)
def train_ensemble_task(
    X: pd.DataFrame,
    y: pd.Series,
    test: pd.DataFrame,
    best_lgbm_params: dict,
    best_xgb_params: dict,
) -> tuple[list, list, np.ndarray, list[str], float]:
    return train_ensemble(X, y, test, best_lgbm_params, best_xgb_params)


@task(name="register-model")
def register_model_task(
    lgb_models: list,
    xgb_models: list,
    feature_list: list[str],
    best_params: dict,
    new_auc: float,
) -> None:
    register_model(lgb_models, xgb_models, feature_list, best_params, new_auc)


@task(name="save-submission")
def save_submission_task(
    test_id: pd.Series,
    final_test_preds: np.ndarray,
    project_dir: Path,
) -> None:
    submission_df = pd.DataFrame({"id": test_id, "Churn": final_test_preds})
    submission_path = project_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    mlflow.log_artifact(str(submission_path))
    print(f"Submission saved to {submission_path}")


# ── Batch prediction tasks ───────────────────────────────────────


@task(name="load-batch-data")
def load_batch_data_task(
    input_path: str,
) -> tuple[pd.DataFrame, pd.Series | None]:
    df = pd.read_csv(input_path)
    ids = df.pop("id") if "id" in df.columns else None
    return df, ids


@task(name="preprocess-batch")
def preprocess_batch_task(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering as training for prediction-only data.

    Mirrors engineer_features() in train.py but without target column handling.
    The model's predict() handles column reindexing automatically.
    """
    df = df.copy()

    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TotalServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int)

    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        df[col] = df[col].map(BINARY_MAP)

    df = df.drop(columns=["gender", "Churn"], errors="ignore")

    return df


@task(name="load-champion-model")
def load_champion_model_task():
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    return mlflow.pyfunc.load_model(model_uri)


@task(name="predict-batch")
def predict_batch_task(model, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


@task(name="save-predictions")
def save_predictions_task(
    predictions: np.ndarray,
    ids: pd.Series | None,
    output_path: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame({"churn_probability": predictions})
    if ids is not None:
        result.insert(0, "id", ids.values)
    result["churn"] = (result["churn_probability"] >= 0.5).astype(int)
    result.to_csv(output, index=False)
    print(f"Predictions saved to {output} ({len(result)} rows)")


# ── Flows ────────────────────────────────────────────────────────


def _resolve_tracking_uri(tracking_uri: str) -> str:
    """Return the given URI or fall back to MLFLOW_TRACKING_URI env var."""
    return tracking_uri or os.environ.get(
        "MLFLOW_TRACKING_URI", "http://localhost:5001"
    )


@flow(name="training-pipeline", log_prints=True)
def training_flow(
    data_dir: str = "project/dataset",
    tracking_uri: str = "",
    n_trials: int = 5,
) -> None:
    """Run the full training pipeline: load, engineer, tune, train, register."""
    tracking_uri = _resolve_tracking_uri(tracking_uri)
    data_path = Path(data_dir).resolve()
    project_dir = data_path.parent

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("Customer_Churn_Prediction")
    print(f"MLflow tracking URI: {tracking_uri}")

    train_df, test_df, test_id = load_data_task(data_path)
    X, y, test = engineer_features_task(train_df, test_df)

    best_lgbm_params, best_xgb_params = tune_hyperparams_task(
        X,
        y,
        storage_path=f"sqlite:///{project_dir / 'optuna_studies.db'}",
        n_trials=n_trials,
    )

    lgb_models, xgb_models, final_test_preds, feature_list, ensemble_auc = (
        train_ensemble_task(X, y, test, best_lgbm_params, best_xgb_params)
    )

    save_submission_task(test_id, final_test_preds, project_dir)

    best_params = {"lgbm": best_lgbm_params, "xgb": best_xgb_params}
    register_model_task(
        lgb_models, xgb_models, feature_list, best_params, ensemble_auc
    )

    run_id = mlflow.active_run().info.run_id
    mlflow.end_run()

    create_markdown_artifact(
        key="training-summary",
        markdown=(
            f"## Training Pipeline Complete\n\n"
            f"- **Ensemble Mean CV AUC**: {ensemble_auc:.6f}\n"
            f"- **MLflow Run ID**: `{run_id}`\n"
            f"- **Optuna Trials**: {n_trials}\n"
        ),
    )
    print(f"Pipeline complete! Run ID: {run_id}")


@flow(name="batch-prediction", log_prints=True)
def batch_prediction_flow(
    input_path: str,
    output_path: str,
    tracking_uri: str = "",
) -> None:
    """Load the champion model and generate predictions for a CSV of customers."""
    tracking_uri = _resolve_tracking_uri(tracking_uri)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    df, ids = load_batch_data_task(input_path)
    features = preprocess_batch_task(df)
    model = load_champion_model_task()
    predictions = predict_batch_task(model, features)
    save_predictions_task(predictions, ids, output_path)

    create_markdown_artifact(
        key="batch-prediction-summary",
        markdown=(
            f"## Batch Prediction Complete\n\n"
            f"- **Input**: `{input_path}`\n"
            f"- **Output**: `{output_path}`\n"
            f"- **Rows processed**: {len(predictions)}\n"
            f"- **Mean churn probability**: {float(np.mean(predictions)):.4f}\n"
            f"- **Predicted churners**: {int((predictions >= 0.5).sum())}\n"
        ),
    )
