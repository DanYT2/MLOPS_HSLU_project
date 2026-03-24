from __future__ import annotations

import os
from contextlib import asynccontextmanager
from enum import Enum

import mlflow.pyfunc
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas.schemas import BatchPredictionResponse, CustomerData, PredictionResponse

# ── Preprocessing (mirrors the training pipeline exactly) ────────

BINARY_MAP = {"Yes": 1, "No": 0}

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

REGISTERED_MODEL_NAME = "CustomerChurnEnsemble"


def _to_container_mlruns_path(source_path: str) -> str | None:
    """Map host-local MLflow artifact paths to the container mount path."""
    marker = "/mlruns/"
    if marker not in source_path:
        return None
    relative = source_path.split(marker, maxsplit=1)[1].rstrip("/.")
    if not relative:
        return None
    return f"/mlflow/mlruns/{relative}"


def _load_registry_model(model_name: str, alias: str):
    models_state.clear()
    model_uri = f"models:/{model_name}@{alias}"
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as first_exc:
        try:
            client = mlflow.tracking.MlflowClient()
            version = client.get_model_version_by_alias(name=model_name, alias=alias)
            source = version.source or ""

            rewritten = _to_container_mlruns_path(source)
            if rewritten:
                return mlflow.pyfunc.load_model(rewritten)

            if source.startswith("models:/"):
                model_id = source[len("models:/"):]
                run = client.get_run(version.run_id)
                exp = client.get_experiment(run.info.experiment_id)
                base = _to_container_mlruns_path(exp.artifact_location or "")
                if base:
                    return mlflow.pyfunc.load_model(
                        f"{base}/models/{model_id}/artifacts"
                    )

            raise first_exc
        except Exception:
            raise first_exc


def preprocess(customers: list[CustomerData]) -> pd.DataFrame:
    """Convert raw customer records into the model-ready feature matrix.

    Column alignment to the training feature order happens inside the
    pyfunc model itself, so we only need to engineer features here.
    """
    rows = [c.model_dump() for c in customers]
    for row in rows:
        for k, v in row.items():
            if isinstance(v, Enum):
                row[k] = v.value
    df = pd.DataFrame(rows)

    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TotalServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int)

    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        df[col] = df[col].map(BINARY_MAP)

    return df


# ── Model loading & lifespan ─────────────────────────────────────

models_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        models_state["model"] = _load_registry_model(
            REGISTERED_MODEL_NAME,
            alias="champion",
        )
    except Exception as exc:
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
        raise RuntimeError(
            f"Failed to load model from registry ({model_uri}). "
            "Run train.py first to register the model against the active MLflow "
            "server (for Docker: MLFLOW_TRACKING_URI=http://localhost:5001). "
            f"Error: {exc}"
        ) from exc

    yield
    models_state.clear()


# ── FastAPI app ──────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "LightGBM + XGBoost ensemble (5-fold CV) "
        "for binary customer churn prediction."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


def _predict(customers: list[CustomerData]) -> list[PredictionResponse]:
    model = models_state["model"]
    df = preprocess(customers)
    probs: np.ndarray = model.predict(df)

    return [
        PredictionResponse(
            churn_probability=round(float(probs[i]), 6),
            churn=bool(probs[i] >= 0.5),
        )
        for i in range(len(customers))
    ]


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": REGISTERED_MODEL_NAME,
        "loaded": "model" in models_state,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    try:
        return _predict([customer])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(customers: list[CustomerData]):
    if not customers:
        raise HTTPException(status_code=400, detail="Empty customer list")
    try:
        return BatchPredictionResponse(predictions=_predict(customers))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/info")
def model_info():
    model = models_state.get("model")
    if model is None:
        return {"error": "Model not loaded"}

    meta = model.metadata
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
    flavors = getattr(meta, "flavors", {}) or {}
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "model_uri": model_uri,
        "run_id": getattr(meta, "run_id", None),
        "artifact_path": getattr(meta, "artifact_path", None),
        "flavors": list(flavors.keys()),
    }
