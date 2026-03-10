from __future__ import annotations

import json
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from schemas.schemas import CustomerData, PredictionResponse, BatchPredictionResponse


# ── Preprocessing (mirrors the notebook pipeline exactly) ───────

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


def preprocess(customers: list[CustomerData], feature_list: list[str]) -> pd.DataFrame:
    """Convert raw customer records into the model-ready feature matrix."""
    rows = [c.model_dump() for c in customers]
    for row in rows:
        for k, v in row.items():
            if isinstance(v, Enum):
                row[k] = v.value
    df = pd.DataFrame(rows)

    # 1. Feature engineering (before encoding, uses raw string values)
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TotalServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    # 2. One-hot encoding (drop_first=True to match training)
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int)

    # 3. Binary encoding
    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        df[col] = df[col].map(BINARY_MAP)

    # Align columns to the exact training feature order, filling any
    # missing one-hot columns with 0 (can happen if the request doesn't
    # trigger every category).
    df = df.reindex(columns=feature_list, fill_value=0)
    return df


# ── Model loading & lifespan ────────────────────────────────────

MODELS_DIR = Path(__file__).resolve().parent / "models"

models_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    feature_path = MODELS_DIR / "feature_list.json"
    if not feature_path.exists():
        raise RuntimeError(
            f"Model artifacts not found in {MODELS_DIR}. "
            "Run the notebook first to export them."
        )

    with open(feature_path) as f:
        models_state["feature_list"] = json.load(f)

    lgb_models = []
    xgb_models = []
    for i in range(1, 6):
        lgb_models.append(joblib.load(MODELS_DIR / f"lgbm_fold_{i}.joblib"))
        xgb_models.append(joblib.load(MODELS_DIR / f"xgb_fold_{i}.joblib"))

    models_state["lgb_models"] = lgb_models
    models_state["xgb_models"] = xgb_models

    with open(MODELS_DIR / "best_params.json") as f:
        models_state["best_params"] = json.load(f)

    yield
    models_state.clear()


# ── FastAPI app ─────────────────────────────────────────────────

app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "LightGBM + XGBoost ensemble (5-fold CV) "
        "for binary customer churn prediction."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _predict(customers: list[CustomerData]) -> list[PredictionResponse]:
    feature_list: list[str] = models_state["feature_list"]
    lgb_models: list = models_state["lgb_models"]
    xgb_models: list = models_state["xgb_models"]

    df = preprocess(customers, feature_list)
    arr = df.values

    lgb_probs = np.mean(
        [m.predict_proba(arr)[:, 1] for m in lgb_models], axis=0
    )
    xgb_probs = np.mean(
        [m.predict_proba(arr)[:, 1] for m in xgb_models], axis=0
    )
    ensemble_probs = (lgb_probs + xgb_probs) / 2

    return [
        PredictionResponse(
            churn_probability=round(float(ensemble_probs[i]), 6),
            churn=bool(ensemble_probs[i] >= 0.5),
            lgbm_probability=round(float(lgb_probs[i]), 6),
            xgb_probability=round(float(xgb_probs[i]), 6),
        )
        for i in range(len(customers))
    ]


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "model": "LightGBM + XGBoost Ensemble",
        "n_models": len(models_state.get("lgb_models", []))
        + len(models_state.get("xgb_models", [])),
        "features": len(models_state.get("feature_list", [])),
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
    return {
        "ensemble_method": "simple_average",
        "models": {
            "lightgbm": {
                "count": len(models_state.get("lgb_models", [])),
                "params": models_state.get("best_params", {}).get("lgbm"),
            },
            "xgboost": {
                "count": len(models_state.get("xgb_models", [])),
                "params": models_state.get("best_params", {}).get("xgb"),
            },
        },
        "feature_list": models_state.get("feature_list", []),
    }
