# Enables PEP 604 union syntax (str | None) in annotations for Python <3.10
from __future__ import annotations

import os

# asynccontextmanager: decorator that turns an async generator function into
# an async context manager (used with `async with`). FastAPI's lifespan
# protocol requires an async context manager that handles startup and shutdown.
from contextlib import asynccontextmanager
from enum import Enum

# mlflow.pyfunc: the generic "Python function" model flavor in MLflow.
# Our ChurnEnsembleModel was registered as a pyfunc model, so we load
# it with this module. pyfunc models expose a single .predict() method
# regardless of the underlying framework (LightGBM, XGBoost, etc.).
import mlflow.pyfunc
import numpy as np
import pandas as pd

# FastAPI: the web framework. It automatically generates OpenAPI/Swagger docs,
# validates request bodies against Pydantic models, and handles JSON serialization.
# HTTPException: raises HTTP error responses with a status code and detail message.
from fastapi import FastAPI, HTTPException

# Import the Pydantic schemas that define the shape of API requests and responses.
# These are defined in a separate module for separation of concerns.
from schemas.schemas import BatchPredictionResponse, CustomerData, PredictionResponse

# ── Preprocessing (mirrors the training pipeline exactly) ────────
# CRITICAL: These constants and the preprocess() function must stay
# in sync with the corresponding code in train.py (engineer_features).
# If the training pipeline changes how features are created, this file
# must be updated identically, or the model will receive mismatched features
# at inference time, producing silently wrong predictions.

# Lookup for converting binary "Yes"/"No" string columns to 0/1 integers.
# Must match the BINARY_MAP in train.py exactly.
BINARY_MAP = {"Yes": 1, "No": 0}

# The 8 telecom service columns used to compute the TotalServices feature.
# Order doesn't matter for the boolean comparison, but the list must contain
# the same column names as in train.py.
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

# Categorical columns to one-hot encode. Must be identical to ONE_HOT_COLUMNS
# in train.py — same columns, same order — to produce the same dummy variables.
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

# Name of the model in MLflow's Model Registry. The lifespan handler
# loads the model by this name + the "champion" alias.
REGISTERED_MODEL_NAME = "CustomerChurnEnsemble"


def _to_container_mlruns_path(source_path: str) -> str | None:
    """Rewrite a host-local MLflow artifact path to the Docker container's mount path.

    When the model is trained on the host machine, MLflow stores artifact
    paths using the host's absolute filesystem paths (e.g. /Users/dan/project/mlruns/...).
    Inside a Docker container, the mlruns directory is mounted at /mlflow/mlruns/.
    This function detects the /mlruns/ segment and rewrites the path so the
    container can find the artifacts.

    Args:
        source_path: The original artifact path from MLflow's metadata.

    Returns:
        The rewritten container path, or None if the path doesn't contain /mlruns/
        (meaning it doesn't need rewriting or isn't an artifact path).
    """
    marker = "/mlruns/"
    if marker not in source_path:
        return None
    # Split on the marker and take everything after it — this is the relative
    # path within the mlruns directory. rstrip("/.") cleans trailing artifacts.
    relative = source_path.split(marker, maxsplit=1)[1].rstrip("/.")
    if not relative:
        return None
    return f"/mlflow/mlruns/{relative}"


def _load_registry_model(model_name: str, alias: str):
    """Load the model from MLflow Model Registry with Docker-aware fallback logic.

    MLflow's standard model loading assumes the artifact store paths are valid
    on the current machine. In Docker environments, the paths from the host
    don't exist. This function implements a cascading strategy:

    1. Try the standard MLflow URI (works on the host or if paths match)
    2. If that fails, query the registry for the version's source path and
       rewrite it to the container's mount location
    3. If the source uses models:/ URI scheme, resolve through the run/experiment
       metadata to reconstruct the container-local path

    Args:
        model_name: Registered model name (e.g. "CustomerChurnEnsemble").
        alias: Model alias to load (e.g. "champion").

    Returns:
        The loaded MLflow pyfunc model object.

    Raises:
        The original exception from step 1 if all fallback strategies fail.
    """
    # Clear any previously loaded model to prevent stale state
    models_state.clear()

    # Standard MLflow model URI format: "models:/ModelName@alias"
    model_uri = f"models:/{model_name}@{alias}"
    try:
        # Attempt 1: direct load via the standard URI. This works when the
        # MLflow server and artifact store paths are accessible from the
        # current environment (e.g. running directly on the host).
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as first_exc:
        try:
            # Attempt 2: query the MLflow registry for the model version's
            # source path, then rewrite it for the container filesystem.
            client = mlflow.tracking.MlflowClient()
            version = client.get_model_version_by_alias(name=model_name, alias=alias)
            source = version.source or ""

            # Try rewriting the source path from host → container
            rewritten = _to_container_mlruns_path(source)
            if rewritten:
                return mlflow.pyfunc.load_model(rewritten)

            # Attempt 3: if the source is a models:/ URI (not a file path),
            # we need to resolve it through the run and experiment metadata
            # to find the actual artifact location on disk.
            if source.startswith("models:/"):
                model_id = source[len("models:/"):]
                run = client.get_run(version.run_id)
                exp = client.get_experiment(run.info.experiment_id)
                base = _to_container_mlruns_path(exp.artifact_location or "")
                if base:
                    return mlflow.pyfunc.load_model(
                        f"{base}/models/{model_id}/artifacts"
                    )

            # All fallback strategies exhausted — re-raise the original error
            # so the error message points to the root cause, not the fallback
            raise first_exc
        except Exception:
            raise first_exc


def preprocess(customers: list[CustomerData]) -> pd.DataFrame:
    """Convert validated Pydantic customer records into a model-ready feature DataFrame.

    This is the serving-time equivalent of `engineer_features()` in train.py.
    It must apply the exact same transformations (derived features, one-hot
    encoding, binary encoding) to produce a feature matrix compatible with
    the trained model.

    Note: column alignment to the training feature order is NOT done here —
    it happens inside ChurnEnsembleModel.predict() via df.reindex(). This
    function only needs to produce the correct column names and values.

    Args:
        customers: List of validated CustomerData Pydantic objects.

    Returns:
        A pandas DataFrame with engineered features ready for model.predict().
    """
    # Convert each Pydantic model to a plain Python dict using .model_dump()
    # (Pydantic v2 API; replaces .dict() from v1).
    rows = [c.model_dump() for c in customers]

    # Resolve Pydantic Enum values to their underlying strings.
    # model_dump() may return Enum instances (e.g. YesNo.yes) rather than
    # plain strings ("Yes"). Pandas and the ML model expect plain strings,
    # so we extract .value from each Enum instance. Non-Enum values (ints,
    # floats) are left unchanged.
    for row in rows:
        for k, v in row.items():
            if isinstance(v, Enum):
                row[k] = v.value

    # Build a DataFrame from the list of dicts. Each dict becomes one row.
    df = pd.DataFrame(rows)

    # Derived features — identical to train.py's _feature_eng():
    # AvgMonthlyCharge: estimated average monthly spend over the customer's tenure.
    # +1 prevents division by zero for brand-new customers (tenure=0).
    df["AvgMonthlyCharge"] = df["TotalCharges"] / (df["tenure"] + 1)

    # TotalServices: count of subscribed services (0-8). Compares each service
    # column to "Yes" (producing a boolean DataFrame) and sums across columns per row.
    df["TotalServices"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    # One-hot encode categorical columns. pd.get_dummies creates binary indicator
    # columns for each category. drop_first=True removes one category per column
    # to avoid multicollinearity (the dummy variable trap). dtype=int produces
    # 0/1 integers rather than True/False booleans.
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, drop_first=True, dtype=int)

    # Binary encode the four simple Yes/No columns to 0/1 integers.
    # These columns are not in ONE_HOT_COLUMNS because they only have two
    # values, so a single 0/1 column is sufficient (no need for dummies).
    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        df[col] = df[col].map(BINARY_MAP)

    return df


# ── Model loading & lifespan ─────────────────────────────────────

# Module-level dictionary that acts as the global model container.
# Using a mutable dict rather than a plain global variable avoids needing
# `global model` declarations inside the lifespan function. The dict is
# shared across all request handlers — they access models_state["model"]
# to get the loaded ML model.
models_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan handler: manages startup and shutdown of the application.

    The lifespan protocol (introduced in FastAPI 0.93+ / Starlette 0.20+)
    replaces the older @app.on_event("startup") and @app.on_event("shutdown")
    decorators. Code before `yield` runs at startup; code after `yield` runs
    at shutdown. The `yield` itself is the application's running period.

    Startup:
        1. Configure MLflow tracking URI from environment variable
        2. Load the champion model from MLflow Model Registry
        3. Store the model in models_state for request handlers to use

    Shutdown:
        1. Clear models_state to release the model from memory
    """
    # Read the MLflow tracking URI from the environment. In Docker Compose,
    # this is set to "http://mlflow:5001" (the MLflow container's hostname).
    # Locally, it might be "http://localhost:5001" or unset.
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    try:
        # Load the model registered as "CustomerChurnEnsemble" with alias "champion".
        # This calls ChurnEnsembleModel.load_context() internally, which deserializes
        # all 10 fold models (5 LightGBM + 5 XGBoost) from joblib files.
        models_state["model"] = _load_registry_model(
            REGISTERED_MODEL_NAME,
            alias="champion",
        )
    except Exception as exc:
        # If the model can't be loaded (e.g. train.py hasn't been run yet,
        # or the MLflow server is unreachable), fail fast with a clear error
        # message rather than starting a server that can't serve predictions.
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"
        raise RuntimeError(
            f"Failed to load model from registry ({model_uri}). "
            "Run train.py first to register the model against the active MLflow "
            "server (for Docker: MLFLOW_TRACKING_URI=http://localhost:5001). "
            f"Error: {exc}"
        ) from exc

    # yield marks the transition from startup to running. The application
    # serves requests until a shutdown signal is received (Ctrl+C, SIGTERM).
    yield

    # Shutdown: clean up the loaded model to free memory
    models_state.clear()


# ── FastAPI app ──────────────────────────────────────────────────

# Instantiate the FastAPI application. Parameters here populate the
# auto-generated OpenAPI spec (visible at /docs and /redoc):
# - title: shown at the top of the Swagger UI
# - description: appears below the title
# - version: API version string
# - lifespan: the startup/shutdown handler defined above
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
    """Shared prediction logic used by both single and batch endpoints.

    Factored into a private function to avoid code duplication between
    /predict (single customer) and /predict/batch (multiple customers).
    The single endpoint wraps its input in a list and takes [0] of the result.

    Args:
        customers: List of validated CustomerData objects.

    Returns:
        List of PredictionResponse objects, one per input customer.
    """
    # Retrieve the loaded pyfunc model from the global state
    model = models_state["model"]

    # Transform the raw customer data into a feature DataFrame
    df = preprocess(customers)

    # model.predict() calls ChurnEnsembleModel.predict(), which:
    #   1. Reindexes the DataFrame to match training column order
    #   2. Averages predictions across all 5 LightGBM fold models
    #   3. Averages predictions across all 5 XGBoost fold models
    #   4. Returns the mean of both averages (the ensemble prediction)
    # The result is a 1D numpy array of churn probabilities.
    probs: np.ndarray = model.predict(df)

    # Convert each probability into a PredictionResponse with:
    #   - churn_probability: rounded to 6 decimal places for clean JSON output
    #   - churn: binary decision using the 0.5 threshold
    return [
        PredictionResponse(
            churn_probability=round(float(probs[i]), 6),
            churn=bool(probs[i] >= 0.5),
        )
        for i in range(len(customers))
    ]


# ── API Endpoints ────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check endpoint for monitoring and load balancers.

    Returns service status, the model name, and whether the model is loaded.
    In practice, "loaded" is always True on a running server because the
    lifespan handler raises RuntimeError if loading fails (the server
    won't start). This endpoint is useful for container orchestrators
    (e.g. Docker health checks, Kubernetes readiness probes).
    """
    return {
        "status": "healthy",
        "model": REGISTERED_MODEL_NAME,
        "loaded": "model" in models_state,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """Predict churn for a single customer.

    FastAPI automatically:
      1. Parses the JSON request body
      2. Validates it against CustomerData (returns 422 on invalid data)
      3. Passes the validated object as the `customer` parameter

    The response_model parameter tells FastAPI to validate and serialize
    the return value as PredictionResponse, and documents the response
    schema in the OpenAPI spec.

    Args:
        customer: A validated CustomerData object from the request body.

    Returns:
        PredictionResponse with churn_probability and churn boolean.

    Raises:
        HTTPException(400): If prediction fails (e.g. preprocessing error).
    """
    try:
        # Wrap single customer in a list for batch-compatible processing,
        # then extract the first (only) result
        return _predict([customer])[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(customers: list[CustomerData]):
    """Predict churn for multiple customers in a single request.

    More efficient than calling /predict N times because:
      1. Only one HTTP round-trip instead of N
      2. Feature preprocessing is vectorized across all rows at once
      3. Model inference processes the full batch as a single matrix operation

    Args:
        customers: List of validated CustomerData objects from the request body.

    Returns:
        BatchPredictionResponse containing a list of PredictionResponse objects
        in the same order as the input customers.

    Raises:
        HTTPException(400): If the customer list is empty or prediction fails.
    """
    # Explicitly reject empty lists. Without this check, the model would
    # receive an empty DataFrame which might produce confusing errors.
    if not customers:
        raise HTTPException(status_code=400, detail="Empty customer list")
    try:
        return BatchPredictionResponse(predictions=_predict(customers))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/model/info")
def model_info():
    """Return metadata about the currently loaded model.

    Useful for debugging and operational monitoring — tells you which
    model version is serving predictions, which MLflow run produced it,
    and what flavors (model formats) it supports.

    Returns:
        JSON object with registered model name, URI, run ID, artifact path,
        and list of supported MLflow flavors. Returns an error object if
        no model is loaded.
    """
    model = models_state.get("model")
    if model is None:
        return {"error": "Model not loaded"}

    # The metadata attribute on a loaded pyfunc model contains information
    # from the MLmodel file that was logged alongside the model artifacts.
    meta = model.metadata
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

    # flavors: the model formats the artifact supports (e.g. "python_function",
    # "lightgbm"). Our ensemble uses only "python_function" (pyfunc) since
    # it's a custom wrapper, not a single native model.
    # getattr with fallback handles cases where metadata might be incomplete.
    flavors = getattr(meta, "flavors", {}) or {}
    return {
        "registered_model": REGISTERED_MODEL_NAME,
        "model_uri": model_uri,
        "run_id": getattr(meta, "run_id", None),
        "artifact_path": getattr(meta, "artifact_path", None),
        "flavors": list(flavors.keys()),
    }
