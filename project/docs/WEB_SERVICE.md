# `web_service.py` — FastAPI Web Service Documentation

## Overview

`web_service.py` is the serving layer for the Customer Churn Prediction system. It exposes a FastAPI application that loads the trained ensemble model from the MLflow Model Registry at startup and provides HTTP endpoints for real-time churn predictions. The service handles all necessary feature preprocessing to convert raw customer data into the format the model expects.

```bash
# Local development
uvicorn project.web_service:app --reload

# Via Docker Compose
cd project && docker-compose up --build
```

---

## Architecture

```
HTTP Request
    │
    ▼
┌─────────────────────────┐
│  FastAPI (Pydantic       │  ← Input validation via CustomerData schema
│  validation layer)       │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  preprocess()            │  ← Feature engineering (mirrors train.py)
│  Enum → string → encode  │
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  ChurnEnsembleModel      │  ← MLflow pyfunc: 5 LightGBM + 5 XGBoost
│  .predict(df)            │     models averaged together
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  PredictionResponse      │  ← JSON response with probability + decision
└─────────────────────────┘
```

---

## Constants

| Constant | Value | Purpose |
|---|---|---|
| `BINARY_MAP` | `{"Yes": 1, "No": 0}` | Maps binary string columns to integers. Must match `train.py`. |
| `SERVICE_COLUMNS` | 8 columns | Used to compute the `TotalServices` derived feature. Must match `train.py`. |
| `ONE_HOT_COLUMNS` | 10 columns | Categorical columns to one-hot encode. Must match `train.py`. |
| `REGISTERED_MODEL_NAME` | `"CustomerChurnEnsemble"` | The name of the model in MLflow Registry. |

---

## Helper Functions

### `_to_container_mlruns_path(source_path: str) -> str | None`

**Purpose:** Resolves path mismatches between the host machine and the Docker container.

When the model is trained on the host (e.g. artifacts stored at `/Users/dan/.../mlruns/...`), those absolute paths are baked into MLflow's registry metadata. Inside a Docker container, the artifacts are mounted at `/mlflow/mlruns/...` instead.

This function detects the `/mlruns/` marker in a path and rewrites it to the container's mount point. Returns `None` if the path doesn't contain the marker, signaling that no rewrite is needed.

**Example:**
```
Input:  "/Users/dan/project/mlruns/1/abc123/artifacts/model"
Output: "/mlflow/mlruns/1/abc123/artifacts/model"
```

---

### `_load_registry_model(model_name: str, alias: str)`

**Purpose:** Loads the model from MLflow Registry with robust fallback logic for Docker environments.

**Loading strategy (cascading fallback):**

1. **Direct load:** Tries `mlflow.pyfunc.load_model("models:/CustomerChurnEnsemble@champion")`. This works when the MLflow server and artifact store paths match the current environment.

2. **Path rewrite fallback:** If the direct load fails (common in Docker), it:
   - Fetches the model version metadata via the MLflow client.
   - Extracts the `source` path from the version record.
   - Rewrites the path using `_to_container_mlruns_path()`.
   - Attempts to load from the rewritten path.

3. **Deep artifact resolution:** If the source starts with `models:/`, it resolves through the run and experiment metadata to reconstruct the artifact path under the container mount.

4. **Original error propagation:** If all fallbacks fail, the original exception from step 1 is re-raised.

This function also calls `models_state.clear()` at the start, ensuring any stale model reference is cleaned up before a load attempt.

---

### `preprocess(customers: list[CustomerData]) -> pd.DataFrame`

**Purpose:** Converts validated Pydantic `CustomerData` objects into a model-ready feature DataFrame. This is the serving-time equivalent of `engineer_features()` from `train.py`.

**Steps:**

| Step | Operation | Details |
|---|---|---|
| 1 | Dump to dicts | `c.model_dump()` converts each Pydantic model to a plain dict. |
| 2 | Resolve enums | Iterates over each row and replaces any `Enum` instances with their `.value` string. This is necessary because `pd.DataFrame` doesn't natively handle Pydantic enums. |
| 3 | Create DataFrame | Converts the list of dicts into a pandas DataFrame. |
| 4 | Derived features | Computes `AvgMonthlyCharge` and `TotalServices` — identical formulas to `train.py`. |
| 5 | One-hot encoding | `pd.get_dummies(columns=ONE_HOT_COLUMNS, drop_first=True)` — same columns and settings as training. |
| 6 | Binary encoding | Maps `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` from "Yes"/"No" to 1/0. |

**What it does NOT do:** Column alignment to the training feature order. That responsibility is handled inside the `ChurnEnsembleModel.predict()` method (via `df.reindex(columns=self.feature_list, fill_value=0)`), which fills any missing columns with 0 and drops any extra columns.

**Critical Constraint:** This function must stay in sync with `engineer_features()` in `train.py`. Divergence between the two will cause silent prediction errors (wrong features, wrong encodings).

---

## Application Lifecycle

### `lifespan(app: FastAPI)` — Async Context Manager

Manages the model loading at startup and cleanup at shutdown using FastAPI's lifespan protocol.

**Startup:**
1. Reads `MLFLOW_TRACKING_URI` from the environment. If set, configures MLflow to use that URI (e.g. `http://mlflow:5001` inside Docker).
2. Calls `_load_registry_model()` to load the champion model.
3. Stores the loaded model in `models_state["model"]` — a module-level dictionary that acts as the global model store.
4. If loading fails, raises a `RuntimeError` with a diagnostic message suggesting the user run `train.py` first.

**Shutdown:**
- Clears `models_state`, releasing the model from memory.

### `models_state: dict`

A module-level dictionary that serves as the global model container. Using a mutable dict (rather than a global variable) allows the model to be set inside the lifespan context manager without `global` declarations. The key `"model"` holds the loaded MLflow pyfunc model.

---

## API Endpoints

### `GET /` — Health Check

**Response:** JSON object with service status.

```json
{
  "status": "healthy",
  "model": "CustomerChurnEnsemble",
  "loaded": true
}
```

The `loaded` field reflects whether a model is currently in `models_state`. If the model failed to load at startup, the application wouldn't start at all (the lifespan raises a `RuntimeError`), so in practice this is always `true` on a running server.

---

### `POST /predict` — Single Customer Prediction

**Request body:** A single `CustomerData` JSON object.

**Response model:** `PredictionResponse`

```json
{
  "churn_probability": 0.234567,
  "churn": false
}
```

**Error handling:** Wraps the prediction call in a try/except. Any exception (e.g. preprocessing failure) returns a 400 status with the error message in the `detail` field.

**Implementation:** Delegates to `_predict([customer])[0]` — wraps the single customer in a list for batch-compatible processing, then returns the first (only) result.

---

### `POST /predict/batch` — Batch Prediction

**Request body:** A JSON array of `CustomerData` objects.

**Response model:** `BatchPredictionResponse`

```json
{
  "predictions": [
    {"churn_probability": 0.234567, "churn": false},
    {"churn_probability": 0.891234, "churn": true}
  ]
}
```

**Validation:** Rejects empty lists with a 400 error before attempting prediction.

**Implementation:** Passes the full list to `_predict()` and wraps the result in `BatchPredictionResponse`.

---

### `GET /model/info` — Model Metadata

**Response:** JSON object with metadata about the currently loaded model.

```json
{
  "registered_model": "CustomerChurnEnsemble",
  "model_uri": "models:/CustomerChurnEnsemble@champion",
  "run_id": "abc123def456",
  "artifact_path": "ensemble_model",
  "flavors": ["python_function"]
}
```

Extracts metadata from the MLflow model's `metadata` attribute. Returns an error object if the model is not loaded.

---

## Internal Prediction Flow

### `_predict(customers: list[CustomerData]) -> list[PredictionResponse]`

The shared prediction logic used by both `/predict` and `/predict/batch`.

**Steps:**
1. Retrieves the model from `models_state["model"]`.
2. Calls `preprocess(customers)` to build the feature DataFrame.
3. Calls `model.predict(df)` — this invokes `ChurnEnsembleModel.predict()`, which:
   - Reindexes columns to match the training feature order.
   - Averages probabilities across all 5 LightGBM models.
   - Averages probabilities across all 5 XGBoost models.
   - Returns the mean of both averages.
4. Maps each probability to a `PredictionResponse` with:
   - `churn_probability` — rounded to 6 decimal places.
   - `churn` — `True` if probability >= 0.5, `False` otherwise.

---

## Docker Deployment Notes

When deployed via `docker-compose.yml`, the web service container:
- Sets `MLFLOW_TRACKING_URI=http://mlflow:5001` to reach the MLflow container by its service name.
- Mounts the host's `mlruns/` directory into the container at `/mlflow/mlruns/`, which is why `_to_container_mlruns_path()` exists — to bridge the path mismatch between host-trained artifacts and the container's filesystem.

**Startup order matters:** The MLflow server container must be running and the model must be registered (via `train.py`) before the API container starts. If the model is not found in the registry, the API will fail to start with a `RuntimeError`.

---

## Dependencies

| Library | Role |
|---|---|
| `fastapi` | Web framework, routing, request validation |
| `mlflow.pyfunc` | Loading the registered ensemble model |
| `pandas` | DataFrame construction and feature engineering |
| `numpy` | Probability array handling |
| `pydantic` | Input/output validation (via schemas) |
