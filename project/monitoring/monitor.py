"""Monitoring pipeline for the Customer Churn FastAPI model.

This module acts as a small production simulator for the churn prediction
service. It reads the project CSV files, reserves a stable reference slice from
the training data, then replays the remaining rows in batches. Each batch is
sent to the running FastAPI service so the exact deployed model produces the
prediction probabilities used by the monitoring reports.

For every replayed batch, the script compares the current batch against the
reference slice with Evidently. The resulting drift, data quality, prediction,
and model-quality metrics are flattened into scalar values and inserted into
Postgres. Grafana is provisioned to query that table directly, so this script is
the bridge between the model-serving API and the observability dashboard.

The loop behaves like a lightweight cron: it ticks every ``INTERVAL_SECONDS``
until every batch from both simulation pools has been consumed. Set
``LOOP_FOREVER=1`` to replay the pools repeatedly, or ``RUN_ONCE=1`` to force a
single pass even if the forever flag is enabled.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import requests
from evidently import ColumnMapping
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.report import Report
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Logging ──────────────────────────────────────────────────────
# Log to stdout because Docker captures container stdout/stderr. Keeping the
# format compact makes ``docker compose logs monitor`` useful during demos while
# still preserving timestamps for troubleshooting batch timing.
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("monitor")

# ── Config ───────────────────────────────────────────────────────
# Runtime configuration is read from environment variables so docker-compose can
# tune the simulator without rebuilding the image. Defaults match the service
# names and container paths from project/docker-compose.yml.
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "/app/dataset"))
API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", "10"))
REFERENCE_FRAC = float(os.environ.get("REFERENCE_FRAC", "0.2"))
RUN_ONCE = os.environ.get("RUN_ONCE", "0") == "1"
LOOP_FOREVER = os.environ.get("LOOP_FOREVER", "0") == "1"
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))
API_WAIT_SECONDS = int(os.environ.get("API_WAIT_SECONDS", "180"))

# ── Synthetic drift injection ────────────────────────────────────
# These knobs let us deliberately bend the input distribution so Evidently's
# drift scores rise on the Grafana dashboard. They default to OFF so production
# runs are unaffected — set DRIFT_MODE to anything other than "none" to enable.
#
# Modes:
#   none       — pass-through; no perturbation
#   sustained  — apply full-intensity drift to every batch from DRIFT_START_BATCH on
#   gradual    — linearly ramp intensity 0 → 1 over DRIFT_RAMP_BATCHES batches
#   cycle      — alternate clean / drifted windows of length DRIFT_PERIOD_BATCHES // 2
#
# The intensity scalar (0..1+) modulates three perturbations in lockstep:
#   1. Numeric features get a Gaussian mean shift of ~intensity * NUMERIC_SHIFT_STD
#      reference standard deviations (clamped to schema-valid ranges).
#   2. Categorical features are randomly resampled to another *observed* value
#      with probability intensity * CATEGORICAL_SWAP_PROB (keeps payloads valid).
#   3. After scoring, intensity * MISSING_RATE of feature cells in the Evidently
#      input frame are blanked, which feeds DatasetMissingValuesMetric. We do
#      this post-prediction so the API never receives NaNs.
DRIFT_MODE = os.environ.get("DRIFT_MODE", "none").lower()
DRIFT_START_BATCH = int(os.environ.get("DRIFT_START_BATCH", "0"))
DRIFT_RAMP_BATCHES = max(1, int(os.environ.get("DRIFT_RAMP_BATCHES", "5")))
DRIFT_PERIOD_BATCHES = max(2, int(os.environ.get("DRIFT_PERIOD_BATCHES", "6")))
DRIFT_NUMERIC_SHIFT_STD = float(os.environ.get("DRIFT_NUMERIC_SHIFT_STD", "1.5"))
DRIFT_CATEGORICAL_SWAP_PROB = float(os.environ.get("DRIFT_CATEGORICAL_SWAP_PROB", "0.4"))
DRIFT_MISSING_RATE = float(os.environ.get("DRIFT_MISSING_RATE", "0.0"))
DRIFT_RANDOM_STATE = int(os.environ.get("DRIFT_RANDOM_STATE", "7"))

# psycopg accepts connection settings as keyword arguments. Keeping the DSN as a
# dictionary makes it easy to reuse for the readiness probe and per-batch insert.
PG_DSN = {
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "monitor"),
    "password": os.environ.get("POSTGRES_PASSWORD", "monitor"),
    "dbname": os.environ.get("POSTGRES_DB", "monitoring"),
}

# Features the model sees. These names and broad dtypes must stay aligned with
# project/schemas/schemas.py::CustomerData; otherwise /predict/batch rejects the
# payload before the model can score it.
NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]
CATEGORICAL_FEATURES = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Columns in the CSV that are NOT part of the model input and must be dropped
# before POSTing to /predict/batch (id is a row identifier, gender is unused,
# Churn is the label).
NON_FEATURE_COLUMNS = ["id", "gender", "Churn"]

# ``churn_probability`` is added after the API call. ``Churn`` comes from the
# labeled training data and is converted from Yes/No into 1/0 before metric
# calculation.
PREDICTION_COL = "churn_probability"
TARGET_COL = "Churn"

# Single-row insert for the dashboard table created by monitoring/init.sql.
# Parameter binding keeps values type-safe and avoids string interpolation for
# nullable metrics such as test-set accuracy, which is unavailable without labels.
INSERT_SQL = """
INSERT INTO monitoring_metrics (
    data_source, batch_id, batch_size,
    num_drifted_columns, share_drifted_columns, prediction_drift,
    share_missing_values, mean_predicted_churn_prob, churn_rate,
    accuracy, roc_auc, log_loss, drift_intensity
) VALUES (
    %(data_source)s, %(batch_id)s, %(batch_size)s,
    %(num_drifted_columns)s, %(share_drifted_columns)s, %(prediction_drift)s,
    %(share_missing_values)s, %(mean_predicted_churn_prob)s, %(churn_rate)s,
    %(accuracy)s, %(roc_auc)s, %(log_loss)s, %(drift_intensity)s
);
"""

# Idempotent migration applied at startup so existing pgdata volumes (created
# before drift_intensity existed) get the column without manual intervention.
MIGRATION_SQL = """
ALTER TABLE monitoring_metrics
    ADD COLUMN IF NOT EXISTS drift_intensity DOUBLE PRECISION NOT NULL DEFAULT 0;
"""


# ── Helpers ──────────────────────────────────────────────────────
@dataclass
class Pool:
    """Named source of replay data that can be sliced into fixed-size batches.

    ``has_labels`` tells the metric layer whether it can compute supervised
    metrics such as accuracy, ROC AUC, and log loss. The holdout slice from
    train.csv has labels; test.csv usually does not.
    
    A simulation pool - a dataframe sliced into fixed-size batches.
    """

    name: str
    frame: pd.DataFrame
    has_labels: bool

    def batches(self, batch_size: int):
        """Yield ``(batch_id, batch_frame)`` pairs without mutating the pool."""
        for i in range(0, len(self.frame), batch_size):
            yield i // batch_size, self.frame.iloc[i : i + batch_size].copy()


def _to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only API input columns, normalized to the schema's expected types.

    The raw CSVs contain identifiers and labels that are useful for evaluation
    but must not be sent to the prediction endpoint. This function gives both
    the reference scoring step and the replay loop one shared definition of the
    payload shape.
    """
    cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    out = df[cols].copy()
    # TotalCharges/MonthlyCharges must be numeric. Some Telco variants contain
    # whitespace for brand-new customers (tenure == 0) — coerce those to 0.0
    # so Pydantic's float validator accepts the payload.
    for col in ("MonthlyCharges", "TotalCharges"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "tenure" in out.columns:
        out["tenure"] = out["tenure"].astype(int)
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    return out


# ── Synthetic drift injection ────────────────────────────────────
class DriftInjector:
    """Optionally perturb a batch so Evidently's drift metrics light up.

    Intensity is computed from a *global* batch counter (across pools) so the
    requested DRIFT_MODE behaves predictably even though the simulator iterates
    multiple pools per pass. ``intensity`` is exposed so the caller can persist
    it next to the drift score it produced — that's what makes the Grafana
    "Injected Drift Intensity" panel line up with rises in Evidently metrics.
    """

    def __init__(
        self,
        mode: str,
        *,
        start_batch: int,
        ramp_batches: int,
        period_batches: int,
        numeric_shift_std: float,
        categorical_swap_prob: float,
        missing_rate: float,
        reference: pd.DataFrame,
        seed: int,
    ) -> None:
        self.mode = mode
        self.start_batch = start_batch
        self.ramp_batches = ramp_batches
        self.period_batches = period_batches
        self.numeric_shift_std = numeric_shift_std
        self.categorical_swap_prob = categorical_swap_prob
        self.missing_rate = missing_rate
        self._rng = np.random.default_rng(seed)

        # Cache reference distribution stats so every batch is shifted relative
        # to the same baseline. Computing per-batch stats would let drift "drag
        # itself" and produce inconsistent shifts across pools.
        self._numeric_std = {
            col: float(reference[col].astype(float).std(ddof=0) or 1.0)
            for col in NUMERICAL_FEATURES
            if col != "SeniorCitizen" and col in reference.columns
        }
        self._categorical_choices = {
            col: reference[col].dropna().unique().tolist()
            for col in CATEGORICAL_FEATURES
            if col in reference.columns
        }

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def intensity_for(self, global_batch: int) -> float:
        """Return drift intensity ∈ [0, 1] for the given global batch index."""
        if self.mode == "none":
            return 0.0
        if global_batch < self.start_batch:
            return 0.0
        offset = global_batch - self.start_batch
        if self.mode == "sustained":
            return 1.0
        if self.mode == "gradual":
            return min(1.0, (offset + 1) / self.ramp_batches)
        if self.mode == "cycle":
            half = max(1, self.period_batches // 2)
            phase = (offset // half) % 2
            return 1.0 if phase == 1 else 0.0
        # Unknown mode — be silent and effectively disabled.
        return 0.0

    def perturb_features(self, batch: pd.DataFrame, intensity: float) -> pd.DataFrame:
        """Shift numeric distributions and resample categoricals in-place-safe.

        The output is always schema-valid for the API: numerics stay ≥ 0, tenure
        and SeniorCitizen stay integral, and categoricals only take values that
        were observed in the reference set.
        """
        if intensity <= 0:
            return batch
        out = batch.copy()
        n = len(out)

        for col, std in self._numeric_std.items():
            if col not in out.columns:
                continue
            shift = intensity * self.numeric_shift_std * std
            # Add a Gaussian centered on the shift so the *mean* moves while the
            # *spread* changes only modestly — closer to a real-world covariate
            # shift than a pure constant offset.
            noise = self._rng.normal(loc=shift, scale=0.5 * std, size=n)
            new = pd.to_numeric(out[col], errors="coerce").fillna(0.0) + noise
            new = new.clip(lower=0.0)
            if col == "tenure":
                new = new.round().astype(int)
            out[col] = new

        # SeniorCitizen is Literal[0, 1]; flip a fraction of rows instead of
        # shifting it numerically so the payload stays schema-valid.
        if "SeniorCitizen" in out.columns:
            flip_prob = min(0.5, intensity * 0.4)
            if flip_prob > 0:
                mask = self._rng.random(n) < flip_prob
                if mask.any():
                    flipped = 1 - out.loc[mask, "SeniorCitizen"].astype(int)
                    out.loc[mask, "SeniorCitizen"] = flipped

        swap_prob = min(1.0, intensity * self.categorical_swap_prob)
        if swap_prob > 0:
            for col, choices in self._categorical_choices.items():
                if col not in out.columns or not choices:
                    continue
                mask = self._rng.random(n) < swap_prob
                k = int(mask.sum())
                if k:
                    sampled = self._rng.choice(choices, size=k, replace=True)
                    out.loc[mask, col] = sampled

        return out

    def perturb_evaluation_frame(
        self, frame: pd.DataFrame, intensity: float
    ) -> pd.DataFrame:
        """Inject NaNs post-prediction so DatasetMissingValuesMetric responds.

        We blank cells *after* scoring so the API still sees a clean payload —
        otherwise Pydantic would reject the batch before the model ever runs.
        """
        rate = intensity * self.missing_rate
        if rate <= 0:
            return frame
        out = frame.copy()
        feature_cols = [c for c in FEATURE_COLUMNS if c in out.columns]
        for col in feature_cols:
            mask = self._rng.random(len(out)) < rate
            if mask.any():
                out.loc[mask, col] = np.nan
        return out


def ensure_schema(dsn: dict) -> None:
    """Apply idempotent schema migrations needed for newer monitor features."""
    with psycopg.connect(**dsn) as conn, conn.cursor() as cur:
        cur.execute(MIGRATION_SQL)
        conn.commit()


def wait_for_api(url: str, timeout: int) -> None:
    """Block until the FastAPI root endpoint reports that the model is loaded."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # The API root returns JSON that includes a ``loaded`` flag. A plain
            # HTTP 200 is not enough because the web process may be up before it
            # has loaded the MLflow model artifact.
            r = requests.get(url + "/", timeout=5)
            if r.ok and r.json().get("loaded"):
                log.info("api is up and model is loaded")
                return
        except Exception as exc:  # noqa: BLE001 - best-effort probe
            log.info("waiting for api at %s (%s)", url, exc)
        time.sleep(3)
    raise RuntimeError(f"api at {url} never became healthy")


def wait_for_postgres(dsn: dict, timeout: int = 120) -> None:
    """Block until Postgres accepts connections and can execute a trivial query."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with psycopg.connect(**dsn, connect_timeout=3) as conn:
                conn.execute("SELECT 1;")
            log.info("postgres is reachable")
            return
        except Exception as exc:  # noqa: BLE001
            log.info("waiting for postgres (%s)", exc)
        time.sleep(2)
    raise RuntimeError("postgres never became reachable")


def predict(features: pd.DataFrame) -> np.ndarray:
    """POST rows to /predict/batch and return probabilities in batch order.

    The monitor deliberately calls the public API instead of importing model
    code. That keeps monitoring faithful to the deployed service contract and
    catches serving-time schema or model-loading problems.
    """
    records = features.to_dict(orient="records")
    resp = requests.post(f"{API_URL}/predict/batch", json=records, timeout=120)
    resp.raise_for_status()
    preds = resp.json()["predictions"]
    return np.asarray([p["churn_probability"] for p in preds], dtype=float)


# ── Reference / current dataframe shaping ────────────────────────
def build_reference(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split train.csv into (reference, simulation pool).

    The reference frame represents the known-good baseline distribution used by
    Evidently. Stratification keeps the churn/non-churn class balance stable in
    the reference slice, which makes downstream drift and quality metrics less
    noisy for small datasets.

    The returned reference keeps labels and receives a ``churn_probability``
    column from the live model. The returned simulation pool keeps labels too,
    but it is scored later batch-by-batch to mimic production traffic.
    """
    log.info("loaded %d train rows; splitting %.0f%% as reference",
             len(train_df), REFERENCE_FRAC * 100)
    ref_df, sim_df = train_test_split(
        train_df,
        test_size=1 - REFERENCE_FRAC,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COL],
    )
    ref_df = ref_df.reset_index(drop=True)
    sim_df = sim_df.reset_index(drop=True)

    log.info("scoring reference (%d rows) via api", len(ref_df))
    ref_df[PREDICTION_COL] = predict(_to_features(ref_df))
    ref_df[TARGET_COL] = ref_df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    return ref_df, sim_df


def prepare_pools(sim_train: pd.DataFrame, test_df: pd.DataFrame) -> list[Pool]:
    """Create the labeled and unlabeled replay pools consumed by the main loop."""
    sim_train = sim_train.copy()
    sim_train[TARGET_COL] = sim_train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    return [
        Pool(name="train_holdout", frame=sim_train, has_labels=True),
        Pool(name="test", frame=test_df.copy(), has_labels=False),
    ]


# ── Metric extraction ────────────────────────────────────────────
def _column_mapping(has_target: bool) -> ColumnMapping:
    """Describe dataframe semantics for Evidently's metric calculations."""
    return ColumnMapping(
        target=TARGET_COL if has_target else None,
        prediction=PREDICTION_COL,
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
    )


def compute_metrics(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    has_labels: bool,
) -> dict:
    """Run an Evidently Report and collapse nested results into dashboard fields.

    Evidently returns a rich nested report. Grafana is easier to build against a
    narrow relational table, so this function extracts only the metrics used by
    the dashboard and stores unavailable supervised metrics as ``None``.
    """
    mapping = _column_mapping(has_target=has_labels)

    # The report combines dataset-level drift, missing-value share, and a
    # dedicated drift score for the model's prediction column. The prediction
    # drift panel is useful because feature distributions can look stable while
    # the resulting score distribution changes.
    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name=PREDICTION_COL),
        ]
    )
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=mapping,
    )
    result = report.as_dict()

    # Pull by metric name so ordering changes don't break us.
    by_name = {m["metric"]: m["result"] for m in result["metrics"]}

    dataset_drift = by_name.get("DatasetDriftMetric", {}) or {}
    missing = by_name.get("DatasetMissingValuesMetric", {}) or {}
    pred_drift = by_name.get("ColumnDriftMetric", {}) or {}

    missing_share = (
        (missing.get("current") or {}).get("share_of_missing_values")
        if isinstance(missing.get("current"), dict)
        else missing.get("share_of_missing_values")
    )

    metrics = {
        "num_drifted_columns": dataset_drift.get("number_of_drifted_columns"),
        "share_drifted_columns": dataset_drift.get("share_of_drifted_columns"),
        "prediction_drift": pred_drift.get("drift_score"),
        "share_missing_values": missing_share,
        "mean_predicted_churn_prob": float(current[PREDICTION_COL].mean()),
        "churn_rate": None,
        "accuracy": None,
        "roc_auc": None,
        "log_loss": None,
    }

    if has_labels:
        # Supervised metrics are only valid for the labeled holdout replay. The
        # unlabeled test replay still contributes drift, missingness, and average
        # predicted churn probability.
        y_true = current[TARGET_COL].astype(int).to_numpy()
        y_prob = current[PREDICTION_COL].astype(float).to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        metrics["churn_rate"] = float(y_true.mean())
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        if len(np.unique(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        # log_loss needs probabilities clipped away from 0/1
        clipped = np.clip(y_prob, 1e-6, 1 - 1e-6)
        metrics["log_loss"] = float(log_loss(y_true, clipped, labels=[0, 1]))

    return metrics


# ── Main loop ────────────────────────────────────────────────────
def run() -> None:
    """Coordinate dependency readiness, data loading, scoring, and persistence."""
    log.info("monitor starting; api=%s pg=%s batch_size=%d interval=%ds",
             API_URL, PG_DSN["host"], BATCH_SIZE, INTERVAL_SECONDS)
    wait_for_postgres(PG_DSN)
    ensure_schema(PG_DSN)
    wait_for_api(API_URL, API_WAIT_SECONDS)

    # Load the same CSV layout used by training. The id column is not predictive
    # and is dropped early so all later feature selections operate on stable
    # column names regardless of whether the CSV includes row identifiers.
    train_df = pd.read_csv(DATASET_DIR / "train.csv")
    test_df = pd.read_csv(DATASET_DIR / "test.csv")
    train_df = train_df.drop(columns=[c for c in ("id",) if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in ("id",) if c in test_df.columns])

    reference, sim_train = build_reference(train_df)
    pools = prepare_pools(sim_train, test_df)

    # Keep a compact reference dataframe with only Evidently-relevant columns.
    # This avoids accidentally including raw identifier or unused columns in
    # drift calculations.
    ref_for_eval = reference[FEATURE_COLUMNS + [TARGET_COL, PREDICTION_COL]]

    drift_injector = DriftInjector(
        mode=DRIFT_MODE,
        start_batch=DRIFT_START_BATCH,
        ramp_batches=DRIFT_RAMP_BATCHES,
        period_batches=DRIFT_PERIOD_BATCHES,
        numeric_shift_std=DRIFT_NUMERIC_SHIFT_STD,
        categorical_swap_prob=DRIFT_CATEGORICAL_SWAP_PROB,
        missing_rate=DRIFT_MISSING_RATE,
        reference=reference,
        seed=DRIFT_RANDOM_STATE,
    )
    if drift_injector.enabled:
        log.info(
            "drift injection enabled: mode=%s start=%d ramp=%d period=%d "
            "shift_std=%.2f swap_prob=%.2f missing_rate=%.2f",
            DRIFT_MODE, DRIFT_START_BATCH, DRIFT_RAMP_BATCHES,
            DRIFT_PERIOD_BATCHES, DRIFT_NUMERIC_SHIFT_STD,
            DRIFT_CATEGORICAL_SWAP_PROB, DRIFT_MISSING_RATE,
        )

    # Counts every batch processed across pools and passes so DRIFT_MODE
    # behaviors (gradual ramps, cycles) advance smoothly even though we iterate
    # multiple pools per pass.
    global_batch = 0

    while True:
        produced = 0
        for pool in pools:
            for batch_id, batch in pool.batches(BATCH_SIZE):
                intensity = drift_injector.intensity_for(global_batch)
                global_batch += 1

                # Apply synthetic drift to the *raw* batch before scoring so
                # the deployed model genuinely reacts to the shifted input.
                # That way prediction drift, accuracy, and ROC AUC all move
                # together — the Grafana dashboard tells a coherent story.
                drifted_batch = drift_injector.perturb_features(batch, intensity)

                # Score one batch through the API, then stitch predictions back
                # onto the raw batch so feature, target, and prediction columns
                # remain aligned by row.
                features = _to_features(drifted_batch)
                try:
                    probs = predict(features)
                except Exception as exc:  # noqa: BLE001
                    log.error("prediction failed for %s batch %d: %s",
                              pool.name, batch_id, exc)
                    continue

                current = drifted_batch.copy()
                current[PREDICTION_COL] = probs
                if pool.has_labels:
                    # Labeled batches can be evaluated against a reference that
                    # includes the target column, enabling supervised metrics.
                    # Labels come from the *original* batch — drift perturbs
                    # features, but the ground truth doesn't move.
                    current[TARGET_COL] = batch[TARGET_COL].astype(int)
                    cur_for_eval = current[FEATURE_COLUMNS + [TARGET_COL, PREDICTION_COL]]
                    ref_slice = ref_for_eval
                else:
                    # Reference frame for test pool intentionally drops the target
                    # so Evidently doesn't try to compute target drift.
                    ref_slice = ref_for_eval.drop(columns=[TARGET_COL])
                    cur_for_eval = current[FEATURE_COLUMNS + [PREDICTION_COL]]

                # Inject missingness *after* scoring so the API never receives
                # NaNs (Pydantic would reject them) but Evidently still picks
                # up an elevated share-of-missing-values.
                cur_for_eval = drift_injector.perturb_evaluation_frame(
                    cur_for_eval, intensity
                )

                metrics = compute_metrics(ref_slice, cur_for_eval, pool.has_labels)
                metrics.update(
                    data_source=pool.name,
                    batch_id=batch_id,
                    batch_size=len(batch),
                    drift_intensity=float(intensity),
                )

                # Open a short-lived connection per batch. For this low-volume
                # simulator that keeps failure handling simple and avoids stale
                # connections if Postgres restarts during a replay.
                with psycopg.connect(**PG_DSN) as conn, conn.cursor() as cur:
                    cur.execute(INSERT_SQL, metrics)
                    conn.commit()

                log.info(
                    "wrote %s/%d drifted=%s share=%.3f pred_drift=%.4f acc=%s injected=%.2f",
                    pool.name,
                    batch_id,
                    metrics.get("num_drifted_columns"),
                    metrics.get("share_drifted_columns") or 0.0,
                    metrics.get("prediction_drift") or 0.0,
                    metrics.get("accuracy"),
                    intensity,
                )
                produced += 1
                time.sleep(INTERVAL_SECONDS)

        log.info("pass complete (%d batches produced)", produced)
        # By default the monitor exits after one full replay, which is convenient
        # for repeatable coursework/demo runs. LOOP_FOREVER turns the same code
        # into a simple ongoing data generator.
        if RUN_ONCE:
            return
        if not LOOP_FOREVER:
            return
        log.info("LOOP_FOREVER=1 — sleeping before replaying pools")
        time.sleep(INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log.exception("monitor crashed")
        raise
