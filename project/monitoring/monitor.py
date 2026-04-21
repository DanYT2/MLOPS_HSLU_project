"""Monitoring pipeline for the Customer Churn FastAPI model.

Replays the training and test CSVs as simulated production batches, asks the
FastAPI service for predictions, runs Evidently drift + quality checks against
a held-out reference batch, and writes one row per batch into Postgres. The
Grafana dashboard (provisioned in ``monitoring/grafana``) reads straight from
that table.

The loop behaves like a lightweight cron: it ticks every ``INTERVAL_SECONDS``
until every batch from both simulation pools has been consumed (or forever, if
``RUN_ONCE=0`` and ``LOOP_FOREVER=1``).
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
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("monitor")

# ── Config ───────────────────────────────────────────────────────
DATASET_DIR = Path(os.environ.get("DATASET_DIR", "/app/dataset"))
API_URL = os.environ.get("API_URL", "http://api:8000").rstrip("/")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1000"))
INTERVAL_SECONDS = int(os.environ.get("INTERVAL_SECONDS", "10"))
REFERENCE_FRAC = float(os.environ.get("REFERENCE_FRAC", "0.2"))
RUN_ONCE = os.environ.get("RUN_ONCE", "0") == "1"
LOOP_FOREVER = os.environ.get("LOOP_FOREVER", "0") == "1"
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))
API_WAIT_SECONDS = int(os.environ.get("API_WAIT_SECONDS", "180"))

PG_DSN = {
    "host": os.environ.get("POSTGRES_HOST", "postgres"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "monitor"),
    "password": os.environ.get("POSTGRES_PASSWORD", "monitor"),
    "dbname": os.environ.get("POSTGRES_DB", "monitoring"),
}

# Features the model sees. Must match project/schemas/schemas.py::CustomerData.
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

PREDICTION_COL = "churn_probability"
TARGET_COL = "Churn"

INSERT_SQL = """
INSERT INTO monitoring_metrics (
    data_source, batch_id, batch_size,
    num_drifted_columns, share_drifted_columns, prediction_drift,
    share_missing_values, mean_predicted_churn_prob, churn_rate,
    accuracy, roc_auc, log_loss
) VALUES (
    %(data_source)s, %(batch_id)s, %(batch_size)s,
    %(num_drifted_columns)s, %(share_drifted_columns)s, %(prediction_drift)s,
    %(share_missing_values)s, %(mean_predicted_churn_prob)s, %(churn_rate)s,
    %(accuracy)s, %(roc_auc)s, %(log_loss)s
);
"""


# ── Helpers ──────────────────────────────────────────────────────
@dataclass
class Pool:
    """A simulation pool — a dataframe sliced into fixed-size batches."""

    name: str
    frame: pd.DataFrame
    has_labels: bool

    def batches(self, batch_size: int):
        for i in range(0, len(self.frame), batch_size):
            yield i // batch_size, self.frame.iloc[i : i + batch_size].copy()


def _to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Strip id/gender/Churn and cast numeric columns so the API validates."""
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


def wait_for_api(url: str, timeout: int) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url + "/", timeout=5)
            if r.ok and r.json().get("loaded"):
                log.info("api is up and model is loaded")
                return
        except Exception as exc:  # noqa: BLE001 - best-effort probe
            log.info("waiting for api at %s (%s)", url, exc)
        time.sleep(3)
    raise RuntimeError(f"api at {url} never became healthy")


def wait_for_postgres(dsn: dict, timeout: int = 120) -> None:
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
    """POST rows to /predict/batch and return a numpy array of probabilities."""
    records = features.to_dict(orient="records")
    resp = requests.post(f"{API_URL}/predict/batch", json=records, timeout=120)
    resp.raise_for_status()
    preds = resp.json()["predictions"]
    return np.asarray([p["churn_probability"] for p in preds], dtype=float)


# ── Reference / current dataframe shaping ────────────────────────
def build_reference(train_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split train.csv into (reference, simulation pool).

    Both frames keep the Churn label and get a ``churn_probability`` column
    filled from the live model.
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
    sim_train = sim_train.copy()
    sim_train[TARGET_COL] = sim_train[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)
    return [
        Pool(name="train_holdout", frame=sim_train, has_labels=True),
        Pool(name="test", frame=test_df.copy(), has_labels=False),
    ]


# ── Metric extraction ────────────────────────────────────────────
def _column_mapping(has_target: bool) -> ColumnMapping:
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
    """Run an Evidently Report and collapse it into flat scalar metrics."""
    mapping = _column_mapping(has_target=has_labels)

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
    log.info("monitor starting; api=%s pg=%s batch_size=%d interval=%ds",
             API_URL, PG_DSN["host"], BATCH_SIZE, INTERVAL_SECONDS)
    wait_for_postgres(PG_DSN)
    wait_for_api(API_URL, API_WAIT_SECONDS)

    train_df = pd.read_csv(DATASET_DIR / "train.csv")
    test_df = pd.read_csv(DATASET_DIR / "test.csv")
    train_df = train_df.drop(columns=[c for c in ("id",) if c in train_df.columns])
    test_df = test_df.drop(columns=[c for c in ("id",) if c in test_df.columns])

    reference, sim_train = build_reference(train_df)
    pools = prepare_pools(sim_train, test_df)

    ref_for_eval = reference[FEATURE_COLUMNS + [TARGET_COL, PREDICTION_COL]]

    while True:
        produced = 0
        for pool in pools:
            for batch_id, batch in pool.batches(BATCH_SIZE):
                features = _to_features(batch)
                try:
                    probs = predict(features)
                except Exception as exc:  # noqa: BLE001
                    log.error("prediction failed for %s batch %d: %s",
                              pool.name, batch_id, exc)
                    continue

                current = batch.copy()
                current[PREDICTION_COL] = probs
                if pool.has_labels:
                    current[TARGET_COL] = batch[TARGET_COL].astype(int)
                    cur_for_eval = current[FEATURE_COLUMNS + [TARGET_COL, PREDICTION_COL]]
                    ref_slice = ref_for_eval
                else:
                    # Reference frame for test pool intentionally drops the target
                    # so Evidently doesn't try to compute target drift.
                    ref_slice = ref_for_eval.drop(columns=[TARGET_COL])
                    cur_for_eval = current[FEATURE_COLUMNS + [PREDICTION_COL]]

                metrics = compute_metrics(ref_slice, cur_for_eval, pool.has_labels)
                metrics.update(
                    data_source=pool.name,
                    batch_id=batch_id,
                    batch_size=len(batch),
                )

                with psycopg.connect(**PG_DSN) as conn, conn.cursor() as cur:
                    cur.execute(INSERT_SQL, metrics)
                    conn.commit()

                log.info(
                    "wrote %s/%d drifted=%s share=%.3f pred_drift=%.4f acc=%s",
                    pool.name,
                    batch_id,
                    metrics.get("num_drifted_columns"),
                    metrics.get("share_drifted_columns") or 0.0,
                    metrics.get("prediction_drift") or 0.0,
                    metrics.get("accuracy"),
                )
                produced += 1
                time.sleep(INTERVAL_SECONDS)

        log.info("pass complete (%d batches produced)", produced)
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
