# Monitoring pipeline

This document describes the drift + performance monitoring stack that ships
alongside the churn API.

## Overview

```mermaid
flowchart LR
  TrainCSV["dataset/train.csv"] --> Monitor
  TestCSV["dataset/test.csv"] --> Monitor
  Monitor["monitor (cron-style loop)"] -->|"POST /predict/batch"| API["api (FastAPI, :8000)"]
  API --> Monitor
  Monitor -->|INSERT| Postgres[("postgres :5432")]
  Postgres --> Adminer["adminer :8080"]
  Postgres --> Grafana["grafana :3000"]
  API -->|champion model| MLflow["mlflow :5001"]
```

All services live in [`project/docker-compose.yml`](../docker-compose.yml) and
spin up together with `docker compose up --build`.

## What the monitor does

The script [`monitor.py`](../monitoring/monitor.py) runs inside its own
container and, on every tick:

1. Reads the next `BATCH_SIZE` rows from each simulation pool:
   - **train_holdout** — 80% of `dataset/train.csv` (stratified by `Churn`);
     ground-truth labels are available so classification metrics are reported.
   - **test** — all of `dataset/test.csv`; unlabelled, so only drift and
     data-quality metrics are reported.
2. POSTs the batch to `http://api:8000/predict/batch` and attaches the
   returned `churn_probability` to the dataframe.
3. Builds an [Evidently](https://docs.evidentlyai.com/) `Report` with:
   - `DatasetDriftMetric` — number/share of drifted features.
   - `DatasetMissingValuesMetric` — share of missing values.
   - `ColumnDriftMetric("churn_probability")` — prediction drift (PSI / EMD
     depending on sample size).
4. Computes `accuracy`, `roc_auc`, and `log_loss` with scikit-learn when
   labels are available, plus the realised `churn_rate` and
   `mean_predicted_churn_prob`.
5. Writes one row into the Postgres table `monitoring_metrics` (schema in
   [`monitoring/init.sql`](../monitoring/init.sql)).

## Services

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| `postgres` | `postgres:16-alpine` | `5432` | Metrics store (`monitoring` DB) |
| `adminer` | `adminer:5` | `8080` | Web UI to browse Postgres |
| `grafana` | `grafana/grafana-oss:11.3.0` | `3000` | Dashboards |
| `monitor` | built from `monitoring/Dockerfile.monitor` | — | Runs the loop |

Grafana is provisioned automatically from
[`monitoring/grafana/provisioning`](../monitoring/grafana/provisioning) and
ships with the **Customer Churn Monitoring** dashboard
([`churn_monitoring.json`](../monitoring/grafana/dashboards/churn_monitoring.json)).

## Running

Prerequisite: the champion model must already be registered in MLflow (see
[TRAIN.md](TRAIN.md) — run `uv run python project/train.py` against a running
MLflow server first).

```bash
cd project
docker compose up --build
```

Once everything is up:

- Grafana: <http://localhost:3000> — anonymous viewer access is enabled; log in
  as `admin` / `admin` to edit.
- Adminer: <http://localhost:8080> — system `PostgreSQL`, server `postgres`,
  user `monitor`, password `monitor`, database `monitoring`.
- Postgres (direct): `psql postgresql://monitor:monitor@localhost:5432/monitoring`.

A fresh Postgres volume will run `init.sql` automatically and create the
`monitoring_metrics` table. If you want to reset the metrics, drop the named
volume:

```bash
docker compose down
docker volume rm project_pgdata
docker compose up --build
```

## Configuring the monitor

All settings are environment variables on the `monitor` service in
`docker-compose.yml`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `API_URL` | `http://api:8000` | Base URL of the FastAPI service |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `postgres` / `5432` | Postgres location |
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `monitor` / `monitor` / `monitoring` | Postgres credentials |
| `BATCH_SIZE` | `1000` | Rows per simulated batch |
| `INTERVAL_SECONDS` | `10` | Sleep between batches (cron-style cadence) |
| `REFERENCE_FRAC` | `0.2` | Share of train.csv reserved as Evidently reference |
| `RUN_ONCE` | `0` | If `1`, exit after a single pass through both pools |
| `LOOP_FOREVER` | `0` | If `1`, replay pools indefinitely (ignored when `RUN_ONCE=1`) |
| `RANDOM_STATE` | `42` | Seed for the stratified reference split |
| `API_WAIT_SECONDS` | `180` | How long to wait for the API health check on startup |

## Schema

```sql
CREATE TABLE monitoring_metrics (
    id                         SERIAL PRIMARY KEY,
    ts                         TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    data_source                TEXT             NOT NULL,  -- 'train_holdout' | 'test'
    batch_id                   INT              NOT NULL,
    batch_size                 INT,
    num_drifted_columns        INT,
    share_drifted_columns      DOUBLE PRECISION,
    prediction_drift           DOUBLE PRECISION,
    share_missing_values       DOUBLE PRECISION,
    mean_predicted_churn_prob  DOUBLE PRECISION,
    churn_rate                 DOUBLE PRECISION,
    accuracy                   DOUBLE PRECISION,
    roc_auc                    DOUBLE PRECISION,
    log_loss                   DOUBLE PRECISION
);
```

Classification metrics (`accuracy`, `roc_auc`, `log_loss`, `churn_rate`) are
always `NULL` for `data_source = 'test'` because `test.csv` has no label.

## Dashboard panels

1. **Stat row** — latest ROC AUC, accuracy, number of drifted columns, total
   batches scored.
2. **Number of Drifted Columns** (time series, per `data_source`).
3. **Share of Drifted Columns** (time series, red threshold at 0.5).
4. **Prediction Drift** for `churn_probability` (time series).
5. **Share of Missing Values** (time series).
6. **Predicted vs. Actual Churn Rate** — model calibration on the labelled
   train-holdout pool.
7. **Model Quality** — accuracy / ROC AUC / log loss trends on train-holdout.
8. **Recent Batches** — the last 20 rows of `monitoring_metrics`.
