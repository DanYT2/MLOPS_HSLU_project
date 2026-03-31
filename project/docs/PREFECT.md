# Prefect Orchestration

Prefect manages scheduled retraining and batch prediction for the customer
churn pipeline. Two flows are defined in `flows.py`:

| Flow | Schedule | What it does |
|------|----------|--------------|
| `training-pipeline` | Sunday 02:00 | Full retrain: load data, Optuna HPO, 5-fold CV ensemble, MLflow registration, champion promotion |
| `batch-prediction` | Monday 06:00 | Load champion model, predict on a CSV, write results |

## Architecture

```
┌──────────────────┐       ┌───────────────┐
│  Prefect Server  │◄──────│ Prefect Worker │
│  (UI + scheduler)│       │ (runs flows)   │
│  :4200           │       └──────┬─────────┘
└──────────────────┘              │
                                  ▼
                         ┌────────────────┐
                         │  MLflow Server  │
                         │  :5001          │
                         └────────────────┘
```

## Quick start (Docker)

Start the full stack (MLflow + API + Prefect):

```bash
cd project
docker compose up --build
```

Services:

| Service | URL |
|---------|-----|
| Prefect UI | http://localhost:4200 |
| MLflow UI | http://localhost:5001 |
| Prediction API | http://localhost:8000/docs |

The worker container (`prefect-worker`) automatically registers both
deployments and starts polling for scheduled runs.

## Quick start (local)

Make sure the MLflow server is running (`docker compose up mlflow` or a local
instance), then start the Prefect server and deployment runner in separate
terminals:

```bash
# Terminal 1 – Prefect server
prefect server start

# Terminal 2 – deployment runner (blocks, executes scheduled flows)
cd project
python deploy.py
```

## Trigger a flow manually

### Via the Prefect UI

1. Open http://localhost:4200
2. Navigate to **Deployments**
3. Click **scheduled-training** or **scheduled-batch-prediction**
4. Click **Run** → **Quick run** (uses default parameters) or **Custom run**
   to override parameters

### Via the CLI

```bash
# Training with default parameters
prefect deployment run 'training-pipeline/scheduled-training'

# Training with custom parameters
prefect deployment run 'training-pipeline/scheduled-training' \
  --param n_trials=20

# Batch prediction
prefect deployment run 'batch-prediction/scheduled-batch-prediction' \
  --param input_path=project/dataset/test.csv \
  --param output_path=project/predictions/output.csv
```

### Via Python

```python
from flows import training_flow, batch_prediction_flow

# Run locally (no server needed)
training_flow(data_dir="project/dataset", n_trials=5)

# Or trigger a deployment run
from prefect.deployments import run_deployment

run_deployment("training-pipeline/scheduled-training", parameters={"n_trials": 20})
```

## Changing schedules

Edit `deploy.py` and modify the `cron` parameter on either deployment:

```python
training_dep = training_flow.to_deployment(
    name="scheduled-training",
    cron="0 2 * * 0",  # ← change this (standard cron syntax)
    ...
)
```

Restart the worker after editing:

```bash
docker compose restart prefect-worker
```

## Monitoring

- **Flow runs**: Prefect UI → Flow Runs shows status, duration, logs, and
  task-level detail for every execution.
- **Artifacts**: Each completed flow creates a Markdown artifact summarising
  key metrics (visible in the Prefect UI under the run's Artifacts tab).
- **MLflow**: Training runs log all CV metrics, plots, and model artifacts to
  MLflow at http://localhost:5001.

## File overview

| File | Purpose |
|------|---------|
| `flows.py` | `@task` wrappers around `train.py` functions + `@flow` definitions |
| `deploy.py` | Creates Prefect deployments with cron schedules and starts serving |
| `Dockerfile.worker` | Container image for the Prefect worker (includes all ML deps) |
| `docker-compose.yml` | Adds `prefect-server` and `prefect-worker` services |
