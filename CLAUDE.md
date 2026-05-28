# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project scope

The repository is HSLU AI coursework, but the only meaningful runtime lives under `project/` — a Customer Churn Prediction stack (LightGBM + XGBoost CV ensemble, Optuna HPO, MLflow tracking/registry, FastAPI serving, Evidently-based monitoring). Repo-root `main.py` is a placeholder; `dashboard.html` is a standalone static page unrelated to the Python runtime.

Python 3.13+, managed with `uv` (`pyproject.toml` + `uv.lock`).

## Commands

Install: `uv sync` (add `--group dev` for test/lint deps).

Lint / format:
```
uv run ruff check .
uv run ruff format .          # or `ruff format --check .` to mirror CI
```

Tests (CI invocation — required `PYTHONPATH=project`, see "Import layout" below):
```
PYTHONPATH=project uv run pytest project/tests
```
Single test: `uv run pytest project/tests/test_web_service.py::test_predict_returns_probability`
With coverage as CI does: `uv run pytest project/tests --cov=project --cov-report=term --cov-fail-under=60`

Train (requires MLflow reachable at `http://localhost:5001`, which `train.py` hardcodes):
```
uv run python project/train.py
```
Outputs: registers `CustomerChurnEnsemble` (alias `champion` once promoted), `project/submission.csv`, `project/optuna_studies.db`.

API locally — must run with `project/` on the path because of how schemas are imported:
```
cd project && uv run uvicorn web_service:app --reload --host 0.0.0.0 --port 8000
# or:  PYTHONPATH=project uv run uvicorn web_service:app --reload ...
```
Set `MLFLOW_TRACKING_URI` if MLflow isn't at the default for your environment. The app loads `models:/CustomerChurnEnsemble@champion` at startup, so train against the same registry first.

Full local stack (API + MLflow + Postgres + Adminer + Grafana + monitor replay):
```
cd project && docker compose up --build
```
Ports: API `8000`, MLflow `5001`, Grafana `3000`, Adminer `8080`, Postgres host `5431` → container `5432`.

## Architecture

Three loosely-coupled pieces glued together by the MLflow registry and a Postgres metrics table:

1. **Training** (`project/train.py`) — Loads `project/dataset/{train,test}.csv` (gitignored), runs feature engineering, Optuna HPO, then 5-fold stratified CV that trains both LightGBM and XGBoost on each fold. The 10 fold models are wrapped as a single `pyfunc` model (`CustomerChurnEnsemble`) and registered to MLflow. Optuna studies persist to `project/optuna_studies.db`. Deep dive: `project/docs/TRAIN.md`.

2. **Serving** (`project/web_service.py`) — FastAPI app. On startup it pulls `CustomerChurnEnsemble@champion` from the registry; `/predict` and `/predict/batch` re-apply the *same* preprocessing as training before calling the pyfunc model. Pydantic schemas live in `project/schemas/schemas.py`. Endpoints: `GET /`, `POST /predict`, `POST /predict/batch`, `GET /model/info`. Deep dive: `project/docs/WEB_SERVICE.md`.

3. **Monitoring** (`project/monitoring/monitor.py`) — A replay worker that streams `dataset/train.csv` (stratified 80/20 reference split) and `dataset/test.csv` through the API's `/predict/batch`, computes Evidently drift + classification metrics, and writes one row per batch to the `monitoring_metrics` Postgres table that Grafana panels read. Schema in `monitoring/init.sql`. Synthetic drift injection is configurable via `DRIFT_*` env vars on the `monitor` service in `docker-compose.yml` (off by default; modes like `gradual`/`periodic`). Deep dive: `project/docs/MONITORING.md`.

Inside the Compose network, services address each other by service name (e.g. the API uses `MLFLOW_TRACKING_URI=http://mlflow:5001`), so URIs differ between host-run and container-run code.

## Import layout (gotcha)

`web_service.py` and tests import via `from schemas.schemas import ...`, which only resolves when `project/` is on `sys.path`. This is wired in three places — keep them consistent if you restructure:

- `pyproject.toml` → `[tool.pytest.ini_options] pythonpath = ["project"]`
- `project/tests/conftest.py` re-adds it as a safety net for bare `pytest` invocations
- CI sets `PYTHONPATH=project` explicitly before running pytest

Coverage is configured to omit `project/train.py` and `project/monitoring/grafana/*` (see `[tool.coverage.run]`).

## CI

GitHub Actions in `.github/workflows/`: `ci.yml` (ruff + pytest with `--cov-fail-under=50` on Python 3.13), plus `docker.yml`, `release.yml`, `secret-scan.yml`, `train.yml`. CodeQL is enabled via GitHub's *default setup* (Settings → Code security → Code scanning), not via a workflow file. The lint job runs `ruff format --check` with `continue-on-error: true`, but `ruff check` is enforcing.

## Data

`project/dataset/` is gitignored. Place `train.csv` and `test.csv` there before training. The Compose `monitor` service expects the same files mounted into the container.
