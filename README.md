# Artificial Intelligence (HSLU)

Coursework and project work for **HSLU Artificial Intelligence**. The main deliverable is a **Customer Churn Prediction** stack under [`project/`](project/): a **LightGBM + XGBoost** ensemble (5-fold stratified cross-validation), **Optuna** hyperparameter optimization, **MLflow** experiment tracking and Model Registry, a **FastAPI** service for real-time inference, and an **Evidently + Postgres + Grafana** monitoring loop — all containerized and wired into a multi-stage GitHub Actions pipeline (lint, tests, image builds, SBOM/provenance, secret scanning, tag-driven releases).

The repository root [`main.py`](main.py) is a minimal placeholder. [`dashboard.html`](dashboard.html) is a standalone static "MLOps blueprint" page (Tailwind CSS, Chart.js); it is not part of the Python runtime.

## MLOps practices at a glance

| Capability | Where it lives |
|------------|----------------|
| Experiment tracking | MLflow runs logged from [`project/train.py`](project/train.py); UI on port `5001` |
| Hyperparameter optimization | Optuna TPE study in `tune_hyperparams()` ([`project/train.py`](project/train.py)); studies persist to `project/optuna_studies.db` |
| Cross-validated ensembling | 5-fold stratified CV over LightGBM + XGBoost, wrapped as a single MLflow `pyfunc` ([`project/train.py`](project/train.py)) |
| Model registry + aliasing | `CustomerChurnEnsemble` registered model with `@champion` alias (MLflow Model Registry) |
| Reproducible environment | [`uv`](https://docs.astral.sh/uv/) + [`pyproject.toml`](pyproject.toml) + frozen [`uv.lock`](uv.lock) |
| Containerization | [`project/Dockerfile.api`](project/Dockerfile.api), [`project/Dockerfile.mlflow`](project/Dockerfile.mlflow), [`project/monitoring/Dockerfile.monitor`](project/monitoring/Dockerfile.monitor) |
| Service orchestration | 6-service [`project/docker-compose.yml`](project/docker-compose.yml) with healthchecks and start ordering |
| Online inference API | FastAPI [`project/web_service.py`](project/web_service.py) — `/predict`, `/predict/batch`, `/model/info` |
| Input / output validation | Pydantic schemas in [`project/schemas/schemas.py`](project/schemas/schemas.py) |
| Drift + performance monitoring | Evidently replay worker [`project/monitoring/monitor.py`](project/monitoring/monitor.py) → Postgres → Grafana |
| Synthetic drift simulation | `DRIFT_*` env vars on the `monitor` service in [`project/docker-compose.yml`](project/docker-compose.yml) |
| Test suite + coverage gate | [`project/tests/`](project/tests/) (schemas, web service, monitor); CI enforces `--cov-fail-under=60` |
| Lint / format | [Ruff](https://docs.astral.sh/ruff/) (`ruff check`, `ruff format --check`) |
| CI gating | [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| Image build + push | [`.github/workflows/docker.yml`](.github/workflows/docker.yml) (CycloneDX SBOM + SLSA provenance attestations) |
| Release automation | [`.github/workflows/release.yml`](.github/workflows/release.yml) (GitHub Release notes, SBOM asset, digest-preserving image tag promotion) |
| Manual training in CI | [`.github/workflows/train.yml`](.github/workflows/train.yml) (artifacts: `mlruns/`, `submission.csv`, `optuna_studies.db`) |
| Secret scanning | [`.github/workflows/secret-scan.yml`](.github/workflows/secret-scan.yml) (Gitleaks, SARIF to Code Scanning) |
| Branch protection | Required status checks documented in [`docs/CI_CD.md`](docs/CI_CD.md) |

## Tech stack

| Area | Notes |
|------|--------|
| Runtime | Python **3.13+** (see [`.python-version`](.python-version)) |
| Packaging | [`uv`](https://docs.astral.sh/uv/) with [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) |
| ML | LightGBM, XGBoost, scikit-learn, pandas, NumPy, seaborn, joblib |
| Training | Optuna (HPO), MLflow (tracking + registry) |
| Serving | FastAPI, Pydantic ([`project/schemas/`](project/schemas/schemas.py)) |
| Monitoring | Evidently, Postgres, Grafana, Adminer |
| CI/CD | GitHub Actions, GHCR, CycloneDX SBOM, SLSA provenance, Gitleaks |
| Orchestration | [Prefect](https://www.prefect.io/) is listed as a dependency; the main training path is [`project/train.py`](project/train.py) |

Install dependencies from the repository root:

```bash
uv sync               # runtime deps
uv sync --group dev   # + pytest, pytest-cov, httpx, pip-audit
```

## Repository layout

| Path | Role |
|------|------|
| [`project/train.py`](project/train.py) | End-to-end training: load data → feature engineering → Optuna → CV ensemble → register `CustomerChurnEnsemble` → `submission.csv` |
| [`project/web_service.py`](project/web_service.py) | FastAPI app; loads `models:/CustomerChurnEnsemble@champion` from MLflow |
| [`project/schemas/`](project/schemas/) | Pydantic request and response models |
| [`project/docker-compose.yml`](project/docker-compose.yml) | API on **8000**, MLflow on **5001**, Postgres on **5431→5432**, Adminer on **8080**, Grafana on **3000**, monitor replay worker |
| [`project/monitoring/`](project/monitoring/) | Evidently-based drift + performance monitoring (Dockerfile, Grafana provisioning, Postgres schema) |
| [`project/tests/`](project/tests/) | pytest suite — schemas, web service, monitor |
| [`project/docs/`](project/docs/) | Deep dives: [TRAIN.md](project/docs/TRAIN.md), [WEB_SERVICE.md](project/docs/WEB_SERVICE.md), [SCHEMAS.md](project/docs/SCHEMAS.md), [MONITORING.md](project/docs/MONITORING.md), [TESTS.md](project/docs/TESTS.md) |
| [`docs/CI_CD.md`](docs/CI_CD.md) | CI/CD architecture: workflow matrix, triggers, gates, branch protection, local reproduction |
| [`project/customer-churn-eda-multimodelensemble.ipynb`](project/customer-churn-eda-multimodelensemble.ipynb) | EDA and exploratory modeling |
| [`.github/workflows/`](.github/workflows/) | `ci.yml`, `docker.yml`, `release.yml`, `secret-scan.yml`, `train.yml` |

## Data

Place your course or competition CSVs under **`project/dataset/`**:

- `project/dataset/train.csv`
- `project/dataset/test.csv`

The `dataset/` directory is gitignored ([`.gitignore`](.gitignore)); do not commit raw data. The Compose `monitor` service mounts the same files into its container for replay.

## Architecture overview

```mermaid
flowchart LR
  subgraph training [Training]
    CSV[dataset CSVs]
    Train[train.py]
    Registry[MLflow Registry]
    CSV --> Train
    Train --> Registry
  end
  subgraph serving [Serving]
    Client[HTTP client]
    API[FastAPI]
    Pre[preprocess]
    Model[ChurnEnsemble pyfunc]
    Client --> API
    API --> Pre
    Pre --> Model
    Model --> API
  end
  Registry -.->|champion alias| Model
  subgraph monitoring [Monitoring]
    Monitor[monitor loop]
    PG[(postgres)]
    Adminer[adminer]
    Grafana[grafana]
    Monitor --> PG
    PG --> Adminer
    PG --> Grafana
  end
  API --> Monitor
```

Three loosely-coupled components glued together by the MLflow registry and a Postgres metrics table. Training writes a versioned `CustomerChurnEnsemble`; promotion is a registry alias flip, not a redeploy. Serving pulls `@champion` at startup. Monitoring replays held-out and test data through the live API, scores drift with Evidently, and persists time-series metrics that Grafana surfaces.

## The MLOps pipeline

The next five subsections walk through each stage of the pipeline — training, registry, serving, monitoring, CI/CD — explaining *what* the project does, *why* it matters, and *where* to look in the codebase.

### 1. Training and experimentation

The training pipeline ([`project/train.py`](project/train.py)) is one Python entrypoint that owns the full path from raw CSV to a registry-resident model. Every run is recorded in MLflow as an experiment, so any model in the registry can be traced back to its hyperparameters, fold metrics, code version, and dataset hash.

```mermaid
flowchart LR
  CSV[train.csv / test.csv] --> FE[engineer_features]
  FE --> Opt[Optuna TPE study]
  Opt -->|best LGBM params| CV
  Opt -->|best XGB params| CV
  CV[5-fold stratified CV] --> LGBM[LGBMClassifier × 5]
  CV --> XGB[XGBClassifier × 5]
  LGBM --> Wrap
  XGB --> Wrap
  Wrap[ChurnEnsemble pyfunc] --> MLflow[(MLflow tracking)]
  MLflow --> Reg[MLflow registry: CustomerChurnEnsemble]
  Reg -->|set alias| Champ[@champion]
```

Key constants and choices, all visible in [`project/train.py`](project/train.py):

- `N_SPLITS = 5`, `RANDOM_STATE = 42` — deterministic fold assignment.
- Feature engineering derives `AvgMonthlyCharge` and `TotalServices`, one-hot encodes 10 categorical columns, binary-maps 4 yes/no columns, drops `gender`.
- Optuna TPE sampler with **SQLite-backed** storage (`optuna_studies.db`) so studies survive across runs and can be inspected post-hoc with `optuna-dashboard`.
- Search space: 9 LightGBM hyperparameters (`n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`), 8 XGBoost equivalents.
- Per-fold metrics (ROC-AUC, log-loss) are logged to MLflow alongside the best Optuna trial.
- The 10 trained boosters (5 LGBM + 5 XGB) are wrapped in a single `pyfunc` so serving doesn't need to know about the ensemble shape.

Training expects an MLflow server reachable at the URI hardcoded in `train.py` (currently `http://localhost:5001`). Start MLflow on that port, then:

```bash
uv run python project/train.py
```

Outputs: registered model `CustomerChurnEnsemble`, `project/submission.csv`, `project/optuna_studies.db`, and metrics/artifacts in the MLflow store.

See [`project/docs/TRAIN.md`](project/docs/TRAIN.md) for the full function-by-function walkthrough.

### 2. Model registry and promotion

The project uses **MLflow's post-2.x alias model**, not the deprecated `Staging` / `Production` stage names. Concretely:

- Every successful training run registers a new version of `CustomerChurnEnsemble`.
- A human operator (or a future promotion workflow) sets the `champion` alias on the version they want serving.
- The API loads `models:/CustomerChurnEnsemble@champion` at startup — no code change, no redeploy.

This decoupling is the central MLOps payoff of the registry: **promotion is metadata**. To roll back, point `@champion` at the previous version. To shadow-test, introduce a `@challenger` alias and route a percentage of traffic to it.

Promoting via the CLI:

```bash
mlflow models update --name CustomerChurnEnsemble --version <N> \
  --alias champion --tracking-uri http://localhost:5001
```

Or via the MLflow UI: Models → `CustomerChurnEnsemble` → version → "Set alias" → `champion`.

### 3. Serving

The FastAPI app ([`project/web_service.py`](project/web_service.py)) is a thin shell around the pyfunc model. The two responsibilities worth calling out are (a) **startup model load via the registry alias**, and (b) **preprocessing parity with training** — the live request path must apply the exact same feature engineering that `train.py` applied, or the model receives an out-of-distribution input and silently regresses.

```mermaid
flowchart LR
  Client[HTTP client] -->|JSON| Validate[Pydantic CustomerData]
  Validate -->|valid row| Preprocess[engineer_features]
  Validate -. 422 .-> Client
  Preprocess --> Predict[ChurnEnsemble pyfunc.predict]
  Predict --> Response[PredictionResponse JSON]
  Response --> Client
  Registry[(MLflow registry @champion)] -.->|lifespan startup| Predict
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/`              | Health and model metadata |
| `POST` | `/predict`       | Single customer prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET`  | `/model/info`    | Registered name, version, alias, URI |

Pydantic schemas ([`project/schemas/schemas.py`](project/schemas/schemas.py)) act as the **contract boundary**: enum-typed categorical fields reject typos with HTTP 422 before they ever reach the model. See [`project/docs/WEB_SERVICE.md`](project/docs/WEB_SERVICE.md) and [`project/docs/SCHEMAS.md`](project/docs/SCHEMAS.md).

Imports use `from schemas.schemas import …`, so the app must be run with **`project/`** on `PYTHONPATH`, or from inside **`project/`**.

```bash
# Option A — change into project/
cd project
uv run uvicorn web_service:app --reload --host 0.0.0.0 --port 8000

# Option B — from the repo root
PYTHONPATH=project uv run uvicorn web_service:app --reload --host 0.0.0.0 --port 8000
```

Set `MLFLOW_TRACKING_URI` if your MLflow server is not at the default for your environment. Interactive docs: `http://localhost:8000/docs` (Swagger), `http://localhost:8000/redoc`.

### 4. Monitoring and observability

Production ML systems fail quietly — accuracy degrades, but the API keeps returning 200s. The monitoring stack exists to make that failure mode visible. A replay worker ([`project/monitoring/monitor.py`](project/monitoring/monitor.py)) acts as a synthetic production-traffic generator: it samples batches from a held-out reference pool (80/20 stratified split of `train.csv`) and from `test.csv`, sends them through the live `/predict/batch` endpoint, computes Evidently drift + classification metrics, and writes one row per batch into a Postgres time-series table that Grafana visualizes.

```mermaid
flowchart LR
  Pool[train.csv 80/20 + test.csv] --> Sampler[batch sampler]
  Sampler --> Drift[optional drift injector]
  Drift -->|HTTP| API[/POST /predict/batch/]
  API --> Eval[Evidently report]
  API --> Metrics[sklearn metrics if labels]
  Eval --> PG[(postgres: monitoring_metrics)]
  Metrics --> PG
  PG --> Grafana[Grafana dashboards]
  PG --> Adminer[Adminer SQL UI]
```

What lands in Postgres ([`project/monitoring/init.sql`](project/monitoring/init.sql)): one row per batch with timestamps, `data_source`, `batch_id`, `batch_size`, dataset-level drift share, prediction drift, missing-value share, mean predicted churn probability, ground-truth churn rate, and (when labels are available) accuracy, ROC-AUC, and log-loss. Grafana dashboards are auto-provisioned from [`project/monitoring/grafana/`](project/monitoring/grafana/) — no manual import step.

**Synthetic drift injection** is configurable on the `monitor` service in [`project/docker-compose.yml`](project/docker-compose.yml) via `DRIFT_*` env vars:

- `DRIFT_MODE` ∈ `none` | `gradual` | `sustained` | `cycle`
- Tunable ramp/period, numeric shift magnitude, categorical swap probability, missing-value rate.

This exists so the Grafana dashboards have something interesting to show during a demo — it is explicitly **not** a production feature. Default is `none`.

See [`project/docs/MONITORING.md`](project/docs/MONITORING.md) for the dashboard layout and the full env-var reference.

### 5. CI/CD and release engineering

CI/CD is the part of MLOps that's least about ML and most about ops: tests, image builds, supply-chain attestations, secret scanning, and reproducible releases. Five workflows under [`.github/workflows/`](.github/workflows/):

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| [`ci.yml`](.github/workflows/ci.yml) | push (all branches), PR → main | Ruff lint + format check; pytest on Python 3.13; coverage gate `--cov-fail-under=60`; uploads coverage XML |
| [`docker.yml`](.github/workflows/docker.yml) | push main, push tag `v*.*.*`, PR → main (build-only), manual | Builds `churn-api`, `churn-mlflow`, `churn-monitor` images in parallel; pushes to GHCR on non-PR events; emits CycloneDX SBOM (`sbom: true`) and SLSA v1 provenance (`provenance: mode=max`) attestations alongside each manifest |
| [`release.yml`](.github/workflows/release.yml) | push tag `v*.*.*` | Generates GitHub Release with auto-notes since last tag; exports `uv.lock` → CycloneDX SBOM and attaches it as a release asset; uses `docker buildx imagetools create` to re-tag the existing `sha-<short>` image manifest under `:vX.Y.Z`, `:X.Y`, `:X`, `:latest` — **digest is preserved**, so the SBOM/provenance attestations remain valid under all aliases |
| [`secret-scan.yml`](.github/workflows/secret-scan.yml) | PR → main, push main, weekly cron | Gitleaks scan of full git history (`fetch-depth: 0`); SARIF report uploaded to GitHub Code Scanning; blocks PRs that introduce committed secrets |
| [`train.yml`](.github/workflows/train.yml) | manual (`workflow_dispatch`) | Provisions dataset (URL input or `TEST_DATASET_URL` secret), spawns local MLflow server, runs full `train.py`, uploads `submission.csv`, `optuna_studies.db`, and the entire `mlruns/` directory as artifacts |

```mermaid
flowchart LR
  Push[push / PR] --> CI[ci.yml: lint + tests + cov]
  Push --> Sec[secret-scan.yml: Gitleaks]
  Push --> Docker[docker.yml: build 3 images]
  Docker -->|PR| BuildOnly[no push]
  Docker -->|main| GHCR[(GHCR :main, :sha)]
  CI --> Merge{merge to main?}
  Sec --> Merge
  Merge -->|yes| Tag[git tag vX.Y.Z]
  Tag --> Docker2[docker.yml: rebuild + semver push]
  Tag --> Release[release.yml: GH Release + SBOM + tag promotion]
  Release --> GHCR2[(GHCR :vX.Y.Z :X.Y :X :latest)]
  Manual[manual dispatch] -.-> Train[train.yml: full training run]
  Train --> Artifacts[(mlruns/ + submission.csv)]
```

**Branch protection** (configured in repo settings, documented in [`docs/CI_CD.md`](docs/CI_CD.md)) requires `ci.yml` checks to pass before merging to `main`. The CI workflow is read-only-token; image push and release write happen only after merge or on a tag.

**Local reproduction of the CI gates** — if these pass locally, CI almost always passes on push:

```bash
# Lint
uv run ruff check .

# Tests + coverage gate (matches ci.yml exactly)
PYTHONPATH=project uv run pytest project/tests \
  --cov=project --cov-report=term --cov-fail-under=60

# pip-audit against the locked dependency graph
uv export --frozen --no-hashes --no-dev -o /tmp/req.txt
uvx pip-audit --requirement /tmp/req.txt --strict

# Image builds (matches docker.yml builds)
cd project
docker compose build api mlflow monitor
```

See [`docs/CI_CD.md`](docs/CI_CD.md) for the full workflow architecture, trigger reference, branching/release model, image references, secrets, caching strategy, gating logic, and troubleshooting recipes.

## Running the full stack locally

The canonical demo / development environment is the Compose stack from [`project/docker-compose.yml`](project/docker-compose.yml):

```bash
cd project && docker compose up --build
```

| Service | URL | Notes |
|---------|-----|-------|
| FastAPI | http://localhost:8000 | API container sets `MLFLOW_TRACKING_URI=http://mlflow:5001` |
| Swagger | http://localhost:8000/docs | Interactive endpoint explorer |
| MLflow  | http://localhost:5001 | Tracking server + registry UI |
| Grafana | http://localhost:3000 | Anonymous viewer; `admin` / `admin` to edit |
| Adminer | http://localhost:8080 | System `PostgreSQL`, server `postgres`, user/password/db = `monitor`/`monitor`/`monitoring` |
| Postgres | localhost:5431 → 5432 | `postgresql://monitor:monitor@localhost:5431/monitoring` |

The `monitor` service starts replaying as soon as the API and Postgres are healthy. Watch the Grafana dashboards populate over the next few minutes.

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | [`project/web_service.py`](project/web_service.py) | MLflow server for registry and artifacts when loading the model. Optional if the default file-based URI matches your setup |
| `API_URL` | [`project/monitoring/monitor.py`](project/monitoring/monitor.py) | Target API for batch replay (default in Compose: `http://api:8000`) |
| `BATCH_SIZE`, `INTERVAL_SECONDS`, `REFERENCE_FRAC`, `RUN_ONCE`, `LOOP_FOREVER` | monitor service | Replay loop pacing and dataset partitioning |
| `DRIFT_MODE`, `DRIFT_START_BATCH`, `DRIFT_RAMP_BATCHES`, `DRIFT_PERIOD_BATCHES`, `DRIFT_NUMERIC_SHIFT`, `DRIFT_CATEGORICAL_SWAP_PROB`, `DRIFT_MISSING_RATE` | monitor service | Synthetic drift injection knobs (see [MONITORING.md](project/docs/MONITORING.md)) |
| `TEST_DATASET_URL` | [`.github/workflows/train.yml`](.github/workflows/train.yml) | Optional repo secret for fetching the dataset on the CI training runner |

Training currently sets the tracking URI in code; align your MLflow deployment or edit [`project/train.py`](project/train.py) if you use a different host or port.

## Reproducing results end-to-end

```bash
# 1. Install
uv sync --group dev

# 2. Drop the dataset in
mkdir -p project/dataset
cp /path/to/train.csv project/dataset/
cp /path/to/test.csv  project/dataset/

# 3. Start MLflow (and the rest, if you want them up)
cd project
docker compose up -d mlflow

# 4. Train — produces a new version of CustomerChurnEnsemble
cd ..
uv run python project/train.py

# 5. Promote the new version
mlflow models update --name CustomerChurnEnsemble --version 1 \
  --alias champion --tracking-uri http://localhost:5001

# 6. Bring up the full stack (API loads @champion at startup)
cd project && docker compose up --build

# 7. Smoke-test the API
curl -s http://localhost:8000/ | jq
curl -s http://localhost:8000/model/info | jq

# 8. Watch Grafana fill in
open http://localhost:3000
```

## Development

```bash
uv run ruff check .          # lint (matches CI)
uv run ruff format .         # auto-format
uv run ruff format --check . # CI-style check without modifying files
```

Run the tests the way CI does (the `PYTHONPATH=project` is required because of how `schemas/` is imported):

```bash
PYTHONPATH=project uv run pytest project/tests \
  --cov=project --cov-report=term --cov-fail-under=60
```

## Further reading

- [Training pipeline](project/docs/TRAIN.md) — functions, constants, HPO spaces, CV, registration
- [Web service](project/docs/WEB_SERVICE.md) — endpoints, preprocessing parity with training, Docker path handling
- [Pydantic schemas](project/docs/SCHEMAS.md) — enums and request/response models
- [Monitoring pipeline](project/docs/MONITORING.md) — Evidently + Postgres + Adminer + Grafana
- [Test suite](project/docs/TESTS.md) — fixtures, mocks, what each module covers
- [CI/CD architecture](docs/CI_CD.md) — workflow matrix, triggers, branch protection, troubleshooting

## Use of AI assistance

AI tools were used during the development of this project as an **assistive and advisory aid** — to draft and refine documentation, scaffold and review code, surface alternative approaches, and help with debugging. The primary assistant was Anthropic's Claude, used via the Claude Code CLI.

All substantive decisions were made by the author. Architecture, model and library selection, the cross-validation and hyperparameter-optimization strategy, evaluation methodology, the monitoring and CI/CD design, and the final implementation reflect the author's own judgement. Every AI-generated suggestion was reviewed, tested, and then accepted, modified, or rejected at the author's discretion — AI guided the work but did not direct it. The author takes full responsibility for the content, correctness, and academic integrity of this repository.
