# Artificial Intelligence (HSLU)

Coursework and project work for **HSLU Artificial Intelligence**. The main deliverable is a **Customer Churn Prediction** stack under [`project/`](project/): a **LightGBM + XGBoost** ensemble (5-fold stratified cross-validation), **Optuna** hyperparameter optimization, **MLflow** experiment tracking and Model Registry, and a **FastAPI** service for real-time inference on telecom-style tabular data.

The repository root [`main.py`](main.py) is a minimal placeholder. Smaller exercises live under [`lessons/`](lessons/) (for example [`lessons/sw_03/web_service.py`](lessons/sw_03/web_service.py)). [`dashboard.html`](dashboard.html) is a standalone static “MLOps blueprint” page (Tailwind CSS, Chart.js); it is not part of the Python runtime.

## Tech stack

| Area | Notes |
|------|--------|
| Runtime | Python **3.13+** (see [`.python-version`](.python-version)) |
| Packaging | [`uv`](https://docs.astral.sh/uv/) with [`pyproject.toml`](pyproject.toml) and [`uv.lock`](uv.lock) |
| ML | LightGBM, XGBoost, scikit-learn, pandas, NumPy, seaborn, joblib |
| Training | Optuna (HPO), MLflow (tracking + registry) |
| Serving | FastAPI, Pydantic ([`project/schemas/`](project/schemas/schemas.py)) |
| Orchestration | [Prefect](https://www.prefect.io/) is listed as a dependency; the main training path is [`project/train.py`](project/train.py) |

Install dependencies from the repository root:

```bash
uv sync
```

## Repository layout

| Path | Role |
|------|------|
| [`project/train.py`](project/train.py) | End-to-end training: load data → feature engineering → Optuna → CV ensemble → register `CustomerChurnEnsemble` → `submission.csv` |
| [`project/web_service.py`](project/web_service.py) | FastAPI app; loads `models:/CustomerChurnEnsemble@champion` from MLflow |
| [`project/schemas/`](project/schemas/) | Pydantic request and response models |
| [`project/docker-compose.yml`](project/docker-compose.yml) | API on **8000**, MLflow on **5001**, Postgres on **5432**, Adminer on **8080**, Grafana on **3000** |
| [`project/monitoring/`](project/monitoring/) | Evidently-based drift + performance monitoring (Dockerfile, Grafana provisioning, Postgres schema) |
| [`project/docs/`](project/docs/) | Deep dives: [TRAIN.md](project/docs/TRAIN.md), [WEB_SERVICE.md](project/docs/WEB_SERVICE.md), [SCHEMAS.md](project/docs/SCHEMAS.md), [MONITORING.md](project/docs/MONITORING.md) |
| [`project/customer-churn-eda-multimodelensemble.ipynb`](project/customer-churn-eda-multimodelensemble.ipynb) | EDA and exploratory modeling |

## Data

Place your course or competition CSVs under **`project/dataset/`**:

- `project/dataset/train.csv`
- `project/dataset/test.csv`

The `dataset/` directory is gitignored ([`.gitignore`](.gitignore)); do not commit raw data.

## Training (MLflow + `train.py`)

Training expects an MLflow server compatible with the URI configured in [`project/train.py`](project/train.py) (currently **`http://localhost:5001`** for both tracking and registry). Start MLflow on that port, then run:

```bash
uv run python project/train.py
```

Typical outputs and artifacts (details in [TRAIN.md](project/docs/TRAIN.md)):

- Registered model name **`CustomerChurnEnsemble`**, with alias **`champion`** when promoted
- **`project/submission.csv`** (test predictions)
- **`project/optuna_studies.db`** (Optuna studies)
- Metrics and artifacts in MLflow (artifact root as configured on the server)

### Docker Compose (MLflow + API + Monitoring)

From `project/`:

```bash
cd project && docker compose up --build
```

- MLflow UI / API: **http://localhost:5001**
- FastAPI: **http://localhost:8000** (sets `MLFLOW_TRACKING_URI=http://mlflow:5001` for the API container)
- Grafana: **http://localhost:3000** (anonymous viewer; admin / admin to edit)
- Adminer: **http://localhost:8080** (system `PostgreSQL`, server `postgres`, user/password/db = `monitor`/`monitor`/`monitoring`)
- Postgres: **localhost:5432** (`postgresql://monitor:monitor@localhost:5432/monitoring`)

The `monitor` service replays `dataset/train.csv` (stratified 80/20 reference split) and `dataset/test.csv` through `/predict/batch` and writes Evidently drift + classification metrics to the `monitoring_metrics` Postgres table. See [project/docs/MONITORING.md](project/docs/MONITORING.md) for the full pipeline, dashboard layout, and tunable env vars.

## Running the API locally

Imports use `from schemas.schemas import …`, so the app must be run with **`project/`** on `PYTHONPATH`, or from inside **`project/`**.

**Option A — change into `project/` (recommended):**

```bash
cd project
uv run uvicorn web_service:app --reload --host 0.0.0.0 --port 8000
```

**Option B — from the repository root:**

```bash
PYTHONPATH=project uv run uvicorn web_service:app --reload --host 0.0.0.0 --port 8000
```

Set **`MLFLOW_TRACKING_URI`** if your MLflow server is not the default for your environment (for example `http://localhost:5001` when the API runs on the host and MLflow is exposed on that port). The service loads **`CustomerChurnEnsemble@champion`** at startup; run `train.py` against the same registry first.

Interactive API docs: **http://localhost:8000/docs** (Swagger) and **http://localhost:8000/redoc**.

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Health / info |
| `POST` | `/predict` | Single customer prediction |
| `POST` | `/predict/batch` | Batch predictions |
| `GET` | `/model/info` | Model metadata |

Request and response fields are documented in [SCHEMAS.md](project/docs/SCHEMAS.md).

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | [`project/web_service.py`](project/web_service.py) | MLflow server for registry and artifacts when loading the model (optional if the default file-based URI matches your setup) |

Training currently sets the tracking URI in code; align your MLflow deployment or edit [`train.py`](project/train.py) if you use a different host or port.

## Architecture (high level)

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
  Registry -.->|champion model| Model
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

## Development

Formatting and linting: [Ruff](https://docs.astral.sh/ruff/) is available via the project environment, for example:

```bash
uv run ruff check .
uv run ruff format .
```

## Further reading

- [Training pipeline](project/docs/TRAIN.md) — functions, constants, HPO spaces, CV, registration
- [Web service](project/docs/WEB_SERVICE.md) — endpoints, preprocessing parity with training, Docker path handling
- [Pydantic schemas](project/docs/SCHEMAS.md) — enums and request/response models
- [Monitoring pipeline](project/docs/MONITORING.md) — Evidently + Postgres + Adminer + Grafana





Practical CI/CD Walkthrough

  0. One-time setup (do this once per repo / fork)

  1. Push the new workflows to GitHub:
  git add .github/workflows/ docs/CI_CD.md CLAUDE.md
  git commit -m "ci: comprehensive CI/CD pipeline + docs"
  git push origin ft-ci-cd
  2. Enable workflow write permissions (required by docker.yml, release.yml):
    - GitHub → repo → Settings → Actions → General → Workflow permissions
    - Select Read and write permissions → Save.
  3. Enable GitHub Code Scanning (so SARIF uploads from Trivy/CodeQL/Gitleaks appear):
    - Settings → Code security and analysis → enable Code scanning (Default setup is fine; advanced setup is unnecessary because workflows upload SARIF themselves).
  4. (Optional) Add the training secret:
    - Settings → Secrets and variables → Actions → New repository secret
    - Name: TEST_DATASET_URL, Value: a URL to your test.csv.
  5. Configure branch protection on main (this is what actually makes CI a gate):
    - Settings → Branches → Branch protection rules → Add rule
    - Branch name pattern: main
    - Tick Require status checks to pass before merging and add:
        - CI / Lint (ruff)
      - CI / Tests (pytest) (3.13)
      - Smoke Test / Compose stack smoke
      - Build and Push Images / Build churn-api
      - Build and Push Images / Build churn-mlflow
      - Build and Push Images / Build churn-monitor
      - Security Scans / pip-audit (uv lockfile)
      - Secret Scan / Gitleaks
    - (These names appear after the first run of each workflow — open a throwaway PR first if the dropdown is empty.)

  ---
  1. Day-to-day: a feature change

  # 1. Branch off main
  git checkout main && git pull
  git checkout -b ft-my-change

  # 2. Make changes, then validate locally before pushing
  uv run ruff check .
  PYTHONPATH=project uv run pytest project/tests --cov=project --cov-fail-under=60

  # 3. Commit and push — this triggers ci.yml on the feature branch
  git add -A
  git commit -m "feat: my change"
  git push -u origin ft-my-change

  Open the PR (gh pr create or in the UI). On PR open, these run automatically:

  ┌────────────────┬──────────────────────────────┐
  │    Workflow    │ What you'll see in PR checks │
  ├────────────────┼──────────────────────────────┤
  │ ci.yml         │ Lint + tests                 │
  ├─────────────────┼──────────────────────────────┤
  │ smoke-test.yml  │ Compose boot                 │
  ├─────────────────┼──────────────────────────────┤
  │ docker.yml      │ 3 image builds (no push)     │
  ├─────────────────┼──────────────────────────────┤
  │ security.yml    │ pip-audit + 3 Trivy scans    │
  ├─────────────────┼──────────────────────────────┤
  │ codeql.yml      │ Python SAST                  │
  ├─────────────────┼──────────────────────────────┤
  │ secret-scan.yml │ Gitleaks                     │
  └─────────────────┴──────────────────────────────┘

  Wait for green. If anything red:

  # Watch the latest run from the CLI
  gh run watch

  # Tail the failed job's logs
  gh run view --log-failed

  After merge, on main the same checks rerun and docker.yml now actually pushes images to GHCR.

  ---
  2. Cutting a release

  # 1. Make sure main is green
  gh run list --branch main --limit 5

  # 2. Tag and push
  git checkout main && git pull
  git tag -a v1.0.0 -m "1.0.0"
  git push origin v1.0.0

  Two workflows fire automatically off the tag push:

  - docker.yml builds + pushes images tagged v1.0.0, 1.0, 1, latest (with SBOM + provenance).
  - release.yml then:
    a. Generates a GitHub Release with auto-generated notes since the previous tag.
    b. Exports uv.lock → CycloneDX SBOM → attaches sbom-python.cdx.json to the release.
    c. Re-tags the sha-<short> image manifest under the semver aliases (digest-preserving).

  Verify:
  gh release view v1.0.0
  gh release download v1.0.0 -p "sbom-python.cdx.json"

  # Pull a released image
  docker pull ghcr.io/<your-org-lowercase>/churn-api:1.0.0

  ---
  3. Running training manually

  UI path:
  - Actions → Train Model → Run workflow dropdown
  - Set experiment_name (default ci-training)
  - Optionally set dataset_url (or rely on the TEST_DATASET_URL secret)
  - Click Run workflow

  CLI path:
  gh workflow run train.yml \
    -f experiment_name=ci-training \
    -f dataset_url="https://example.com/train.csv"
   
  gh run watch              # follow the run

  After it completes, download artifacts:
  RUN=$(gh run list --workflow train.yml --limit 1 --json databaseId -q '.[0].databaseId')
  gh run download $RUN -n training-artifacts
  gh run download $RUN -n mlruns -D ./mlruns

  ---
  4. Triggering scans on demand

  # Run security scans against current main without waiting for the cron
  gh workflow run security.yml

  # Same for CodeQL and secrets
  gh workflow run codeql.yml
  gh workflow run secret-scan.yml

  Read the results:
  - Security tab → Code scanning — Trivy, CodeQL, Gitleaks findings (grouped by category set in each workflow).
  - Actions tab → run → Job summary — Gitleaks markdown summary; pip-audit findings inline in logs.

  ---
  5. Reproducing CI locally before pushing
  
  A copy-pasteable equivalent of every gate (also in docs/CI_CD.md §11):

  # Lint
  uv run ruff check .

  # Tests + coverage gate (matches ci.yml exactly)
  PYTHONPATH=project uv run pytest project/tests \
    --cov=project --cov-report=term --cov-fail-under=60

  # pip-audit (matches security.yml)
  uv export --frozen --no-hashes --no-dev -o /tmp/req.txt
  uvx pip-audit --requirement /tmp/req.txt --strict

  # Image builds + smoke (matches smoke-test.yml)
  cd project
  docker compose build api mlflow monitor
  docker compose up -d postgres mlflow
  docker compose exec -T postgres psql -U monitor -d monitoring -c '\dt monitoring_metrics'
  curl -fsS http://localhost:5001/health
  docker compose run --rm --no-deps --entrypoint python api -c "import web_service; print('OK')"
  docker compose run --rm --no-deps --entrypoint python monitor -c "import monitor; print('OK')"
  docker compose down -v

  If those all pass locally, CI almost always passes on push.

  ---
  6. When something breaks — diagnostic recipes

  # What ran on the latest commit?
  gh run list --branch $(git branch --show-current) --limit 10

  # Failed job logs
  gh run view <run-id> --log-failed

  # Re-run only the failed jobs
  gh run rerun <run-id> --failed

  # Cancel a stuck run (e.g. training timeout)
  gh run cancel <run-id>

  Specific failure modes are catalogued in docs/CI_CD.md §12 (postgres init-volume issues, GHCR 403s, release-promotion missing source tag, etc.).

  ---
  TL;DR cheat sheet

  ┌────────────────────────────┬────────────────────────────────────────────────────────┐
  │        You want to…        │                         You do                         │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ See what CI runs on a push │ gh run list                                            │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Run training               │ gh workflow run train.yml                              │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Cut a release              │ git tag -a vX.Y.Z -m "X.Y.Z" && git push origin vX.Y.Z │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Force a security scan now  │ gh workflow run security.yml                           │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Reproduce CI locally       │ the block in §5                                        │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Find a CVE finding         │ repo Security tab → Code scanning                      │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Download a trained model   │ gh run download <id> -n mlruns                         │
  └────────────────────────────┴────────────────────────────────────────────────────────┘

✻ Churned for 42s

⏺ The Trivy action diagnostic is a linter false positive — the action and
  version both exist; this is the version that was already in the working
  file. Continuing with the secret scan workflow.

⏺ Write(.github/workflows/secret-scan.yml)
  ⎿  Wrote 66 lines to .github/workflows/secret-scan.yml
       1 # ──────────────────────────────────────────────────────────────
       3 # ──────────────────────────────────────────────────────────────
     ───────────────
       5 # Detects committed secrets (AWS keys, GitHub tokens, generic
     high-entropy
       6 # strings matching curated patterns) in the diff. Complements
       8 # secret scanning by:
       9 #   - running on every PR before merge (built-in secret scanning
      10 #     catches secrets matching partner providers, and only
     post-push)
     … +78 lines (ctrl+o to expand)

  Searched for 2 patterns (ctrl+o to expand)

⏺ Write(.github/workflows/smoke-test.yml)
  ⎿  Wrote 176 lines to .github/workflows/smoke-test.yml
       1 # ──────────────────────────────────────────────────────────────
       2 # Smoke test — boot a partial stack to prove the images
       3 # ──────────────────────────────────────────────────────────────
       4 #──────────
       5 # Unit tests prove individual modules work. Security scans prove
       6 # CVEs. Neither catches "this image was built but immediately
       7 # startup", or "the postgres init.sql doesn't create the schema
       8 # expects", or "the API image is missing an OS package that one
       9 # Python wheels dlopen's at import time."
      10 #
     … +238 lines (ctrl+o to expand)

⏺ Now the comprehensive docs file.

⏺ Write(docs/CI_CD.md)
  ⎿  Wrote 412 lines to docs/CI_CD.md
       2 # CI/CD Architecture
       3 This document describes the continuous-integration and
       4 setup for the Customer Churn Prediction stack. It is the source
     of truth
       5 for **what runs when, why each job exists, what it gates, and
       6 extend or debug it**. The workflow YAML files in
       7 heavily commented and are the second source — this document is
     the
       9 big-picture view that those comments can't easily express.
      10 ---
     … +516 lines (ctrl+o to expand)

⏺ Quick verification — list final workflow state.

  Listed 1 directory (ctrl+o to expand)

⏺ Done. Here's what landed:

  Rewritten (with heavy inline documentation):
  - .github/workflows/ci.yml — lint + tests + coverage; comments explain the
  uv caching strategy, why lint gates tests, and the PYTHONPATH=project
  belt-and-braces.
  - .github/workflows/docker.yml — adds sbom: true and provenance: mode=max to
   every image build; comments cover GHCR auth, the tag rule precedence, and
  why PRs build-but-don't-push.
  - .github/workflows/release.yml — adds CycloneDX SBOM generation from
  uv.lock and attaches it to the GitHub Release; comments explain why we
  re-tag via imagetools rather than rebuilding (digest stability for
  SBOM/provenance attestations).
  - .github/workflows/security.yml — same pip-audit + Trivy, but documents the
   soft/hard gate split and SARIF category usage.
  - .github/workflows/train.yml — manual training, now documented end-to-end
  including the MLflow-as-background-process pattern and the TEST_DATASET_URL
  secret.

  New workflows:
  - .github/workflows/codeql.yml — Python SAST with the security-extended
  query suite, weekly cron + per-PR.
  - .github/workflows/secret-scan.yml — Gitleaks on full history (fetch-depth:
   0) so historical secrets surface even if removed from HEAD.
  - .github/workflows/smoke-test.yml — docker compose build, boot
  postgres+mlflow, verify monitoring_metrics exists and MLflow /health
  responds, then import-only smoke for the api/monitor images (the api can't
  fully boot in CI without a champion model — explained in the file header).

  Documentation:
  - docs/CI_CD.md — 13-section architecture doc: goals, pipeline diagram,
  workflow matrix, trigger reference, branching/release model, image
  references, secrets, caching, gating logic, training workflow, local
  reproduction, troubleshooting, extension recipes.

  Two notes:
  - The aquasecurity/trivy-action@0.28.0 diagnostic the IDE shows is a linter
  false positive — that exact version was already in the working file and
  exists on the marketplace.
  - train.yml still surfaces a warning about secrets.TEST_DATASET_URL being
  potentially undefined; that's intentional (the script handles its absence).

✻ Brewed for 9m 3s

❯ give me a step by step guide on how to run the CI/CD practically            

⏺ Practical CI/CD Walkthrough

  0. One-time setup (do this once per repo / fork)

  1. Push the new workflows to GitHub:
  git add .github/workflows/ docs/CI_CD.md CLAUDE.md
  git commit -m "ci: comprehensive CI/CD pipeline + docs"
  git push origin ft-ci-cd
  2. Enable workflow write permissions (required by docker.yml, release.yml):
    - GitHub → repo → Settings → Actions → General → Workflow permissions
    - Select Read and write permissions → Save.
  3. Enable GitHub Code Scanning (so SARIF uploads from Trivy/CodeQL/Gitleaks
  appear):
    - Settings → Code security and analysis → enable Code scanning (Default
  setup is fine; advanced setup is unnecessary because workflows upload SARIF
  themselves).
  4. (Optional) Add the training secret:
    - Settings → Secrets and variables → Actions → New repository secret
    - Name: TEST_DATASET_URL, Value: a URL to your test.csv.
  5. Configure branch protection on main (this is what actually makes CI a
  gate):
    - Settings → Branches → Branch protection rules → Add rule
    - Branch name pattern: main
    - Tick Require status checks to pass before merging and add:
        - CI / Lint (ruff)
      - CI / Tests (pytest) (3.13)
      - Smoke Test / Compose stack smoke
      - Build and Push Images / Build churn-api
      - Build and Push Images / Build churn-mlflow
      - Build and Push Images / Build churn-monitor
      - Security Scans / pip-audit (uv lockfile)
      - Secret Scan / Gitleaks
    - (These names appear after the first run of each workflow — open a
  throwaway PR first if the dropdown is empty.)

  ---
  1. Day-to-day: a feature change

  # 1. Branch off main
  git checkout main && git pull
  git checkout -b ft-my-change

  # 2. Make changes, then validate locally before pushing
  uv run ruff check .
  PYTHONPATH=project uv run pytest project/tests --cov=project
  --cov-fail-under=60

  # 3. Commit and push — this triggers ci.yml on the feature branch
  git add -A
  git commit -m "feat: my change"
  git push -u origin ft-my-change

  Open the PR (gh pr create or in the UI). On PR open, these run
  automatically:

  ┌─────────────────┬──────────────────────────────┐
  │    Workflow     │ What you'll see in PR checks │
  ├─────────────────┼──────────────────────────────┤
  │ ci.yml          │ Lint + tests                 │
  ├─────────────────┼──────────────────────────────┤
  │ smoke-test.yml  │ Compose boot                 │
  ├─────────────────┼──────────────────────────────┤
  │ docker.yml      │ 3 image builds (no push)     │
  ├─────────────────┼──────────────────────────────┤
  │ security.yml    │ pip-audit + 3 Trivy scans    │
  ├─────────────────┼──────────────────────────────┤
  │ codeql.yml      │ Python SAST                  │
  ├─────────────────┼──────────────────────────────┤
  │ secret-scan.yml │ Gitleaks                     │
  └─────────────────┴──────────────────────────────┘

  Wait for green. If anything red:

  # Watch the latest run from the CLI
  gh run watch

  # Tail the failed job's logs
  gh run view --log-failed

  After merge, on main the same checks rerun and docker.yml now actually
  pushes images to GHCR.

  ---
  2. Cutting a release

  # 1. Make sure main is green
  gh run list --branch main --limit 5

  # 2. Tag and push
  git checkout main && git pull
  git tag -a v1.0.0 -m "1.0.0"
  git push origin v1.0.0

  Two workflows fire automatically off the tag push:

  - docker.yml builds + pushes images tagged v1.0.0, 1.0, 1, latest (with SBOM
   + provenance).
  - release.yml then:
    a. Generates a GitHub Release with auto-generated notes since the previous
   tag.
    b. Exports uv.lock → CycloneDX SBOM → attaches sbom-python.cdx.json to the
   release.
    c. Re-tags the sha-<short> image manifest under the semver aliases
  (digest-preserving).

  Verify:
  gh release view v1.0.0
  gh release download v1.0.0 -p "sbom-python.cdx.json"

  # Pull a released image
  docker pull ghcr.io/<your-org-lowercase>/churn-api:1.0.0

  ---
  3. Running training manually

  UI path:
  - Actions → Train Model → Run workflow dropdown
  - Set experiment_name (default ci-training)
  - Optionally set dataset_url (or rely on the TEST_DATASET_URL secret)
  - Click Run workflow

  CLI path:
  gh workflow run train.yml \
    -f experiment_name=ci-training \
    -f dataset_url="https://example.com/train.csv"

  gh run watch              # follow the run

  After it completes, download artifacts:
  RUN=$(gh run list --workflow train.yml --limit 1 --json databaseId -q
  '.[0].databaseId')
  gh run download $RUN -n training-artifacts
  gh run download $RUN -n mlruns -D ./mlruns

  ---
  4. Triggering scans on demand

  # Run security scans against current main without waiting for the cron
  gh workflow run security.yml

  # Same for CodeQL and secrets
  gh workflow run codeql.yml
  gh workflow run secret-scan.yml

  Read the results:
  - Security tab → Code scanning — Trivy, CodeQL, Gitleaks findings (grouped
  by category set in each workflow).
  - Actions tab → run → Job summary — Gitleaks markdown summary; pip-audit
  findings inline in logs.

  ---
  5. Reproducing CI locally before pushing

  A copy-pasteable equivalent of every gate (also in docs/CI_CD.md §11):

  # Lint
  uv run ruff check .

  # Tests + coverage gate (matches ci.yml exactly)
  PYTHONPATH=project uv run pytest project/tests \
    --cov=project --cov-report=term --cov-fail-under=60

  # pip-audit (matches security.yml)
  uv export --frozen --no-hashes --no-dev -o /tmp/req.txt
  uvx pip-audit --requirement /tmp/req.txt --strict

  # Image builds + smoke (matches smoke-test.yml)
  cd project
  docker compose build api mlflow monitor
  docker compose up -d postgres mlflow
  docker compose exec -T postgres psql -U monitor -d monitoring -c '\dt
  monitoring_metrics'
  curl -fsS http://localhost:5001/health
  docker compose run --rm --no-deps --entrypoint python api -c "import
  web_service; print('OK')"
  docker compose run --rm --no-deps --entrypoint python monitor -c "import
  monitor; print('OK')"
  docker compose down -v

  If those all pass locally, CI almost always passes on push.

  ---
  6. When something breaks — diagnostic recipes

  # What ran on the latest commit?
  gh run list --branch $(git branch --show-current) --limit 10

  # Failed job logs
  gh run view <run-id> --log-failed

  # Re-run only the failed jobs
  gh run rerun <run-id> --failed

  # Cancel a stuck run (e.g. training timeout)
  gh run cancel <run-id>

  Specific failure modes are catalogued in docs/CI_CD.md §12 (postgres
  init-volume issues, GHCR 403s, release-promotion missing source tag, etc.).

  ---
  TL;DR cheat sheet

  ┌────────────────────────┬──────────────────────────────────────────────┐
  │      You want to…      │                    You do                    │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ See what CI runs on a  │ gh run list                                  │
  │ push                   │                                              │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ Run training           │ gh workflow run train.yml                    │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ Cut a release          │ git tag -a vX.Y.Z -m "X.Y.Z" && git push     │
  │                        │ origin vX.Y.Z                                │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ Force a security scan  │ gh workflow run security.yml                 │
  │ now                    │                                              │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ Reproduce CI locally   │ the block in §5                              │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ Find a CVE finding     │ repo Security tab → Code scanning            │
  ├────────────────────────┼──────────────────────────────────────────────┤
  │ Download a trained     │ gh run download <id> -n mlruns               │
  │ Find a CVE finding         │ repo Security tab → Code scanning                      │
  ├────────────────────────────┼────────────────────────────────────────────────────────┤
  │ Download a trained model   │ gh run download <id> -n mlruns                         │
  └────────────────────────────┴────────────────────────────────────────────────────────┘