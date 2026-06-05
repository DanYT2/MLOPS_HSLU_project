# Test suite

This document describes the automated test suite for the Customer Churn Prediction project — what is tested, what is intentionally not tested, how the fixtures are wired, and how to run everything locally the way CI does.

## Overview

| Metric | Value |
|--------|-------|
| Test files | 3 (`test_schemas.py`, `test_web_service.py`, `test_monitor.py`) |
| Shared infrastructure | 1 (`conftest.py`) |
| Test classes | 7 |
| Test cases | 20 |
| Python version | 3.13 (matrix in [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml)) |
| Test runner | `pytest >= 8.0` |
| Coverage tool | `pytest-cov` |
| CI coverage gate | `--cov-fail-under=50` (see [`ci.yml`](../../.github/workflows/ci.yml)) |
| Coverage scope | Everything under `project/`, except `project/tests/*`, `project/train.py`, `project/monitoring/grafana/*` |
| Speed | Full suite runs in well under a minute on a laptop — no Postgres, no MLflow, no network |

The suite is deliberately tight on **boundary code** (Pydantic schemas, HTTP handlers, drift-injection math, CSV-to-API coercion) and silent on **orchestration code** (`train.py`, the long-running `monitor.run()` replay loop). The latter requires a live MLflow server, Postgres, and the dataset — that's integration territory, not unit testing.

## Running the tests

### The CI invocation (preferred)

```bash
PYTHONPATH=project uv run pytest project/tests \
  --cov=project \
  --cov-report=term \
  --cov-fail-under=50
```

This is the exact command [`ci.yml`](../../.github/workflows/ci.yml) runs. The `PYTHONPATH=project` is the belt-and-braces version of the path injection that [`pyproject.toml`](../../pyproject.toml) (`pythonpath = ["project"]`) and [`conftest.py`](../tests/conftest.py) also perform.

### Bare invocation (works because of the safety net)

```bash
uv run pytest                 # picks up testpaths from pyproject.toml
uv run pytest -k schemas      # only schema tests
uv run pytest --lf            # rerun last failures
uv run pytest -x -vv          # stop at first failure, verbose
```

### Single test

```bash
uv run pytest project/tests/test_web_service.py::TestPredict::test_single_predict_returns_stub_probability
```

### Coverage HTML report (local exploration)

```bash
PYTHONPATH=project uv run pytest project/tests --cov=project --cov-report=html
open htmlcov/index.html
```

## Why `PYTHONPATH=project` matters

Production code in `web_service.py` and `monitoring/monitor.py` imports as `from schemas.schemas import ...`. That import only resolves when `project/` is on `sys.path`. There are **three places** that guarantee this — together they cover every invocation path:

1. **[`pyproject.toml`](../../pyproject.toml)** — `[tool.pytest.ini_options] pythonpath = ["project"]`. Covers `pytest` invoked from the repo root.
2. **[`conftest.py`](../tests/conftest.py)** — programmatic `sys.path.insert(0, …)`. Covers `pytest` invoked from inside `project/`, or via tooling integrations that bypass `pyproject.toml`.
3. **`.github/workflows/ci.yml`** — sets `PYTHONPATH=project` env var explicitly. Covers CI even if (1) or (2) regress.

If you restructure the project tree, keep all three in lockstep.

## Shared fixtures — [`conftest.py`](../tests/conftest.py)

Three responsibilities, none of which are actual tests.

### 1. `sys.path` safety net

The first import-time block resolves `project/` (via `Path(__file__).resolve().parents[1]`) and prepends it to `sys.path` if not already present. Index 0 ensures any system-installed package can't shadow the local `schemas` module.

### 2. `stub_model` fixture (`_StubChurnModel`)

The production model is `models:/CustomerChurnEnsemble@champion` — ten serialized fold models (5 LightGBM + 5 XGBoost) loaded from the MLflow Model Registry on FastAPI startup. CI runners have no registry; loading the real model would mean ~30 s of deserialization and a network round trip to MLflow on every test run.

`_StubChurnModel` exposes only the surface area the FastAPI handlers touch:

| Attribute | Real type | Stub | Used by |
|-----------|-----------|------|---------|
| `.predict(df)` | mlflow pyfunc method returning `np.ndarray[float64]` | `np.full(len(df), 0.42, dtype=float)` | `/predict`, `/predict/batch` |
| `.metadata.flavors` | `dict` | `{"python_function": {}}` | `/model/info` |
| `.metadata.run_id` | `str` | `"test-run-id"` | `/model/info` |
| `.metadata.artifact_path` | `str` | `"model"` | `/model/info` |

The fixed probability `0.42` is deliberate: it sits below the 0.5 decision threshold, so every test can assert `churn_probability ≈ 0.42` **and** `churn is False` with no flakiness.

### 3. `valid_customer_payload` fixture

A schema-valid customer dict lifted from the API's OpenAPI example. It is the single source of truth for sample data across the suite:

- Schema tests mutate **one field at a time** to trigger `ValidationError`.
- Endpoint tests `POST` it unchanged to `/predict`.
- The preprocessing test feeds it through `engineer_features` to assert engineered columns.

Centralizing this means a schema change breaks one place, not five. The values are chosen so that exactly **4 of the 8 service columns** are `"Yes"` — the preprocessing test asserts `TotalServices == 4` against this.

## File 1 — [`test_schemas.py`](../tests/test_schemas.py)

Gates the Pydantic boundary. Every incoming HTTP request passes through `CustomerData`; every response goes out through `PredictionResponse` / `BatchPredictionResponse`. If these break, the API is wrong before the model is ever called.

### `TestCustomerData` (6 tests)

Pattern: baseline + one mutation per test.

| Test | Mutation | Asserts | Why |
|------|----------|---------|-----|
| `test_accepts_valid_payload` | none | Baseline parses; enum-backed fields land on enum **members** (e.g. `Contract.one_year`), not raw strings | If this fails, every other test in the class is suspect. The enum-member check also guards against an accidental `use_enum_values=True` config that would silently change the downstream type |
| `test_rejects_invalid_enum_value` | `Contract = "lifetime"` | `ValidationError` | Without strict enum validation, the one-hot encoder downstream would emit all-zeros for unknown categories — silent garbage prediction |
| `test_rejects_negative_tenure` | `tenure = -1` | `ValidationError` | Logically impossible customers must be caught at the boundary, not skew monitoring |
| `test_rejects_negative_charges` | `MonthlyCharges = -10.0` | `ValidationError` | Same reasoning, currency edition |
| `test_rejects_invalid_senior_citizen` | `SeniorCitizen = 2` | `ValidationError` | Telco dataset encodes this as binary integer (0/1); anything else would be treated as a magnitude |
| `test_rejects_missing_required_field` | drop `MonthlyCharges` | `ValidationError` | Catches the absence at parse time (HTTP 422) rather than as a downstream `KeyError` (HTTP 500) |

### `TestResponseSchemas` (2 tests)

Looser than `TestCustomerData` because the API constructs these — there is no untrusted input.

| Test | Asserts |
|------|---------|
| `test_prediction_response` | `PredictionResponse(churn_probability, churn)` round-trips both fields with correct types |
| `test_batch_prediction_wraps_list` | `BatchPredictionResponse.predictions` is an **ordered list** of 2 items, where index 1 keeps its `churn=True`. Guards against a future refactor that accidentally returns a dict-by-index |

## File 2 — [`test_web_service.py`](../tests/test_web_service.py)

End-to-end tests of the FastAPI app via `fastapi.testclient.TestClient`. The registry loader is monkey-patched so the lifespan handler receives the stub instead of phoning MLflow.

### The `client` fixture — the key plumbing

Three non-obvious choices:

1. **`import web_service` lives inside the fixture body**, not at module scope. This defers the import until after `conftest.py`'s `sys.path` patch has taken effect. Module-level imports would race the path setup.
2. **`monkeypatch.setattr(web_service, "_load_registry_model", lambda *a, **kw: stub_model)`** replaces the registry loader for the test's duration. The `*a, **kw` signature absorbs the real call's `alias="champion"` kwarg without binding to it (insulates the test from signature drift).
3. **`with TestClient(app)`** is used as a context manager so FastAPI's lifespan fires on enter (loads the stub) and on exit (shuts the app down cleanly). Plain `TestClient(app)` would skip lifespan entirely and the model would never be loaded.

### `TestHealthAndInfo` (2 tests)

| Test | Asserts |
|------|---------|
| `test_health_check` | `GET /` → 200 with `status: "healthy"`, `loaded: True`, `model: "CustomerChurnEnsemble"`. The `loaded` field is the "lifespan succeeded" signal |
| `test_model_info` | `GET /model/info` → 200 with `registered_model`, `run_id`, and a `flavors` dict containing `python_function`. Verifies the stub's `SimpleNamespace` matches the real `mlflow.models.Model.metadata` shape |

### `TestPredict` (4 tests)

| Test | Asserts |
|------|---------|
| `test_single_predict_returns_stub_probability` | `POST /predict` → 200 with `churn_probability ≈ 0.42` (`abs=1e-6`) and `churn: False`. Tight tolerance because the stub is deterministic |
| `test_single_predict_rejects_invalid_payload` | Invalid enum value → **422** (Pydantic validation surfaced over HTTP). Any other status means validation fired in the wrong layer |
| `test_batch_predict_returns_one_per_input` | Two identical payloads → two predictions out, both `0.42 / False`. Using identical inputs deliberately catches any accidental `drop_duplicates` in `preprocess` |
| `test_batch_predict_rejects_empty_list` | `POST /predict/batch` with `[]` → **400** (not 200 with empty list). An empty batch is almost always a client bug; the API surfaces it loudly |

### `TestPreprocess` (1 test, no HTTP)

| Test | Asserts |
|------|---------|
| `test_preprocess_produces_engineered_features` | `preprocess([CustomerData(...)])` returns a `DataFrame` with `AvgMonthlyCharge`, `TotalServices`, and binary-encoded YesNo fields (`Partner == 1`, `PhoneService == 1`). `TotalServices == 4` because the fixture has exactly 4 `"Yes"` values across the service columns (`PhoneService`, `OnlineSecurity`, `DeviceProtection`, `TechSupport`) |

This is the one test that exercises the **training/serving preprocessing parity** invariant called out in [`WEB_SERVICE.md`](WEB_SERVICE.md): the request path must apply the exact same feature engineering as `train.py`, or the model receives an out-of-distribution input.

## File 3 — [`test_monitor.py`](../tests/test_monitor.py)

Tests the **pure-function pieces** of [`monitoring/monitor.py`](../monitoring/monitor.py) only. The long-running `run()` replay loop talks to Postgres, the API, and the dataset CSVs — that belongs in an integration test, not here.

The import block at the top of the file intentionally fails loudly on `ImportError`. A previous `try/except → skip` block masked an Evidently API regression and let coverage tank silently; the noisy-failure pattern prevents a recurrence.

### `_build_injector` helper

Constructs a `DriftInjector` with a **5-row reference DataFrame** that spans every category for every categorical feature (the injector samples from the reference when perturbing, so under-covered categories would cause the test to flake). Fixed `seed=7` keeps the RNG deterministic.

### `TestDriftInjectorIntensity` (4 tests)

Verifies `intensity_for(batch_index)` for each `DRIFT_MODE`. The intensity scalar `∈ [0, 1]` modulates three perturbations in lockstep (numeric shift, categorical swap probability, missing-value rate), so any arithmetic error here compounds across all three.

| Test | Mode | Schedule asserted |
|------|------|-------------------|
| `test_none_mode_is_disabled` | `none` | `enabled is False` **and** intensity is 0 for every batch. The `enabled` flag is the fast-path gate in `run()` |
| `test_sustained_mode_is_full_after_start` | `sustained` | 0 before `start_batch=2`, then **1.0 forever** (checked through batch 50). Confirms no decay |
| `test_gradual_mode_ramps_to_one` | `gradual, ramp=4` | 0 → 0.25 → 0.5 → 1.0 over batches 1→5; plateaus at 1.0 afterward. Specific fractional values catch off-by-one ramp errors |
| `test_unknown_mode_returns_zero` | `"does-not-exist"` | Returns 0.0 (graceful degradation — a typo in `DRIFT_MODE` env var should not crash the monitor) |

### `TestToFeatures` (1 test)

| Test | Asserts |
|------|---------|
| `test_drops_non_feature_columns_and_coerces_numeric` | (1) `id`, `gender`, `Churn` are dropped (the API's Pydantic schema rejects these). (2) `tenure` and `SeniorCitizen` string cells are coerced to `int` dtype. (3) Whitespace-only `MonthlyCharges` / `TotalCharges` cells map to `0.0` — matches `train.py`'s missing-data convention |

## What is intentionally not covered

Coverage configuration in [`pyproject.toml`](../../pyproject.toml) explicitly omits three paths:

| Path | Why omitted |
|------|-------------|
| `project/tests/*` | Standard practice — tests should not measure themselves |
| `project/train.py` | Orchestration script: needs MLflow at `:5001`, Optuna with SQLite, the dataset CSVs, ~minutes of compute. Belongs in an integration / nightly job, not unit tests. The CI workflow [`train.yml`](../../.github/workflows/train.yml) runs it end-to-end on demand |
| `project/monitoring/grafana/*` | JSON dashboard definitions and provisioning YAML — not Python |

Other deliberate gaps:

- **`monitor.run()` replay loop** — pure-function helpers are tested; the loop itself (Postgres writes, API HTTP, batch pacing) needs the full Compose stack.
- **MLflow registry roundtrip** — `_load_registry_model` is monkey-patched in tests. The real loader is exercised by the API on container startup and is implicitly validated by the smoke pattern in [`docs/CI_CD.md`](../../docs/CI_CD.md).
- **Grafana dashboard rendering** — covered by manual inspection during `docker compose up`; not automatable without headless browser infra.

## Coverage gate

[`pyproject.toml`](../../pyproject.toml) sets `--cov-fail-under` indirectly via the CI invocation. The actual command in [`ci.yml`](../../.github/workflows/ci.yml) is:

```bash
uv run pytest project/tests \
  --cov=project \
  --cov-report=term \
  --cov-report=xml \
  --cov-fail-under=50
```

Coverage XML is uploaded as a workflow artifact (`coverage-3.13`) on every run, including failed runs (`if: always()` in the upload step), so it can be inspected in the GitHub UI or fed to an external coverage service.

Excluded lines (in addition to omitted files) per `[tool.coverage.report]`:

- `pragma: no cover`
- `if __name__ == "__main__":`
- `raise NotImplementedError`

## Adding a new test

1. **Put it in the right file**: schemas in `test_schemas.py`, anything reachable via HTTP in `test_web_service.py`, monitor helpers in `test_monitor.py`. New top-level modules get their own `test_<module>.py`.
2. **Reuse `valid_customer_payload`** rather than building a new dict — keep the schema-change blast radius to one place.
3. **For new endpoints, reuse the `client` fixture** so the stub model and lifespan handling come for free.
4. **For pure functions, no fixture needed** — import and call directly. Mark with `pytest.mark.parametrize` if you have a table-driven case.
5. **Deterministic data only** — fix RNG seeds (`np.random.default_rng(seed)`, `random.seed(...)`) for anything probabilistic.
6. **Run locally** with the CI command before pushing:
   ```bash
   PYTHONPATH=project uv run pytest project/tests \
     --cov=project --cov-report=term --cov-fail-under=50
   ```

## Related documentation

- [Web service](WEB_SERVICE.md) — what the API tests exercise
- [Pydantic schemas](SCHEMAS.md) — what the schema tests gate
- [Monitoring pipeline](MONITORING.md) — the loop the monitor tests do **not** cover
- [CI/CD architecture](../../docs/CI_CD.md) — how this suite plugs into the gating pipeline
- [Training pipeline](TRAIN.md) — the orchestration code the suite intentionally skips
