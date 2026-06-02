File 1 — conftest.py (shared infrastructure, not a test file)

  Auto-loaded by pytest. Three responsibilities:

  ┌─────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │   Responsibility    │                                           Why it matters                                            │
  ├─────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │                     │ Production code imports from schemas.schemas import …, which only resolves when project/ is on      │
  │ sys.path safety net       │ on sys.path. Three places wire this: pyproject's [tool.pytest.ini_options] pythonpath, CI's   │
  │                           │ PYTHONPATH=project env var, and this conftest. The third is the safety net for bare pytest    │
  │                           │ invocations.                                                                                  │
  ├───────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │                           │ Real model loading needs a reachable MLflow registry and ~30s to deserialize 10 fold models.  │
  │ _StubChurnModel +         │ Stub has the minimum interface the FastAPI code calls: .predict(df) returns a 1-D float       │
  │ stub_model fixture        │ array, .metadata exposes flavors / run_id / artifact_path for /model/info. Fixed probability  │
  │                           │ 0.42 ⇒ all churn assertions are deterministic (False, since 0.42 < 0.5).                      │
  ├───────────────────────────┼───────────────────────────────────────────────────────────────────────────────────────────────┤
  │ valid_customer_payload    │ Schema-valid customer dict from the OpenAPI example. Schema tests mutate it one field at a    │
  │ fixture                   │ time; endpoint tests POST it unchanged; preprocess test feeds it through the feature          │
  │                           │ pipeline. Centralized so a schema change breaks one place.                                    │
  └───────────────────────────┴───────────────────────────────────────────────────────────────────────────────────────────────┘

  File 2 — test_schemas.py (8 tests, 2 classes)

  Gates the request/response Pydantic boundary. Every HTTP request passes through CustomerData; every response goes out through
  PredictionResponse / BatchPredictionResponse.

  Class TestCustomerData (6 tests) — pattern is baseline + one mutation per test:

  ┌─────────────────────────────────┬────────────────────────────────┬────────────────────────────────────────────────────────┐
  │              Test               │        What it asserts         │                          Why                           │
  ├─────────────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────┤
  │                                     │ Baseline parses; enum-backed   │ If this fails, every other test in the class is        │
        │ test_accepts_valid_payload    │ fields land on enum members    │ suspect. The is Contract.one_year check also guards    │
        │                               │ (not raw strings)              │ against accidental use_enum_values=True config.        │
  ├───────────────────────────────┼────────────────────────────────┼────────────────────────────────────────────────────────┤
  │                                     │ Contract = "lifetime" →       │ Without strict validation, the one-hot encoder      │
  │ test_rejects_invalid_enum_value     │ ValidationError               │ downstream would emit all-zeros for unknown         │
  │                                     │                               │ categories — silent garbage prediction.             │
  ├─────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────────┤
  │ test_rejects_negative_tenure        │ tenure = -1 → ValidationError │ Logically impossible customers must be caught at    │
  │                                     │                               │ the boundary, not skew monitoring.                  │
  ├─────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────────┤
  │ test_rejects_negative_charges       │ MonthlyCharges = -10.0 →      │ Same reasoning, currency edition.                   │
  │                                     │ ValidationError               │                                                     │
  ├─────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────────┤
  │ test_rejects_invalid_senior_citizen │ SeniorCitizen = 2 →           │ Telco dataset encodes this as binary integer (0/1); │
  │                                     │ ValidationError               │  anything else would be treated as a magnitude.     │
  ├─────────────────────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────────┤
  │ test_rejects_missing_required_field │ Drop MonthlyCharges →         │ Catches the absence at parse time (HTTP 422) rather │
  │                                     │ ValidationError               │  than as a downstream KeyError (HTTP 500).          │
  └─────────────────────────────────────┴───────────────────────────────┴─────────────────────────────────────────────────────┘

  Class TestResponseSchemas (2 tests) — less stringent because the API constructs these (no untrusted input):

  ┌──────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │               Test               │                                     What it asserts                                     │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ test_prediction_response         │ PredictionResponse(churn_probability, churn) round-trips both fields with correct       │
  │                                  │ types.                                                                                  │
  ├──────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │                                  │ BatchPredictionResponse.predictions is an ordered list of 2 items where index 1 keeps   │
  │ test_batch_prediction_wraps_list │ its churn=True. Guards against a future refactor that accidentally returns a            │
  │                                  │ dict-by-index.                                                                          │
  └──────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────┘

  File 3 — test_web_service.py (8 tests, 3 classes)

  Tests the FastAPI app end-to-end via TestClient, with the registry loader monkey-patched so the lifespan handler receives the
  stub.

  client fixture — the key piece. Two non-obvious bits:
  - import web_service lives inside the fixture (not at module scope) to defer until conftest's sys.path patch has applied.
  - monkeypatch.setattr(web_service, "_load_registry_model", lambda *a, **kw: stub_model) replaces the registry loader for the
  test's duration. *a, **kw matches the real signature including the alias="champion" kwarg.
  - with TestClient(app) triggers FastAPI's lifespan on enter (loads stub) and shutdown on exit.

  Class TestHealthAndInfo (2 tests) — metadata endpoints:

  ┌───────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │       Test        │                                            What it asserts                                             │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ test_health_check │ GET / → 200 with status: healthy, loaded: True, model: "CustomerChurnEnsemble". The loaded field is    │
  │                   │ the "lifespan succeeded" signal.                                                                       │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ test_model_info   │ GET /model/info → 200 with registered_model, run_id, and a flavors dict containing python_function.    │
  │                   │ Verifies the stub's SimpleNamespace matches the real mlflow.models.Model.metadata shape.               │
  └───────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Class TestPredict (4 tests) — inference endpoints:

  ┌──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │                     Test                     │                               What it asserts                               │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_single_predict_returns_stub_probability │ POST /predict → 200 with churn_probability ≈ 0.42 (abs=1e-6) and churn:     │
  │                                              │ False. Tight tolerance because the stub is deterministic.                   │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_single_predict_rejects_invalid_payload  │ Invalid enum value → 422 (Pydantic validation surfaced over HTTP). Anything │
  │                                              │  else means validation fired in the wrong layer.                            │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_batch_predict_returns_one_per_input     │ Two identical payloads in → two predictions out, both 0.42/False. Identical │
  │                                              │  inputs deliberately catch any accidental drop_duplicates in preprocess.    │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_batch_predict_rejects_empty_list        │ POST /predict/batch with [] → 400 (not 200 with empty list). An empty batch │
  │                                              │  is almost always a client bug; the API rejects it.                         │
  └──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘

  Class TestPreprocess (1 test) — feature engineering, no HTTP:

  Test: test_preprocess_produces_engineered_features
  │                                              │ False. Tight tolerance because the stub is deterministic.                   │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_single_predict_rejects_invalid_payload  │ Invalid enum value → 422 (Pydantic validation surfaced over HTTP). Anything │
  │                                              │  else means validation fired in the wrong layer.                            │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ test_model_info   │ GET /model/info → 200 with registered_model, run_id, and a flavors dict containing python_function.    │
  │                   │ Verifies the stub's SimpleNamespace matches the real mlflow.models.Model.metadata shape.               │
  └───────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Class TestPredict (4 tests) — inference endpoints:

  ┌──────────────────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────┐
  │                     Test                     │                               What it asserts                               │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_single_predict_returns_stub_probability │ POST /predict → 200 with churn_probability ≈ 0.42 (abs=1e-6) and churn:     │
  │                                              │ False. Tight tolerance because the stub is deterministic.                   │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_single_predict_rejects_invalid_payload  │ Invalid enum value → 422 (Pydantic validation surfaced over HTTP). Anything │
  │                                              │  else means validation fired in the wrong layer.                            │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_batch_predict_returns_one_per_input     │ Two identical payloads in → two predictions out, both 0.42/False. Identical │
  │                                              │  inputs deliberately catch any accidental drop_duplicates in preprocess.    │
  ├──────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────┤
  │ test_batch_predict_rejects_empty_list        │ POST /predict/batch with [] → 400 (not 200 with empty list). An empty batch │
  │                                              │  is almost always a client bug; the API rejects it.                         │
  └──────────────────────────────────────────────┴─────────────────────────────────────────────────────────────────────────────┘

  Class TestPreprocess (1 test) — feature engineering, no HTTP:

  Test: test_preprocess_produces_engineered_features
  What it asserts: preprocess([CustomerData(...)]) returns a DataFrame with AvgMonthlyCharge, TotalServices, and binary-encoded
    YesNo fields (Partner == 1, PhoneService == 1). TotalServices == 4 because the fixture has exactly 4 'Yes' values in the
    service columns (PhoneService, OnlineSecurity, DeviceProtection, TechSupport).

  File 4 — test_monitor.py (5 tests, 2 classes)

  Tests pure-function pieces of monitoring/monitor.py only — the long run() replay loop talks to Postgres + the API + CSV files
  and belongs in an integration test. The import block at the top intentionally fails loudly on ImportError (the previous
  try/except → skip block masked an Evidently API regression and let coverage tank silently).

  _build_injector helper — constructs a DriftInjector with a small 5-row reference DataFrame that spans every category for every
  categorical feature (needed because the injector samples from the reference when perturbing). Fixed seed=7 keeps the RNG
  deterministic.

  Class TestDriftInjectorIntensity (4 tests) — verifies the intensity_for(batch_index) math per mode. The intensity scalar (0–1)
  modulates three perturbations in lockstep, so any error here compounds.

  ┌─────────────────────────────────────────┬──────────────────┬───────────────────────────────────────────────────────────────┐
  │                  Test                   │       Mode       │                       Schedule asserted                       │
  ├─────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────────────────────────┤
  │ test_none_mode_is_disabled              │ none             │ enabled is False AND intensity is 0 for every batch. The      │
  │                                         │                  │ enabled flag is the fast-path gate in run().                  │
  ├─────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────────────────────────┤
  │ test_sustained_mode_is_full_after_start │ sustained        │ 0 before start_batch=2, then 1.0 forever (including batch     │
  │                                         │                  │ 50). Confirms no decay.                                       │
  ├─────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────────────────────────┤
  │ test_gradual_mode_ramps_to_one          │ gradual, ramp=4  │ 0 → 0.25 → 0.5 → 1.0 over batches 1→5; plateaus at 1.0 after. │
  │                                         │                  │  Specific fractional values catch off-by-one ramp errors.     │
  ├─────────────────────────────────────────┼──────────────────┼───────────────────────────────────────────────────────────────┤
  │ test_unknown_mode_returns_zero          │ "does-not-exist" │ Returns 0.0 (graceful degradation — a typo in DRIFT_MODE env  │
  │                                         │                  │ var should not crash the monitor).                            │
  └─────────────────────────────────────────┴──────────────────┴───────────────────────────────────────────────────────────────┘

  Class TestToFeatures (1 test) — verifies the CSV-to-API transformation:

  ┌────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────┐
  │                        Test                        │                            What it asserts                            │
  ├────────────────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────┤
  │                                                    │ (1) id/gender/Churn are dropped (the API rejects these). (2)          │
  │ test_drops_non_feature_columns_and_coerces_numeric │ tenure/SeniorCitizen strings are coerced to int dtype. (3)            │
  │                                                    │ Whitespace-only MonthlyCharges/TotalCharges cells map to 0.0          │
  │                                                    │ (matching train.py's missing-data convention).                        │
  └────────────────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────┘
