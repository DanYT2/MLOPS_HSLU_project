"""FastAPI endpoint tests with the MLflow registry loader stubbed.

The real web service loads a ``ChurnEnsembleModel`` pyfunc from the
MLflow Model Registry on startup (see ``web_service.lifespan``). That
load:

- requires a reachable MLflow tracking server
- deserializes ten joblib-pickled fold models (~30s cold)
- aborts startup with a clear error if the ``champion`` alias isn't
  pointed at a real run

None of those preconditions hold in CI. To make these endpoint tests
hermetic and fast, we monkeypatch ``web_service._load_registry_model``
*before* constructing FastAPI's ``TestClient``, so the lifespan handler
receives a stub and stores it in ``models_state["model"]`` exactly the
way the real model would be stored.

Test groups:

1. ``TestHealthAndInfo`` — the metadata endpoints (``GET /``,
   ``GET /model/info``). Verifies the lifespan ran, the stub model was
   wired in, and the ``/model/info`` reader walks the metadata fields
   it expects.

2. ``TestPredict`` — the inference endpoints (``POST /predict``,
   ``POST /predict/batch``). Each test asserts both the happy path
   (stub probability flows out as JSON) and one error path (validation
   422, empty-list 400).

3. ``TestPreprocess`` — exercises the feature-engineering function
   directly (no HTTP). Confirms the engineered columns
   (``AvgMonthlyCharge``, ``TotalServices``) and the binary encoding
   step both fire on the sample payload.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, stub_model):
    """A FastAPI ``TestClient`` whose lifespan loaded the stub model.

    Two non-obvious things going on here:

    1. We ``import web_service`` *inside* the fixture, not at module
       scope. That defers the import until pytest has assembled its
       fixture graph and applied the conftest sys.path patch — importing
       at the top of this file would race the path setup on some
       runner configurations.

    2. ``monkeypatch.setattr`` replaces the registry loader for the
       duration of one test. We use a lambda that accepts any args
       (``*a, **kw``) so the patched signature matches the real
       function's, including the ``alias="champion"`` kwarg the
       lifespan passes.

    The ``with TestClient(app)`` context triggers FastAPI's lifespan:
    startup (which calls our patched loader → stub_model) on enter,
    shutdown on exit. ``yield`` hands control to the test.
    """
    import web_service

    monkeypatch.setattr(
        web_service, "_load_registry_model", lambda *a, **kw: stub_model
    )
    with TestClient(web_service.app) as test_client:
        yield test_client


class TestHealthAndInfo:
    """Endpoints that report state rather than do inference.

    These are the first things an oncall or smoke test hits, so we keep
    the asserts tight and meaningful: a 200 with the right shape is the
    'lifespan succeeded and the model is loaded' signal.
    """

    def test_health_check(self, client):
        """``GET /`` returns ``status: healthy`` + the loaded model name.

        ``loaded: True`` is the field that proves the lifespan reached
        ``models_state["model"] = ...``. The ``model`` field is the
        registered name (``CustomerChurnEnsemble``) hard-coded in
        ``web_service.REGISTERED_MODEL_NAME``.
        """
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["loaded"] is True
        assert body["model"] == "CustomerChurnEnsemble"

    def test_model_info(self, client):
        """``GET /model/info`` reads through to ``model.metadata``.

        The fields asserted here are the exact attributes
        ``_StubChurnModel`` populates in its ``SimpleNamespace`` — see
        conftest.py. If this test ever fails, the most likely cause is
        that ``web_service`` reached for a metadata attribute the stub
        doesn't expose, which means the stub interface drifted from the
        real ``mlflow.models.Model`` shape.
        """
        response = client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["registered_model"] == "CustomerChurnEnsemble"
        assert body["run_id"] == "test-run-id"
        assert "python_function" in body["flavors"]


class TestPredict:
    """Inference endpoints — the actual prediction path.

    The stub model returns a fixed 0.42 for every row regardless of
    input, so we can assert exact equality on the probability and the
    derived churn boolean (0.42 < 0.5 → False). Real model variability
    would force these assertions into approximate ranges and dilute
    the signal.
    """

    def test_single_predict_returns_stub_probability(
        self, client, valid_customer_payload
    ):
        """Happy path for ``POST /predict``.

        Round-trip: valid payload → 200 with the expected probability
        and churn=False. ``abs=1e-6`` rather than the default relative
        tolerance because we're asserting against a known constant and
        want the assertion to fail loudly if anything in the path adds
        even a tiny numerical perturbation (e.g., a future thresholding
        step that recomputes the probability).
        """
        response = client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["churn_probability"] == pytest.approx(0.42, abs=1e-6)
        assert body["churn"] is False  # 0.42 < 0.5

    def test_single_predict_rejects_invalid_payload(
        self, client, valid_customer_payload
    ):
        """Invalid enum value should fail at the schema layer (422).

        FastAPI maps Pydantic ``ValidationError`` to HTTP 422
        Unprocessable Entity. Anything other than 422 here means the
        validation either fired too late (e.g., a 500 from preprocess)
        or didn't fire at all (a 200 with garbage prediction). The test
        in ``test_schemas.py`` covers the same case at the Pydantic
        layer; this one ensures FastAPI surfaces it correctly over HTTP.
        """
        valid_customer_payload["Contract"] = "forever"
        response = client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 422

    def test_batch_predict_returns_one_per_input(self, client, valid_customer_payload):
        """``POST /predict/batch`` returns N predictions for N inputs.

        We deliberately use *the same* payload twice rather than two
        different customers — the stub model is deterministic so both
        outputs should be identical, and asserting that explicitly
        guards against any future code that accidentally collapses
        duplicate inputs (e.g., a pandas ``drop_duplicates`` call in
        preprocess would silently halve the response length).
        """
        payload = [valid_customer_payload, valid_customer_payload]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2
        for prediction in body["predictions"]:
            assert prediction["churn_probability"] == pytest.approx(0.42, abs=1e-6)
            assert prediction["churn"] is False

    def test_batch_predict_rejects_empty_list(self, client):
        """Empty batch → 400, not a 200 with an empty predictions list.

        An empty inference request is almost always a client bug, and
        responding ``200 OK [] no problem`` masks it. The web service
        explicitly rejects ``[]`` with 400 Bad Request so the client
        sees the failure immediately.
        """
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 400


class TestPreprocess:
    """Tests for ``web_service.preprocess`` independent of HTTP.

    The preprocess function is the one piece of inference logic that
    sits between FastAPI validation and the model. It has to produce
    feature columns matching the column order the model was trained on
    — a mismatch silently corrupts predictions. Direct unit tests are
    much faster than end-to-end checks and pinpoint regressions to a
    specific engineered feature.
    """

    def test_preprocess_produces_engineered_features(self, valid_customer_payload):
        """Asserts the two derived features and the binary encoding step.

        We re-import the schemas + preprocess function inside the test
        for the same reason the ``client`` fixture defers its import —
        keeping the test's import surface minimal and visible.

        The ``TotalServices`` count is derived from 8 service columns
        (PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies — minus duplicates depending on encoding).
        With the fixture payload, exactly 4 of these are 'Yes':
        PhoneService, OnlineSecurity, DeviceProtection, TechSupport. If
        a future refactor adds or drops one of the counted columns,
        this number changes and the test will surface it.

        The ``Partner == 1`` and ``PhoneService == 1`` checks confirm
        that ``YesNo`` enum values are binary-encoded for the model,
        not passed through as the string ``"Yes"``.
        """
        from schemas.schemas import CustomerData
        from web_service import preprocess

        df = preprocess([CustomerData(**valid_customer_payload)])

        assert "AvgMonthlyCharge" in df.columns
        assert "TotalServices" in df.columns
        # 8 service columns; the example has 4 "Yes" values (PhoneService,
        # OnlineSecurity, DeviceProtection, TechSupport).
        assert int(df.loc[0, "TotalServices"]) == 4
        assert df.loc[0, "Partner"] == 1  # binary-encoded "Yes"
        assert df.loc[0, "PhoneService"] == 1
