"""FastAPI endpoint tests with the MLflow registry loader stubbed.

The real web service loads a ChurnEnsemble pyfunc model from MLflow on startup.
We can't reach a registry in CI, so we monkeypatch ``_load_registry_model``
before constructing the ``TestClient`` (which triggers FastAPI's lifespan).
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, stub_model):
    import web_service

    monkeypatch.setattr(web_service, "_load_registry_model", lambda *a, **kw: stub_model)
    with TestClient(web_service.app) as test_client:
        yield test_client


class TestHealthAndInfo:
    def test_health_check(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["loaded"] is True
        assert body["model"] == "CustomerChurnEnsemble"

    def test_model_info(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200
        body = response.json()
        assert body["registered_model"] == "CustomerChurnEnsemble"
        assert body["run_id"] == "test-run-id"
        assert "python_function" in body["flavors"]


class TestPredict:
    def test_single_predict_returns_stub_probability(self, client, valid_customer_payload):
        response = client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 200
        body = response.json()
        assert body["churn_probability"] == pytest.approx(0.42, abs=1e-6)
        assert body["churn"] is False  # 0.42 < 0.5

    def test_single_predict_rejects_invalid_payload(self, client, valid_customer_payload):
        valid_customer_payload["Contract"] = "forever"
        response = client.post("/predict", json=valid_customer_payload)
        assert response.status_code == 422

    def test_batch_predict_returns_one_per_input(self, client, valid_customer_payload):
        payload = [valid_customer_payload, valid_customer_payload]
        response = client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert len(body["predictions"]) == 2
        for prediction in body["predictions"]:
            assert prediction["churn_probability"] == pytest.approx(0.42, abs=1e-6)
            assert prediction["churn"] is False

    def test_batch_predict_rejects_empty_list(self, client):
        response = client.post("/predict/batch", json=[])
        assert response.status_code == 400


class TestPreprocess:
    def test_preprocess_produces_engineered_features(self, valid_customer_payload):
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
