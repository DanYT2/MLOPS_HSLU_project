"""Shared pytest fixtures for the project test suite.

This module is auto-discovered by pytest and applies to every test file in
``project/tests/``. It serves three purposes:

1. **Import-path safety net.** The production code imports schemas via
   ``from schemas.schemas import ...``, which only resolves when
   ``project/`` is on ``sys.path``. This is also wired in
   ``pyproject.toml`` (``[tool.pytest.ini_options] pythonpath = ["project"]``)
   and re-asserted in CI via ``PYTHONPATH=project``. The block below is the
   third belt-and-braces — it lets the suite run from a bare ``pytest``
   invocation inside ``project/`` or via a tooling integration that
   bypasses pyproject's config.

2. **Stub model fixture (``stub_model``).** The real web service loads ten
   serialized fold models from the MLflow Model Registry on startup. CI
   runners have no registry to talk to, so tests substitute a tiny stand-in
   that mimics the loaded pyfunc's public surface (a ``.predict`` method
   and a ``.metadata`` object). This avoids both the MLflow network round
   trip and the ~30s deserialization cost.

3. **Shared sample payload (``valid_customer_payload``).** Every schema and
   endpoint test starts from this dict. Centralizing it ensures we test
   against a consistent example and that changes to the schema (e.g., a
   new required field) break tests in one place rather than scattered
   across modules.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# ── sys.path safety net ──────────────────────────────────────────
# Resolve the absolute path to project/ (this file lives in project/tests/).
# parents[1] is project/. Inserting at index 0 puts it ahead of any
# system-installed package that might shadow our schemas module.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ── Stub model used by web service tests ─────────────────────────
class _StubChurnModel:
    """Minimal stand-in for the loaded MLflow pyfunc churn model.

    The real model (``ChurnEnsembleModel`` registered as
    ``CustomerChurnEnsemble``) wraps ten fold models — five LightGBM and
    five XGBoost — and averages their predictions on call. For unit tests
    we don't care about the ensemble math; we only need an object whose
    interface matches what the FastAPI handlers call:

    - ``predict(df)`` must return a 1D numeric array with one entry per
      input row. The handlers consume ``predictions[0]`` for single-row
      requests or iterate the full array for batch requests.
    - ``metadata`` must expose the attributes the ``/model/info`` endpoint
      reads: ``flavors`` (dict), ``run_id`` (str), ``artifact_path`` (str).
      ``SimpleNamespace`` is a lightweight shim that gives attribute access
      to those keys without writing a dataclass.

    Using a *fixed* probability (default ``0.42``) makes every endpoint
    assertion deterministic: a probability of 0.42 means
    ``churn = False`` (the threshold is 0.5), so the test can assert both
    values explicitly with no flakiness.
    """

    def __init__(self, fixed_prob: float = 0.42) -> None:
        self.fixed_prob = fixed_prob
        # SimpleNamespace mirrors the shape of mlflow.models.Model.metadata.
        # /model/info reads exactly these three fields, so anything else
        # we omit is genuinely unused at the test boundary.
        self.metadata = SimpleNamespace(
            flavors={"python_function": {}},
            run_id="test-run-id",
            artifact_path="model",
        )

    def predict(self, df) -> np.ndarray:  # noqa: ANN001 - mirrors mlflow signature
        # Returning ``np.full(len(df), p)`` rather than ``[p] * len(df)``
        # matches the real model's dtype (float64 numpy array), so the
        # FastAPI response serializer treats both identically.
        return np.full(len(df), self.fixed_prob, dtype=float)


@pytest.fixture()
def stub_model() -> _StubChurnModel:
    """A deterministic stand-in for the registered churn model.

    Used by ``test_web_service.py``'s ``client`` fixture, which
    monkeypatches ``web_service._load_registry_model`` so the FastAPI
    lifespan hands this stub to ``models_state["model"]`` instead of
    reaching MLflow.
    """
    return _StubChurnModel()


@pytest.fixture()
def valid_customer_payload() -> dict:
    """A schema-valid customer dict, taken from the API's OpenAPI example.

    Tests build on this dict — schema tests mutate one field at a time
    to trigger validation errors; web service tests POST it unchanged to
    ``/predict``; the preprocess test feeds it through the feature
    pipeline to assert engineered columns appear.

    The values are chosen so the resulting preprocess output is
    interesting (4 of the 8 "service" columns are "Yes", so
    ``TotalServices`` should be 4 — see the preprocess test).
    """
    return {
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 29,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 60.10,
        "TotalCharges": 1653.85,
    }
