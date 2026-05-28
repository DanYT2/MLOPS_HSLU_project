"""Shared pytest fixtures.

The web service module imports schemas via ``from schemas.schemas import ...``,
which only resolves when ``project/`` is on ``sys.path``. ``pytest.ini_options``
in ``pyproject.toml`` adds it, but we re-add here as a safety net so the suite
also runs with bare ``pytest`` invocations from ``project/``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


class _StubChurnModel:
    """Minimal stand-in for the loaded MLflow pyfunc churn model.

    The real model loads ten serialized fold models from the registry. For unit
    tests we only need an object that exposes ``predict`` (returning a 1D float
    array of length ``len(df)``) and a ``metadata`` namespace shaped like the
    one the ``/model/info`` endpoint reads.
    """

    def __init__(self, fixed_prob: float = 0.42) -> None:
        self.fixed_prob = fixed_prob
        self.metadata = SimpleNamespace(
            flavors={"python_function": {}},
            run_id="test-run-id",
            artifact_path="model",
        )

    def predict(self, df) -> np.ndarray:  # noqa: ANN001 - mirrors mlflow signature
        return np.full(len(df), self.fixed_prob, dtype=float)


@pytest.fixture()
def stub_model() -> _StubChurnModel:
    """A deterministic model used by web service tests."""
    return _StubChurnModel()


@pytest.fixture()
def valid_customer_payload() -> dict:
    """Schema-valid customer payload pulled from the OpenAPI example."""
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
