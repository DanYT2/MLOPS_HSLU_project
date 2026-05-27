"""Tests for pure-function pieces of the monitoring pipeline.

We avoid exercising the loop-driver (``run()``) because it expects Postgres,
the FastAPI service, and CSVs. Instead we exercise the deterministic helpers:
``DriftInjector.intensity_for`` and ``_to_features``.
"""
from __future__ import annotations

import pandas as pd
import pytest

# monitoring.monitor pulls in evidently + psycopg (libpq) at import time. Both
# are heavy/optional in CI environments and Evidently's public API has changed
# names across versions, so we skip the whole module if any of those imports
# fail rather than failing collection.
try:
    from monitoring.monitor import DriftInjector, _to_features
except ImportError as exc:
    pytest.skip(f"monitoring.monitor unavailable: {exc}", allow_module_level=True)


def _build_injector(mode: str, **overrides) -> DriftInjector:
    reference = pd.DataFrame(
        {
            "SeniorCitizen": [0, 1, 0, 1, 0],
            "tenure": [1, 12, 24, 36, 48],
            "MonthlyCharges": [20.0, 50.0, 70.0, 90.0, 110.0],
            "TotalCharges": [20.0, 600.0, 1680.0, 3240.0, 5280.0],
            "Partner": ["Yes", "No", "Yes", "No", "Yes"],
            "Dependents": ["No", "No", "Yes", "Yes", "No"],
            "PhoneService": ["Yes"] * 5,
            "MultipleLines": ["No"] * 5,
            "InternetService": ["DSL", "Fiber optic", "No", "DSL", "Fiber optic"],
            "OnlineSecurity": ["Yes", "No", "No internet service", "Yes", "No"],
            "OnlineBackup": ["Yes", "No", "No internet service", "Yes", "No"],
            "DeviceProtection": ["Yes", "No", "No internet service", "Yes", "No"],
            "TechSupport": ["Yes", "No", "No internet service", "Yes", "No"],
            "StreamingTV": ["Yes", "No", "No internet service", "Yes", "No"],
            "StreamingMovies": ["Yes", "No", "No internet service", "Yes", "No"],
            "Contract": ["Month-to-month", "One year", "Two year", "One year", "Month-to-month"],
            "PaperlessBilling": ["Yes"] * 5,
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
            ],
        }
    )
    kwargs = {
        "start_batch": 2,
        "ramp_batches": 4,
        "period_batches": 6,
        "numeric_shift_std": 1.5,
        "categorical_swap_prob": 0.4,
        "missing_rate": 0.0,
        "reference": reference,
        "seed": 7,
    }
    kwargs.update(overrides)
    return DriftInjector(mode, **kwargs)


class TestDriftInjectorIntensity:
    def test_none_mode_is_disabled(self):
        injector = _build_injector("none")
        assert injector.enabled is False
        assert injector.intensity_for(0) == 0.0
        assert injector.intensity_for(100) == 0.0

    def test_sustained_mode_is_full_after_start(self):
        injector = _build_injector("sustained")
        assert injector.enabled is True
        assert injector.intensity_for(0) == 0.0  # before start_batch=2
        assert injector.intensity_for(2) == 1.0
        assert injector.intensity_for(50) == 1.0

    def test_gradual_mode_ramps_to_one(self):
        injector = _build_injector("gradual", ramp_batches=4)
        assert injector.intensity_for(1) == 0.0  # before start_batch
        assert injector.intensity_for(2) == pytest.approx(0.25)
        assert injector.intensity_for(3) == pytest.approx(0.5)
        assert injector.intensity_for(5) == pytest.approx(1.0)
        assert injector.intensity_for(20) == pytest.approx(1.0)

    def test_unknown_mode_returns_zero(self):
        injector = _build_injector("does-not-exist")
        assert injector.intensity_for(10) == 0.0


class TestToFeatures:
    def test_drops_non_feature_columns_and_coerces_numeric(self):
        df = pd.DataFrame(
            {
                "id": [1, 2],
                "gender": ["Male", "Female"],
                "Churn": ["Yes", "No"],
                "tenure": ["3", "12"],
                "SeniorCitizen": ["0", "1"],
                "MonthlyCharges": ["29.85", " "],
                "TotalCharges": ["29.85", "  "],
                "Partner": ["Yes", "No"],
            }
        )
        out = _to_features(df)
        assert "id" not in out.columns
        assert "gender" not in out.columns
        assert "Churn" not in out.columns
        assert out["tenure"].dtype.kind in {"i", "u"}
        assert out["SeniorCitizen"].dtype.kind in {"i", "u"}
        assert out.loc[1, "MonthlyCharges"] == 0.0  # whitespace -> 0.0
        assert out.loc[1, "TotalCharges"] == 0.0
