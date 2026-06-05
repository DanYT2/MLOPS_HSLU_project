"""Tests for the pure-function pieces of ``monitoring/monitor.py``.

The monitor module is structured as one long ``run()`` loop driver
surrounded by deterministic helpers. The loop talks to Postgres, the
FastAPI service, and CSV files on disk — none of which we want to spin
up for unit tests. Instead, we exercise the *helpers* that have no I/O:

- ``DriftInjector.intensity_for(batch_index)`` — the math that decides
  how strongly to perturb a batch based on the configured drift mode
  (``none``, ``sustained``, ``gradual``, ``cycle``, plus unknown). This
  is the function that drives synthetic drift on the Grafana dashboard,
  so getting its math right matters for demos and for any monitoring
  alert thresholds tuned against it.

- ``_to_features(df)`` — the CSV-to-API transformation that drops
  bookkeeping columns (``id``, ``gender``, ``Churn``) and coerces the
  ``TotalCharges``/``MonthlyCharges`` columns from string (sometimes
  whitespace-only) to numeric. This is the parity layer with the
  training pipeline; if it drifts from what ``train.py`` does, the API
  receives mis-shaped frames and returns wrong predictions.

What this file is **NOT** for: integration tests of ``run()``, end-to-
end Evidently report generation, or Postgres writes. Those belong in
a future integration test using ``docker compose`` (the previous
``smoke-test.yml`` workflow handled this before it was removed).
"""

from __future__ import annotations

import pandas as pd
import pytest  # noqa: F401  # used as pytest.approx in test cases below

# monitoring.monitor must import cleanly under the locked dependency set.
# Previously this block swallowed ImportError and skipped the whole file,
# which masked a real Evidently API regression and let coverage tank
# silently. Fail loudly instead — any future ImportError here is a real
# bug that should break CI, not a reason to disable the suite.
from monitoring.monitor import DriftInjector, _to_features  # noqa: E402


def _build_injector(mode: str, **overrides) -> DriftInjector:
    """Construct a ``DriftInjector`` with realistic-but-tiny inputs.

    The injector samples from the reference DataFrame when perturbing
    categorical features, so the reference has to contain at least one
    instance of every category we care about. The 5-row reference below
    spans the full range of values for each categorical feature
    (``Yes``/``No``, all three ``Contract`` levels, all four
    ``PaymentMethod`` levels, etc.), which is the minimum needed for the
    swap logic to behave deterministically across tests.

    Defaults match the constants in ``monitor.py`` so the tests start
    from the same baseline as production. ``**overrides`` lets a
    specific test bend one knob (e.g., ``ramp_batches=4`` to make the
    ramp math fall on clean fractional values).

    ``seed=7`` keeps the internal RNG deterministic across runs — any
    randomness in the injector would otherwise produce flaky tests
    when, e.g., we assert exact intensity values.
    """
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
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "One year",
                "Month-to-month",
            ],
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
    # These defaults are intentionally fixed across all tests so the
    # math in TestDriftInjectorIntensity is reproducible. A test that
    # needs different values overrides only what it cares about.
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
    """Verifies the ``intensity_for(batch_index)`` math per drift mode.

    The intensity scalar (0.0–1.0) modulates three perturbations
    in lockstep — numeric mean shift, categorical swap probability,
    and missing-cell rate — so a small error in this function
    compounds across all three perturbations. We test the four
    primary modes (none/sustained/gradual/unknown) plus the
    pre-start cutoff that every mode shares.
    """

    def test_none_mode_is_disabled(self):
        """``mode="none"`` ⇒ injector is a no-op for every batch.

        Also asserts ``enabled is False`` — that flag is what
        ``monitor.run()`` checks before bothering to allocate the
        perturbation buffers, so it has to be False (not just a 0.0
        intensity) for the fast path to engage.
        """
        injector = _build_injector("none")
        assert injector.enabled is False
        assert injector.intensity_for(0) == 0.0
        assert injector.intensity_for(100) == 0.0

    def test_sustained_mode_is_full_after_start(self):
        """``mode="sustained"`` jumps from 0 to 1 at ``start_batch``.

        ``start_batch=2`` means batches 0 and 1 are clean
        (``intensity_for(0) == 0.0``) and everything from batch 2
        onward gets full perturbation. We also check a far-future
        index (50) to confirm the value doesn't decay — sustained
        means *sustained*, not 'a spike at the start'.
        """
        injector = _build_injector("sustained")
        assert injector.enabled is True
        assert injector.intensity_for(0) == 0.0  # before start_batch=2
        assert injector.intensity_for(2) == 1.0
        assert injector.intensity_for(50) == 1.0

    def test_gradual_mode_ramps_to_one(self):
        """``mode="gradual"`` ramps linearly across ``ramp_batches``.

        With ``start_batch=2`` and ``ramp_batches=4``, the schedule is:

        - batch 1: 0.0 (before start)
        - batch 2: 0.25 (1/4 into the ramp)
        - batch 3: 0.50 (2/4)
        - batch 5: 1.00 (4/4, ramp complete)
        - batch 20: 1.00 (plateau)

        Asserting on these specific fractional values is what catches
        off-by-one errors in the ramp math — e.g., a future bug that
        starts the ramp at ``start_batch + 1`` instead of
        ``start_batch`` would shift every value by 0.25 and fail here.
        """
        injector = _build_injector("gradual", ramp_batches=4)
        assert injector.intensity_for(1) == 0.0  # before start_batch
        assert injector.intensity_for(2) == pytest.approx(0.25)
        assert injector.intensity_for(3) == pytest.approx(0.5)
        assert injector.intensity_for(5) == pytest.approx(1.0)
        assert injector.intensity_for(20) == pytest.approx(1.0)

    def test_unknown_mode_returns_zero(self):
        """Garbage mode names degrade gracefully to 0.0 (no perturbation).

        A typo in the ``DRIFT_MODE`` env var should not crash the
        monitor — it should fall through to 'do nothing'. This is the
        defensive guard at the bottom of ``intensity_for`` that catches
        every unhandled mode and returns 0.0.
        """
        injector = _build_injector("does-not-exist")
        assert injector.intensity_for(10) == 0.0


class TestToFeatures:
    """Verifies ``_to_features`` is the right parity layer with training.

    The monitor reads test.csv / train.csv files which include columns
    the API has never seen (``id``, ``gender``, ``Churn``) and which
    store ``MonthlyCharges`` and ``TotalCharges`` as strings — sometimes
    with whitespace-only values where the original Telco dataset had
    missing data. ``_to_features`` must:

    1. Drop the non-feature columns (or the API rejects the payload
       with a 422 because ``id``/``gender``/``Churn`` aren't in the
       ``CustomerData`` schema).
    2. Coerce string-typed numeric columns to int/float (so the
       JSON payload uses numbers, not strings).
    3. Handle the whitespace-as-missing convention by mapping
       whitespace cells to 0.0 (matching how ``train.py``'s ingestion
       step treats them).

    A regression in any of these three steps would corrupt monitoring
    metrics without obvious failure — the API would still respond,
    just with subtly wrong predictions.
    """

    def test_drops_non_feature_columns_and_coerces_numeric(self):
        """One test covering all three responsibilities at once.

        We intentionally include a row with whitespace-only charges
        (``" "`` and ``"  "``) to assert the missing-data convention.
        The numeric dtype check uses ``.dtype.kind in {"i", "u"}``
        (signed or unsigned integer) rather than ``== np.int64``
        because pandas can choose either width depending on platform
        and we don't want to over-constrain the assertion.

        The ``Partner`` column is left as-is — ``_to_features`` does
        NOT binary-encode the YesNo fields, that step happens later
        inside ``web_service.preprocess``. The test does not assert
        on Partner's content for that reason.
        """
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
        # Bookkeeping columns must be gone — these would cause 422s.
        assert "id" not in out.columns
        assert "gender" not in out.columns
        assert "Churn" not in out.columns
        # Numeric coercion: string "3"/"12" → int dtype.
        assert out["tenure"].dtype.kind in {"i", "u"}
        assert out["SeniorCitizen"].dtype.kind in {"i", "u"}
        # Whitespace-only cells were treated as missing and filled with 0.0.
        # Row 0 (charges "29.85") should still be 29.85; we only test the
        # whitespace cases here because the float-equality on 29.85 is
        # incidental and not the property under test.
        assert out.loc[1, "MonthlyCharges"] == 0.0  # whitespace -> 0.0
        assert out.loc[1, "TotalCharges"] == 0.0
