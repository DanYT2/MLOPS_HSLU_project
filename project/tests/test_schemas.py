"""Pydantic model validation tests for ``schemas/schemas.py``.

The web service uses Pydantic models as the request/response boundary.
Every value coming in over HTTP is parsed through ``CustomerData`` (or a
list of them for the batch endpoint), and every prediction going out is
serialized through ``PredictionResponse`` / ``BatchPredictionResponse``.

What this file gates:

- **Type and enum correctness**: the model rejects unknown values for
  fields backed by enums (``Contract``, ``PaymentMethod``,
  ``InternetService``, ``YesNo``). If the API accepted a free-form
  ``Contract`` string, the downstream feature pipeline would silently
  one-hot-encode garbage and the model would emit nonsense without ever
  raising.

- **Range validation**: tenure and charges must be non-negative;
  ``SeniorCitizen`` must be exactly 0 or 1 (the schema encodes it as a
  binary integer rather than a YesNo enum because that's how the
  original Kaggle dataset stores it).

- **Required fields**: removing a required field should fail at parse
  time, not crash deep inside ``preprocess()``.

- **Response models** are simpler — they only have to round-trip the
  values the API generates. We just check that the JSON-serializable
  fields keep their types and that ``BatchPredictionResponse`` is a
  proper container around a list, not a dict-of-dicts.

Tests use the shared ``valid_customer_payload`` fixture defined in
``conftest.py`` as a baseline and mutate one field at a time to isolate
the rejection reason.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

# Importing the enum classes alongside CustomerData lets us assert against
# enum *members* (e.g., ``is Contract.one_year``) rather than raw strings.
# This is stricter — a future refactor that accidentally changes the
# stored value would still pass a string equality check but break here.
from schemas.schemas import (
    BatchPredictionResponse,
    Contract,
    CustomerData,
    InternetService,
    PaymentMethod,
    PredictionResponse,
    YesNo,
)


class TestCustomerData:
    """Validation behaviour of the input model used by ``/predict``.

    The pattern in this class is "baseline + one mutation per test": start
    from the known-good ``valid_customer_payload`` fixture, change exactly
    one field, and assert that Pydantic raises. This isolates the cause of
    each failure and prevents one buggy field from masking another.
    """

    def test_accepts_valid_payload(self, valid_customer_payload):
        """Sanity check: the baseline fixture parses cleanly.

        If this test fails, every other test in the class is suspect
        because they all start from the same baseline. We also verify that
        enum-backed fields land on the *enum member* (not a raw string),
        which guards against accidental ``use_enum_values=True`` config
        that would silently downgrade enums to strings.
        """
        customer = CustomerData(**valid_customer_payload)
        assert customer.tenure == 29
        assert customer.Contract is Contract.one_year
        assert customer.PaymentMethod is PaymentMethod.mailed
        assert customer.InternetService is InternetService.dsl
        assert customer.Partner is YesNo.yes

    def test_rejects_invalid_enum_value(self, valid_customer_payload):
        """Unknown enum strings must raise, not coerce silently.

        ``"lifetime"`` is not in the ``Contract`` enum (allowed values are
        Month-to-month, One year, Two year). If validation were lenient,
        the downstream one-hot encoder would emit an all-zeros row for
        this column and the model would produce a meaningless prediction.
        """
        valid_customer_payload["Contract"] = "lifetime"
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_negative_tenure(self, valid_customer_payload):
        """``tenure`` is a months-as-customer count — must be >= 0.

        Negative tenure would not crash the model (it's just a number to
        the gradient boosters) but it would represent a logically
        impossible customer and skew any monitoring metrics. The schema
        catches it at the boundary.
        """
        valid_customer_payload["tenure"] = -1
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_negative_charges(self, valid_customer_payload):
        """``MonthlyCharges`` is a currency amount — never negative.

        Mirrors the tenure check. If we ever add ``TotalCharges`` range
        validation, that should get its own test rather than being folded
        into this one — the per-test mutation pattern in this class makes
        failures easy to attribute.
        """
        valid_customer_payload["MonthlyCharges"] = -10.0
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_invalid_senior_citizen(self, valid_customer_payload):
        """``SeniorCitizen`` is a binary flag stored as 0 or 1.

        The Kaggle Telco dataset encodes this as an integer rather than a
        ``YesNo`` string, so the schema uses an integer constraint.
        Anything outside {0, 1} (here we try 2) must be rejected before
        reaching the feature pipeline, which would otherwise treat it as
        a magnitude rather than a category.
        """
        valid_customer_payload["SeniorCitizen"] = 2
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_missing_required_field(self, valid_customer_payload):
        """Required fields must surface at parse time, not later.

        We delete ``MonthlyCharges`` from the payload. The preprocess
        pipeline would eventually raise a KeyError when it tries to
        compute ``AvgMonthlyCharge``, but that error is opaque and lands
        in a 500 response. Catching it here gives the client a 422 with
        a precise field name.
        """
        valid_customer_payload.pop("MonthlyCharges")
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)


class TestResponseSchemas:
    """Round-trip checks for the response side of the API.

    Less stringent than the request side because the API itself
    constructs these (no untrusted input). We only need to verify the
    models accept the values the prediction code produces and that
    multi-row responses are wrapped in a proper list.
    """

    def test_prediction_response(self):
        """``PredictionResponse`` carries the probability and the bool.

        ``pytest.approx`` guards against float-equality flakiness even
        though 0.73 is representable exactly — being defensive here is
        cheap and prevents bit-rot if the field is ever computed rather
        than passed through.
        """
        response = PredictionResponse(churn_probability=0.73, churn=True)
        assert response.churn_probability == pytest.approx(0.73)
        assert response.churn is True

    def test_batch_prediction_wraps_list(self):
        """``BatchPredictionResponse.predictions`` is an ordered list.

        We construct a two-item batch with different outcomes (one churn,
        one not) and assert the second item kept its ``churn=True``. This
        catches any future refactor that accidentally returns a dict
        keyed by index — the JSON would still validate but iteration
        order and the client-side schema would break.
        """
        batch = BatchPredictionResponse(
            predictions=[
                PredictionResponse(churn_probability=0.1, churn=False),
                PredictionResponse(churn_probability=0.9, churn=True),
            ]
        )
        assert len(batch.predictions) == 2
        assert batch.predictions[1].churn is True
