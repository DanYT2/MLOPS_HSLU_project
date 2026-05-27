"""Validation tests for the request and response Pydantic models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

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
    def test_accepts_valid_payload(self, valid_customer_payload):
        customer = CustomerData(**valid_customer_payload)
        assert customer.tenure == 29
        assert customer.Contract is Contract.one_year
        assert customer.PaymentMethod is PaymentMethod.mailed
        assert customer.InternetService is InternetService.dsl
        assert customer.Partner is YesNo.yes

    def test_rejects_invalid_enum_value(self, valid_customer_payload):
        valid_customer_payload["Contract"] = "lifetime"
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_negative_tenure(self, valid_customer_payload):
        valid_customer_payload["tenure"] = -1
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_negative_charges(self, valid_customer_payload):
        valid_customer_payload["MonthlyCharges"] = -10.0
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_invalid_senior_citizen(self, valid_customer_payload):
        valid_customer_payload["SeniorCitizen"] = 2
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)

    def test_rejects_missing_required_field(self, valid_customer_payload):
        valid_customer_payload.pop("MonthlyCharges")
        with pytest.raises(ValidationError):
            CustomerData(**valid_customer_payload)


class TestResponseSchemas:
    def test_prediction_response(self):
        response = PredictionResponse(churn_probability=0.73, churn=True)
        assert response.churn_probability == pytest.approx(0.73)
        assert response.churn is True

    def test_batch_prediction_wraps_list(self):
        batch = BatchPredictionResponse(
            predictions=[
                PredictionResponse(churn_probability=0.1, churn=False),
                PredictionResponse(churn_probability=0.9, churn=True),
            ]
        )
        assert len(batch.predictions) == 2
        assert batch.predictions[1].churn is True
