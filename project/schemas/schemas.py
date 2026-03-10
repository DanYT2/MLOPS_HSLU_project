from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

# Enums for the Yes/No columns
class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class MultipleLines(str, Enum):
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"


class InternetService(str, Enum):
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"


class InternetDependent(str, Enum):
    """Columns whose values depend on whether the customer has internet."""

    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class Contract(str, Enum):
    month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethod(str, Enum):
    electronic = "Electronic check"
    mailed = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


class CustomerData(BaseModel):
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: MultipleLines
    InternetService: InternetService
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Contract
    PaperlessBilling: YesNo
    PaymentMethod: PaymentMethod
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    model_config = {"json_schema_extra": {
        "examples": [{
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
        }]
    }}


class PredictionResponse(BaseModel):
    churn_probability: float
    churn: bool
    lgbm_probability: float
    xgb_probability: float


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]