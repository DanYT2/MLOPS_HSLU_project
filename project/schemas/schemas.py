# Enables PEP 604 union syntax (X | Y) and PEP 585 generics (list[str])
# in type annotations at runtime for older Python versions. Without this,
# `list[PredictionResponse]` in a class body would raise TypeError on Python <3.9.
from __future__ import annotations

# Enum: base class for creating enumerations — a fixed set of named constants.
# Using enums for API input validation ensures only predefined values are accepted.
from enum import Enum

# Literal: a type hint that restricts a value to specific literal values.
# Unlike Enum, Literal works for primitive types (ints, strings) without
# needing a class definition. Used here for SeniorCitizen which is 0 or 1.
from typing import Literal

# Pydantic BaseModel: the foundation for data validation in FastAPI.
# When FastAPI receives a JSON request body, it deserializes it into the
# Pydantic model, automatically validating types, constraints, and enum
# membership. Invalid data returns a 422 Unprocessable Entity response
# with detailed error messages.
# Field: allows attaching validation constraints (ge=greater-or-equal,
# le=less-or-equal, etc.) and metadata to individual model fields.
from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────
# These enums define the exact allowed values for each categorical column
# in the telecom customer dataset. By inheriting from both `str` and `Enum`,
# each member is simultaneously a string (JSON-serializable, comparable to
# plain strings) and an enum (validated against the fixed set of values).
# This dual inheritance is the standard pattern for Pydantic/FastAPI enums.

class YesNo(str, Enum):
    """Binary enum for columns that accept only "Yes" or "No".

    Used by: Partner, Dependents, PhoneService, PaperlessBilling.
    These are simple binary attributes with no conditional dependencies.
    """
    yes = "Yes"
    no = "No"


class MultipleLines(str, Enum):
    """Enum for the MultipleLines column.

    Has three possible values because the concept of "multiple lines"
    only applies if the customer has phone service at all. If they don't
    have phone service, the value is "No phone service" rather than
    a simple "No" — this encodes the dependency on PhoneService.
    """
    yes = "Yes"
    no = "No"
    no_phone = "No phone service"


class InternetService(str, Enum):
    """Enum for the type of internet service the customer subscribes to.

    DSL (Digital Subscriber Line) and Fiber optic are the two internet
    technologies offered. "No" means the customer has no internet service,
    which affects several downstream service columns (see InternetDependent).
    """
    dsl = "DSL"
    fiber = "Fiber optic"
    no = "No"


class InternetDependent(str, Enum):
    """Enum for service columns that depend on internet connectivity.

    Six columns share this same three-value domain: OnlineSecurity,
    OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies.
    When a customer has no internet service (InternetService="No"), all of
    these columns take the value "No internet service" rather than "No".
    This distinction matters because "No" means "has internet but opted out"
    while "No internet service" means "can't subscribe to this add-on".
    Factoring this into a single reusable enum avoids defining 6 identical enums.
    """
    yes = "Yes"
    no = "No"
    no_internet = "No internet service"


class Contract(str, Enum):
    """Enum for the customer's contract term length.

    Month-to-month contracts have no lock-in period (highest churn risk).
    One-year and two-year contracts create switching costs that reduce churn.
    This is typically one of the strongest predictive features for churn.
    """
    month = "Month-to-month"
    one_year = "One year"
    two_year = "Two year"


class PaymentMethod(str, Enum):
    """Enum for how the customer pays their bill.

    Four distinct payment methods. Research shows that "Electronic check"
    customers tend to have higher churn rates, possibly because automatic
    payment methods (bank transfer, credit card) indicate higher customer
    commitment and reduce friction that might trigger churn consideration.
    """
    electronic = "Electronic check"
    mailed = "Mailed check"
    bank_transfer = "Bank transfer (automatic)"
    credit_card = "Credit card (automatic)"


# ── Request Model ────────────────────────────────────────────────

class CustomerData(BaseModel):
    """Pydantic model representing a single telecom customer's attributes.

    This is the primary input schema for the prediction API. When FastAPI
    receives a JSON request body, it parses it into this model. Every field
    is validated:
      - Enum fields: must match one of the predefined string values
      - Literal fields: must be one of the listed literal values
      - Field(ge=0): must be greater than or equal to 0
    If any validation fails, FastAPI automatically returns a 422 error with
    a detailed JSON body explaining which field failed and why.

    The 17 fields map directly to the columns in the training dataset
    (minus 'id', 'gender', and 'Churn' which are handled separately).
    """

    # SeniorCitizen is an integer (0 or 1) in the original dataset, not a
    # "Yes"/"No" string like the other binary columns. Literal[0, 1] constrains
    # it to exactly these two values — any other integer (e.g. 2) is rejected.
    SeniorCitizen: Literal[0, 1]

    # Binary demographic attributes: does the customer have a partner or dependents?
    Partner: YesNo
    Dependents: YesNo

    # tenure: number of months the customer has been with the company.
    # Field(ge=0) enforces non-negative values (can't have negative months).
    # tenure=0 means the customer just signed up this month.
    tenure: int = Field(ge=0)

    # Phone service attributes
    PhoneService: YesNo          # Does the customer have phone service at all?
    MultipleLines: MultipleLines  # Multiple phone lines (depends on PhoneService)

    # Internet service type and internet-dependent add-on services.
    # If InternetService is "No", all six InternetDependent fields should
    # be "No internet service" (though this cross-field constraint is not
    # enforced at the schema level — it relies on valid input data).
    InternetService: InternetService
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent

    # Billing attributes
    Contract: Contract          # Contract term length (month-to-month, 1yr, 2yr)
    PaperlessBilling: YesNo     # Whether the customer uses paperless billing
    PaymentMethod: PaymentMethod  # How the customer pays

    # Financial attributes. Both must be non-negative.
    # MonthlyCharges: the customer's current monthly bill amount.
    # TotalCharges: cumulative amount charged over the customer's entire tenure.
    # For a new customer (tenure=0), TotalCharges might equal MonthlyCharges.
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    # model_config: Pydantic v2 configuration dictionary.
    # json_schema_extra injects additional data into the generated JSON Schema.
    # The "examples" key provides a realistic sample payload that appears in
    # FastAPI's auto-generated Swagger UI (/docs), making it easy for developers
    # to test the API by clicking "Try it out" with pre-filled data.
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


# ── Response Models ──────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Response schema for a single customer churn prediction.

    churn_probability: the ensemble model's estimated probability that
    this customer will churn, ranging from 0.0 (definitely won't churn)
    to 1.0 (definitely will churn).

    churn: a binary decision derived from the probability. True if
    churn_probability >= 0.5, False otherwise. This uses 0.5 as the
    default decision threshold, which may not be optimal for all
    business contexts (e.g. if the cost of missing a churner is much
    higher than a false alarm, a lower threshold would be better).
    """
    churn_probability: float
    churn: bool


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction (multiple customers at once).

    Wraps a list of individual PredictionResponse objects. The order of
    predictions matches the order of customers in the input list, so
    predictions[i] corresponds to the i-th customer in the request.
    """
    predictions: list[PredictionResponse]
