# `schemas/schemas.py` — Pydantic Schema Documentation

## Overview

`schemas.py` defines the Pydantic data models and enumerations used by the FastAPI web service for request validation and response serialization. Every incoming customer record is validated against `CustomerData` before it reaches the ML model, ensuring type safety, value constraints, and self-documenting API contracts (auto-generated OpenAPI/Swagger docs).

---

## Enumerations

The enums model the discrete value domains found in the telecom customer dataset. Each inherits from both `str` and `Enum`, making them JSON-serializable and directly comparable to string values.

### `YesNo`

Simple binary enum for columns that accept only "Yes" or "No".

| Member | Value |
|---|---|
| `yes` | `"Yes"` |
| `no` | `"No"` |

**Used by:** `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`

---

### `MultipleLines`

Represents whether the customer has multiple phone lines. Has a third option for customers without phone service at all.

| Member | Value |
|---|---|
| `yes` | `"Yes"` |
| `no` | `"No"` |
| `no_phone` | `"No phone service"` |

**Used by:** `MultipleLines`

---

### `InternetService`

The type of internet connection the customer has.

| Member | Value |
|---|---|
| `dsl` | `"DSL"` |
| `fiber` | `"Fiber optic"` |
| `no` | `"No"` |

**Used by:** `InternetService`

---

### `InternetDependent`

Enum for service columns that depend on whether the customer has internet. If the customer has no internet service, these columns carry the value `"No internet service"` rather than a simple `"No"`.

| Member | Value |
|---|---|
| `yes` | `"Yes"` |
| `no` | `"No"` |
| `no_internet` | `"No internet service"` |

**Used by:** `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

---

### `Contract`

The customer's contract term length.

| Member | Value |
|---|---|
| `month` | `"Month-to-month"` |
| `one_year` | `"One year"` |
| `two_year` | `"Two year"` |

**Used by:** `Contract`

---

### `PaymentMethod`

How the customer pays their bill.

| Member | Value |
|---|---|
| `electronic` | `"Electronic check"` |
| `mailed` | `"Mailed check"` |
| `bank_transfer` | `"Bank transfer (automatic)"` |
| `credit_card` | `"Credit card (automatic)"` |

**Used by:** `PaymentMethod`

---

## Request Model

### `CustomerData(BaseModel)`

The primary input schema representing a single telecom customer. FastAPI uses this model to:
- **Validate** incoming JSON payloads (type checking, enum membership, numeric constraints).
- **Generate** OpenAPI documentation with field descriptions and examples.
- **Serialize** enum values back to their string representations when needed.

#### Fields

| Field | Type | Constraints | Description |
|---|---|---|---|
| `SeniorCitizen` | `Literal[0, 1]` | Must be exactly 0 or 1 | Whether the customer is a senior citizen. Uses `Literal` instead of `bool` because the original dataset stores this as an integer. |
| `Partner` | `YesNo` | Enum-validated | Whether the customer has a partner. |
| `Dependents` | `YesNo` | Enum-validated | Whether the customer has dependents. |
| `tenure` | `int` | `>= 0` | Number of months the customer has stayed with the company. |
| `PhoneService` | `YesNo` | Enum-validated | Whether the customer has phone service. |
| `MultipleLines` | `MultipleLines` | Enum-validated | Whether the customer has multiple phone lines. |
| `InternetService` | `InternetService` | Enum-validated | The customer's internet service provider type. |
| `OnlineSecurity` | `InternetDependent` | Enum-validated | Whether the customer has online security add-on. |
| `OnlineBackup` | `InternetDependent` | Enum-validated | Whether the customer has online backup add-on. |
| `DeviceProtection` | `InternetDependent` | Enum-validated | Whether the customer has device protection add-on. |
| `TechSupport` | `InternetDependent` | Enum-validated | Whether the customer has tech support add-on. |
| `StreamingTV` | `InternetDependent` | Enum-validated | Whether the customer has streaming TV add-on. |
| `StreamingMovies` | `InternetDependent` | Enum-validated | Whether the customer has streaming movies add-on. |
| `Contract` | `Contract` | Enum-validated | The customer's contract term type. |
| `PaperlessBilling` | `YesNo` | Enum-validated | Whether the customer uses paperless billing. |
| `PaymentMethod` | `PaymentMethod` | Enum-validated | The customer's payment method. |
| `MonthlyCharges` | `float` | `>= 0` | The customer's monthly charge amount. |
| `TotalCharges` | `float` | `>= 0` | The total amount charged to the customer over their tenure. |

#### Model Config — JSON Schema Example

The `model_config` dictionary provides an example payload that appears in the auto-generated Swagger UI at `/docs`. This serves as a ready-to-use test case:

```json
{
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
  "TotalCharges": 1653.85
}
```

---

## Response Models

### `PredictionResponse(BaseModel)`

The response for a single customer prediction.

| Field | Type | Description |
|---|---|---|
| `churn_probability` | `float` | The ensemble model's predicted probability of churn (0.0 – 1.0). |
| `churn` | `bool` | Binary churn decision — `true` if `churn_probability >= 0.5`, `false` otherwise. |

**Example response:**
```json
{
  "churn_probability": 0.234567,
  "churn": false
}
```

---

### `BatchPredictionResponse(BaseModel)`

Wraps a list of individual predictions for the batch endpoint.

| Field | Type | Description |
|---|---|---|
| `predictions` | `list[PredictionResponse]` | Ordered list of predictions, one per input customer. Order matches the input list. |

**Example response:**
```json
{
  "predictions": [
    {"churn_probability": 0.234567, "churn": false},
    {"churn_probability": 0.891234, "churn": true}
  ]
}
```

---

## Design Decisions

1. **String enums over plain strings:** By using `str, Enum` subclasses, the schema restricts inputs to the exact values the ML model was trained on. Invalid values like `"maybe"` or `"fiber"` (lowercase) are rejected at the API boundary with a clear 422 Validation Error, preventing silent data issues downstream.

2. **`Literal[0, 1]` for `SeniorCitizen`:** The original dataset represents this field as an integer rather than a "Yes"/"No" string. `Literal` preserves this convention while still constraining the input to exactly two valid values.

3. **`Field(ge=0)` on numeric columns:** `tenure`, `MonthlyCharges`, and `TotalCharges` are constrained to non-negative values, catching obviously invalid inputs (e.g. negative charges).

4. **`InternetDependent` as a separate enum:** Six service columns share the same three-value domain ("Yes", "No", "No internet service"). Factoring this into a single reusable enum avoids repetition and keeps the schema DRY.

5. **`model_config` with examples:** Provides the Swagger UI with a realistic, copy-pasteable example payload, improving developer experience when testing the API interactively.
