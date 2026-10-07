"""Validated input/output schemas for the prediction service (heart.csv feature set)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EXAMPLE = {
    "Age": 54,
    "Sex": "M",
    "ChestPainType": "ASY",
    "RestingBP": 140,
    "Cholesterol": 239,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 122,
    "ExerciseAngina": "Y",
    "Oldpeak": 1.5,
    "ST_Slope": "Flat",
}


class Patient(BaseModel):
    """One patient. Unknown blood pressure / cholesterol must be sent as ``null`` (not 0)."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE]})

    Age: int = Field(ge=18, le=100, description="years")
    Sex: Literal["M", "F"]
    ChestPainType: Literal["TA", "ATA", "NAP", "ASY"] = Field(
        description="TA typical angina, ATA atypical angina, NAP non-anginal pain, ASY asymptomatic"
    )
    RestingBP: float | None = Field(default=None, ge=60, le=250, description="resting systolic BP, mmHg")
    Cholesterol: float | None = Field(default=None, ge=80, le=700, description="serum cholesterol, mg/dL")
    FastingBS: Literal[0, 1] = Field(description="1 if fasting blood sugar > 120 mg/dL")
    RestingECG: Literal["Normal", "ST", "LVH"]
    MaxHR: float = Field(ge=50, le=230, description="maximum heart rate achieved, bpm")
    ExerciseAngina: Literal["Y", "N"]
    Oldpeak: float = Field(ge=-3, le=7, description="ST depression induced by exercise, mm")
    ST_Slope: Literal["Up", "Flat", "Down"]


class Prediction(BaseModel):
    probability: float = Field(description="calibrated probability of coronary artery disease")
    prediction: int = Field(description="1 if probability >= threshold")
    threshold: float
    risk_band: str
    warnings: list[str] = []


class PredictionResponse(BaseModel):
    model: str
    model_version: str
    results: list[Prediction]


class Contribution(BaseModel):
    feature: str
    value: str | float | int | None
    contribution: float = Field(description="Shapley value on the probability scale")


class Explanation(BaseModel):
    model: str
    probability: float
    base_value: float = Field(description="mean predicted probability over the reference sample")
    contributions: list[Contribution]


class ModelInfo(BaseModel):
    model: str
    model_version: str
    threshold: float
    threshold_strategy: str
    calibration: str
    trained_on: dict
    cv_performance: dict
    risk_bands: list
    features: list[str]
    intended_use: str
