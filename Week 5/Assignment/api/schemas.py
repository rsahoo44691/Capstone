"""
Request and response schemas for the heart disease prediction API.

Pydantic validates and range-checks every field before the payload reaches the
model, so malformed or clinically impossible input is rejected at the boundary
with a 422 rather than producing a silently meaningless prediction.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    """The 13 clinical features of the UCI Cleveland heart disease dataset.

    Categorical fields use Literal so only the exact codes the model was
    trained on are accepted. Continuous fields carry clinically plausible
    bounds that are wider than the training range, so the API accepts real
    patients outside the 1989 Cleveland sample while still rejecting nonsense.
    """

    age: int = Field(..., ge=18, le=120, description="Age in years")
    sex: Literal[0, 1] = Field(..., description="0 = female, 1 = male")
    cp: Literal[1, 2, 3, 4] = Field(
        ..., description="Chest pain type: 1 typical angina ... 4 asymptomatic")
    trestbps: float = Field(
        ..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(
        ..., ge=0, le=700, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(
        ..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(
        ..., description="Resting ECG result")
    thalach: float = Field(
        ..., ge=40, le=250, description="Maximum heart rate achieved")
    exang: Literal[0, 1] = Field(
        ..., description="Exercise-induced angina")
    oldpeak: float = Field(
        ..., ge=0.0, le=10.0,
        description="ST depression induced by exercise relative to rest")
    slope: Literal[1, 2, 3] = Field(
        ..., description="Slope of the peak exercise ST segment")
    ca: Literal[0, 1, 2, 3] = Field(
        ..., description="Number of major vessels colored by fluoroscopy")
    thal: Literal[3, 6, 7] = Field(
        ..., description="Thalassemia: 3 normal, 6 fixed defect, 7 reversible")

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233,
                "fbs": 1, "restecg": 2, "thalach": 150, "exang": 0,
                "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
            }]
        }
    }


class PredictionResponse(BaseModel):
    """A traceable prediction: probability, decision, and the model that made it."""

    probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Calibrated probability that heart disease is present")
    prediction: Literal[0, 1] = Field(
        ..., description="Binary decision at the operating threshold")
    label: str = Field(..., description="Human-readable decision")
    threshold: float = Field(
        ..., description="Operating threshold applied to the probability")
    requires_review: bool = Field(
        ...,
        description=("True when the probability falls in the uncertainty band "
                     "where Week 6 error analysis found every misclassification. "
                     "Such predictions should go to a clinician, not be acted on."))
    review_reason: Optional[str] = Field(
        None, description="Why review was flagged, or null if not flagged")
    model_version: str = Field(
        ..., description="Version of the model artifact that served this request")

    model_config = {"protected_namespaces": ()}


class HealthResponse(BaseModel):
    """Load-balancer health check payload."""

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: Optional[str] = None

    model_config = {"protected_namespaces": ()}
