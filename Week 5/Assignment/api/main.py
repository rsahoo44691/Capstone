"""
Heart Disease Prediction API -- FastAPI service.

Implements the deployment design from the Week 5 Deployment Strategy Report:

* ``POST /predict``  -- 13 clinical features in, calibrated probability out,
  validated by Pydantic before the model ever sees the payload.
* ``GET /health``    -- cheap liveness/readiness probe for a load balancer.
* The model is loaded **once at startup** from a serialized artifact and reused
  for every request; it is never retrained per request.
* The serialized pipeline carries its own preprocessing, so serving-time
  transformation is identical to training-time transformation.

Privacy: raw patient features are never written to the logs. Only the model
version, latency, and outcome shape are recorded, which is what the report's
audit-trail requirement asks for.

Run:
    uvicorn main:app --reload
"""
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from schemas import HealthResponse, PatientFeatures, PredictionResponse

ARTIFACT = Path(__file__).parent / "model.joblib"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("heart-api")

# Populated at startup. A dict rather than globals so the health check can
# report honestly when loading failed instead of the process dying silently.
STATE: dict = {"model": None, "meta": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model artifact once, before the first request is served."""
    if ARTIFACT.exists():
        bundle = joblib.load(ARTIFACT)
        STATE["model"] = bundle["model"]
        STATE["meta"] = bundle
        log.info("Loaded model version %s (trained with sklearn %s)",
                 bundle["model_version"], bundle.get("sklearn_version"))
    else:
        # Start anyway so /health can report the problem to the orchestrator
        # rather than crash-looping with no diagnostic surface.
        log.error("Model artifact not found at %s. Run train_model.py first.",
                  ARTIFACT)
    yield
    STATE.clear()


app = FastAPI(
    title="Heart Disease Prediction API",
    description=(
        "Serves the tuned, calibrated logistic-regression model from the "
        "MSAI 699 Capstone. Educational decision support only -- not medical "
        "advice and not a diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Report whether the service can actually serve predictions."""
    loaded = STATE.get("model") is not None
    meta = STATE.get("meta") or {}
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=meta.get("model_version"),
    )


@app.get("/model-info", tags=["ops"])
def model_info() -> dict:
    """Expose model provenance and held-out metrics for auditability."""
    meta = STATE.get("meta")
    if meta is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_version": meta["model_version"],
        "model_type": "Calibrated logistic regression with feature engineering",
        "threshold": meta["threshold"],
        "best_params": meta["best_params"],
        "test_metrics": meta["metrics"],
        "features": meta["features"],
        "sklearn_version": meta.get("sklearn_version"),
        "disclaimer": "Educational use only. Not medical advice.",
    }


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(features: PatientFeatures) -> PredictionResponse:
    """Score one patient and return a traceable, calibrated risk estimate."""
    model = STATE.get("model")
    meta = STATE.get("meta")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Run train_model.py to build model.joblib.")

    started = time.perf_counter()
    # Column order must match training; take it from the artifact, not from
    # the request body's key order.
    row = pd.DataFrame([[getattr(features, f) for f in meta["features"]]],
                       columns=meta["features"])
    probability = float(model.predict_proba(row)[0, 1])
    threshold = meta["threshold"]
    prediction = int(probability >= threshold)
    elapsed_ms = (time.perf_counter() - started) * 1000

    # Deliberately logs no patient features -- only non-identifying metadata.
    log.info("predict model_version=%s outcome=%d latency_ms=%.1f",
             meta["model_version"], prediction, elapsed_ms)

    return PredictionResponse(
        probability=round(probability, 4),
        prediction=prediction,
        label="Disease likely" if prediction else "No disease",
        threshold=threshold,
        model_version=meta["model_version"],
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a stack trace or echo the request body back to the caller."""
    log.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500,
                        content={"detail": "Internal server error"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
