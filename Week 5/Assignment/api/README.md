# Heart Disease Prediction API — Week 5

FastAPI implementation of the serving design described in the Week 5
*Deployment Strategy Report*. The report specified this service; this directory
is the working code for it.

## What this implements

Each item below is a claim made in the Week 5 report, now backed by code:

| Report claim | Implementation |
|---|---|
| FastAPI over Flask for validation + OpenAPI | `main.py`, auto docs at `/docs` |
| `POST /predict` taking 13 typed clinical features | `main.py::predict`, `schemas.py::PatientFeatures` |
| Pydantic rejects malformed / out-of-range input before the model | `schemas.py` — `Literal` codes and `ge`/`le` bounds, returns 422 |
| Response carries probability, decision, and model version for traceability | `schemas.py::PredictionResponse` |
| Calibrated probability | `train_model.py` wraps the tuned pipeline in `CalibratedClassifierCV` |
| Model loaded **once at startup**, not retrained per request | `main.py::lifespan` loads `model.joblib` |
| `GET /health` for load-balancer checks | `main.py::health` |
| Same pipeline as training → no training-serving skew | the fitted `ColumnTransformer` is serialized inside the artifact |
| Raw patient features never written to logs | `main.py::predict` logs only version, outcome, latency |

## Quick start

```bash
pip install -r requirements.txt
python train_model.py          # writes model.joblib (~23 KB)
uvicorn main:app --reload      # http://127.0.0.1:8000/docs
```

`model.joblib` is committed, so you can skip the training step and run the
service straight from a clone.

### Example

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "age":63,"sex":1,"cp":4,"trestbps":145,"chol":233,"fbs":1,"restecg":2,
  "thalach":110,"exang":1,"oldpeak":2.3,"slope":3,"ca":3,"thal":7}'
```

```json
{"probability":0.96,"prediction":1,"label":"Disease likely",
 "threshold":0.5,"model_version":"1.0.0"}
```

A low-risk profile (40F, asymptomatic, `ca=0`, `thal=3`) returns `0.0258`.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Score one patient |
| `GET` | `/health` | Liveness/readiness; reports `degraded` if the artifact failed to load |
| `GET` | `/model-info` | Model provenance, chosen hyperparameters, held-out metrics |
| `GET` | `/docs` | Interactive OpenAPI documentation |

## Tests

```bash
python test_api.py        # no pytest needed
# or: python -m pytest test_api.py -v
```

All 7 pass. They cover the health check, a valid prediction, threshold/decision
agreement, three rejection cases (out-of-range value, invalid category code,
missing field), and a regression test that JSON key order cannot change the
prediction — the guard on the training-serving skew fix, since column order is
taken from the artifact rather than from the caller.

## Model

Reproduces **Model B** from the Week 4 notebook (tuned logistic regression with
feature engineering), selected `C = 0.3`, then calibrated. Held-out test set:

| Metric | Value |
|---|---|
| Accuracy | 0.885 |
| Precision | 0.839 |
| Recall | 0.929 |
| F1 | 0.881 |
| ROC-AUC | 0.968 |
| CV ROC-AUC (full dataset) | 0.921 |

These match the Week 4 report exactly.

### Two notes on honesty

**The two CV numbers.** `train_model.py` records both
`cv_roc_auc_gridsearch_train` (0.904) and `cv_roc_auc_full_dataset` (0.921).
They are different quantities and the Week 4 notebook printed both — 0.904 is
the grid search's best mean score on the training split, 0.921 is a 5-fold CV of
the selected model over the full dataset, which is the "CV ROC-AUC" column in
the report table. Both are recorded so the numbers are never ambiguous again.

**Calibration barely moved anything.** Brier score went from 0.081 uncalibrated
to 0.083 calibrated — very slightly *worse*, well within noise on a 61-row test
set. A regularized logistic regression is already close to well calibrated, so
Platt scaling has little to correct. Calibration is kept because the report
promises a calibrated probability and it costs nothing, but it should not be
claimed as an improvement. Sigmoid was chosen over isotonic because ~240
training rows is far too little for a non-parametric fit.

## Not production ready

This is coursework. Before any real use the service would still need
authentication (API keys or OAuth), TLS termination, rate limiting, subgroup
fairness monitoring in production, and drift detection — all discussed in the
Week 5 report but deliberately out of scope for a local demonstrator.

**Educational use only. Not medical advice.**
