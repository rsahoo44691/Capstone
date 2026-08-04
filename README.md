# Heart Disease Prediction — MSAI 699 Capstone

Coursework code for the MSAI 699 Capstone (University of the Cumberlands): a heart
disease classifier built on the UCI Cleveland Heart Disease dataset (303 patient
records, 13 clinical features, binary target of disease presence vs. absence).

The repository tracks the project week by week — from a baseline model to a tuned,
explained, and evaluated model.

## Live Demo

- **App:** https://rajeshsahoo2006-heart-disease.hf.space
- **Hugging Face Space (source):** https://huggingface.co/spaces/rajeshsahoo2006/Heart_Disease

## Author

**Rajesh Kumar Sahoo**
- Master of Science in Artificial Intelligence, 
- University of the Cumberlands
- rsahoo44691@ucumberlands.edu

## Weeks

### Week 3 — Baseline Model
- **`Week 3/Assignment/baseline_heart_disease.ipynb`** — a logistic-regression
  baseline with an 80/20 stratified split and 5-fold cross-validation, reporting
  accuracy, precision, recall, F1, and ROC-AUC.
- **`Week 3/Assignment/hf_space/`** — the same baseline wrapped in a Gradio app for
  Hugging Face Spaces (`app.py`, `requirements.txt`). The model trains at startup
  and exposes an interactive prediction UI on the free CPU tier.

### Week 4 — Model Optimization
- **`Week 4/Assignment/model_optimization_heart_disease.ipynb`** — builds on the
  baseline with:
  - **Feature engineering** — one-hot encoding of the 8 categorical clinical codes
    and standardization of the 5 continuous features via a scikit-learn
    `ColumnTransformer` (fit only on training folds to prevent leakage).
  - **Hyperparameter tuning** — `GridSearchCV` optimizing ROC-AUC over a tuned
    logistic regression and a gradient-boosting classifier.
  - **Explainability** — SHAP (global feature ranking + per-patient waterfall).
  - **Trade-off analysis** — accuracy vs. efficiency, plus a fairness check across
    sex subgroups.

  The tuned logistic regression with feature engineering is the recommended model
  (test ROC-AUC 0.968, cross-validated 0.921 ± 0.019).

### Week 5 — Deployment: Prediction API
- **`Week 5/Assignment/api/`** — a FastAPI service implementing the serving
  design from the Week 5 Deployment Strategy Report:
  - **`POST /predict`** — 13 typed clinical features in, calibrated probability
    plus a binary decision and the model version out. Pydantic range-checks
    every field and rejects bad input with a 422 before the model sees it.
  - **`GET /health`** and **`GET /model-info`** — load-balancer probe and model
    provenance (hyperparameters, held-out metrics).
  - **No training-serving skew** — the fitted `ColumnTransformer` pipeline is
    serialized into `model.joblib` and loaded once at startup, so serving-time
    preprocessing is identical to training-time.
  - **`train_model.py`** reproduces Week 4's Model B (`C = 0.3`, test ROC-AUC
    0.968) and calibrates it; **`test_api.py`** has 7 passing tests.

### Week 6 — Testing, Evaluation and Debugging
- **`Week 6/Assignment/model_testing_debugging.ipynb`** — statistical testing of the
  Week 4 models, with an HTML render alongside it and MLflow tracking in `mlflow.db`:
  - **Repeated cross-validation** — 10×5 folds, 50 estimates per model with 95%
    intervals, replacing Week 4's single noisy 5-fold run.
  - **A/B testing** — McNemar's exact test plus a Nadeau-Bengio corrected paired
    t-test. Week 4's claim that gradient boosting had the best accuracy turns out to
    rest on **one patient** (*p* = 1.00); logistic regression wins the corrected
    50-fold comparison (*p* = 0.014).
  - **Error analysis** — all 7 errors fall in the 0.2–0.8 probability band, with zero
    errors among the 38 confidently scored patients.
  - **Threshold tuning** — moving from 0.5 to 0.25 removes both false negatives
    (recall 1.000), with a sensitivity check on the assumed cost ratio.

  Both findings were shipped into the API as **v1.1.0**.

## Running the notebooks

Each notebook loads `processed.cleveland.data` from its own directory, so it runs
after a clone with no extra setup.

```bash
pip install scikit-learn pandas numpy matplotlib shap jupyter
jupyter notebook
```

Open either notebook and run all cells. (The committed notebooks already include
their executed outputs, so results are viewable on GitHub without re-running.)

## Running the prediction API locally

```bash
cd "Week 5/Assignment/api"
pip install -r requirements.txt
uvicorn main:app --reload      # http://127.0.0.1:8000/docs
```

`model.joblib` is committed, so the service runs straight from a clone. Rebuild
it with `python train_model.py` if you want to retrain.

## Running the Gradio app locally

```bash
cd "Week 3/Assignment/hf_space"
pip install -r requirements.txt gradio
python app.py
```

## Data

Cleveland database from the UCI Heart Disease repository (Detrano et al., 1989),
included as `processed.cleveland.data`.

## Disclaimer

For educational use only. **Not medical advice.** The dataset is a small,
single-site sample from 1989 and is not suitable for clinical decision-making.
