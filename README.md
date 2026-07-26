# Heart Disease Prediction — MSAI 699 Capstone

Coursework code for the MSAI 699 Capstone (University of the Cumberlands): a heart
disease classifier built on the UCI Cleveland Heart Disease dataset (303 patient
records, 13 clinical features, binary target of disease presence vs. absence).

The repository tracks the project week by week — from a baseline model to a tuned,
explained, and evaluated model.

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

## Running the notebooks

Each notebook loads `processed.cleveland.data` from its own directory, so it runs
after a clone with no extra setup.

```bash
pip install scikit-learn pandas numpy matplotlib shap jupyter
jupyter notebook
```

Open either notebook and run all cells. (The committed notebooks already include
their executed outputs, so results are viewable on GitHub without re-running.)

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
