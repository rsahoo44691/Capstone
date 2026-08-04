# Week 6 — Testing, Evaluation and Debugging

Code deliverable for the Week 6 assignment (*Model Testing & Debugging Report*).

## Files

| File | What it is |
|---|---|
| `model_testing_debugging.ipynb` | The analysis. 34 cells, executed, all outputs and 4 figures saved. |
| `Week6_model_testing_debugging_RESULTS.html` | Rendered copy — every result visible without running Jupyter. |
| `mlflow.db` | MLflow tracking store, 4 logged runs. |
| `processed.cleveland.data` | UCI Cleveland dataset, bundled so the notebook runs from a clean clone. |

## Requirements coverage

| Assignment requirement | Where |
|---|---|
| Conduct A/B testing **or** cross-validation | §3 repeated CV (10×5), §4 A/B test — both |
| Perform error analysis, document common failures | §5, two named failure modes |
| Improve model reliability & document best practices | §6 threshold tuning, §7 calibration, §10 practices |
| Updated code with test results | this notebook + the v1.1.0 API changes below |

## Headline findings

**1. Week 4's model ranking was noise.** The Week 4 report said gradient boosting
"achieved the highest raw test accuracy (0.902)" versus 0.885 for logistic
regression. McNemar's test shows the two models disagree on exactly **two of 61
patients** — one that C gets right, none that B gets right. Exact *p* = **1.00**.
The entire 1.7-point accuracy gap is one patient. On a properly corrected paired
test across 50 folds, B beats C by +0.020 ROC-AUC (*p* = 0.014) — a real difference
in the **opposite** direction.

**2. Errors live in a narrow, identifiable band.** All 7 test errors fall between
probability 0.2 and 0.8. There are **zero errors** among the 20 patients scored
below 0.2 and the 18 scored above 0.8 — 62% of patients are classified with
complete reliability, and the risk is confined to a band you can route around.

**3. Both false negatives were threshold artefacts, not ranking failures.** They
scored 0.303 and 0.421 — just under the 0.5 cut. Moving the threshold to 0.25
eliminates both, taking recall from 0.929 to **1.000**.

**4. Calibration didn't help.** Brier 0.081 → 0.083. Reported as a negative result
rather than dropped.

## What changed in the API as a result

Both findings above were applied to the Week 5 service (`Week 5/Assignment/api/`),
which is now **v1.1.0**:

- **Threshold 0.5 → 0.25.** Recall 1.000, zero false negatives, accuracy down to
  0.836. That trade is deliberate.
- **`requires_review` flag.** Predictions inside the 0.2–0.8 band return
  `requires_review: true` with a reason, so borderline cases go to a clinician.

API test count went from 7 to 10, all passing.

## Running it

```bash
pip install scikit-learn pandas numpy matplotlib seaborn scipy statsmodels mlflow jupyter
jupyter notebook model_testing_debugging.ipynb     # Run All
mlflow ui --backend-store-uri sqlite:///mlflow.db  # inspect the tracked runs
```

Runs offline start to finish — MLflow uses a local SQLite store, no account or API
key. Note MLflow 3.x put the older `./mlruns` file backend into maintenance mode and
raises on it, which is why the tracking URI is `sqlite:///mlflow.db`.

## Honest limitations

The test set is 61 patients. Every subgroup estimate is wide, and the threshold is
tuned on the same small sample it is evaluated on — the clean fix is nested CV or a
separate validation split, which 303 records cannot really support. The 5:1
false-negative cost ratio is a stated assumption, not a derived one; §6 checks how
much the conclusion depends on it. Aggregating the Hungarian, Switzerland, and VA
cohorts remains the highest-value next step.

**Educational use only. Not medical advice.**
