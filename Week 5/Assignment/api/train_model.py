"""
Train and serialize the serving artifact for the heart disease API.

This reproduces Model B from the Week 4 notebook -- the tuned logistic
regression with feature engineering -- and then wraps it in
``CalibratedClassifierCV`` so the API returns a calibrated probability rather
than a raw decision score. Calibration was listed as future work in the Week 4
conclusion; it is delivered here because a probability that is served to a
clinician should mean what it says.

The exact same ``ColumnTransformer`` pipeline used in training is what gets
serialized, so preprocessing at serving time is identical to preprocessing at
training time. That is what prevents training-serving skew.

Run:
    python train_model.py
Produces:
    model.joblib
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
MODEL_VERSION = "1.0.0"
THRESHOLD = 0.5

HERE = Path(__file__).parent
DATA = HERE / "processed.cleveland.data"
ARTIFACT = HERE / "model.joblib"

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    """Week 4 feature engineering: scale the continuous, one-hot the categorical."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL),
    ])


def main() -> None:
    np.random.seed(RANDOM_STATE)

    df = pd.read_csv(DATA, header=None, names=COLS, na_values="?")
    df["target"] = (df["num"] > 0).astype(int)
    X, y = df[FEATURES], df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # Model B: tune the regularization strength on ROC-AUC.
    pipe = Pipeline([("pre", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=5000,
                                                random_state=RANDOM_STATE))])
    grid = GridSearchCV(pipe, {"clf__C": [0.01, 0.1, 0.3, 1, 3, 10]},
                        scoring="roc_auc", cv=cv, n_jobs=1)
    grid.fit(X_train, y_train)
    uncalibrated = grid.best_estimator_

    # Calibrate the probabilities. Sigmoid (Platt) rather than isotonic because
    # ~240 training rows is far too little data for a non-parametric fit.
    model = CalibratedClassifierCV(uncalibrated, method="sigmoid", cv=cv)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    metrics = {
        "accuracy": round(accuracy_score(y_test, pred), 3),
        "precision": round(precision_score(y_test, pred), 3),
        "recall": round(recall_score(y_test, pred), 3),
        "f1": round(f1_score(y_test, pred), 3),
        "roc_auc": round(roc_auc_score(y_test, proba), 3),
        "brier": round(brier_score_loss(y_test, proba), 3),
        "brier_uncalibrated": round(
            brier_score_loss(y_test,
                             uncalibrated.predict_proba(X_test)[:, 1]), 3),
        # Two different quantities, both reported to avoid the ambiguity in the
        # Week 4 write-up: the grid search's best mean CV score on the training
        # split, and a 5-fold CV of the selected model over the full dataset
        # (the latter is the "CV ROC-AUC" column of the Week 4 report table).
        "cv_roc_auc_gridsearch_train": round(grid.best_score_, 3),
        "cv_roc_auc_full_dataset": round(
            cross_val_score(uncalibrated, X, y, cv=cv,
                            scoring="roc_auc", n_jobs=1).mean(), 3),
    }

    joblib.dump({
        "model": model,
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "features": FEATURES,
        "best_params": grid.best_params_,
        "metrics": metrics,
        "sklearn_version": __import__("sklearn").__version__,
    }, ARTIFACT)

    print(f"Best params : {grid.best_params_}")
    print(f"Metrics     : {json.dumps(metrics, indent=2)}")
    print(f"Saved       : {ARTIFACT} "
          f"({ARTIFACT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
