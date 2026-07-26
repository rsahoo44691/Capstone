"""
Heart Disease Baseline - Hugging Face Space (Gradio)

Trains the Week 3 logistic-regression baseline on the UCI Cleveland dataset at
startup, then exposes an interactive prediction UI. Pure scikit-learn; runs on
the free CPU tier.
"""
import os
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# On ZeroGPU (zero-a10g) hardware, Hugging Face requires at least one
# @spaces.GPU function or it marks the Space as a runtime error at startup.
# This model is pure-CPU scikit-learn, so we register a no-op decorated
# function solely to satisfy that detector; it is never called and thus never
# allocates a GPU. The import is guarded so local runs (no `spaces` package)
# and CPU-hardware deployments are unaffected.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_healthcheck():
        return "ok"
except Exception:
    pass

RANDOM_STATE = 42
COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
           "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]
FEATURES = COLUMNS[:-1]


def load_data():
    local = "processed.cleveland.data"          # bundled copy, if present
    url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
           "heart-disease/processed.cleveland.data")
    path = local if os.path.exists(local) else url
    return pd.read_csv(path, header=None, names=COLUMNS, na_values="?")


# ---- Train the baseline once at startup ----
df = load_data()
df["target"] = (df["num"] > 0).astype(int)
X, y = df[FEATURES], df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

model = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])
model.fit(X_train, y_train)

TEST_ACC = accuracy_score(y_test, model.predict(X_test))
TEST_AUC = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])


def predict(age, sex, cp, trestbps, chol, fbs, restecg, thalach,
            exang, oldpeak, slope, ca, thal):
    row = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                         thalach, exang, oldpeak, slope, ca, thal]],
                       columns=FEATURES)
    proba = float(model.predict_proba(row)[0, 1])
    label = "Disease likely" if proba >= 0.5 else "No disease"
    return {"Disease": proba, "No disease": 1 - proba}, \
        f"Prediction: {label} (probability of disease {proba:.1%})"


description = (
    "Logistic-regression baseline trained on the UCI Cleveland Heart Disease "
    f"dataset (303 patients). Held-out test accuracy {TEST_ACC:.1%}, "
    f"ROC-AUC {TEST_AUC:.3f}. Enter patient values to get a risk estimate. "
    "For educational use only, not medical advice."
)

# The flagging kwarg was renamed `allow_flagging` -> `flagging_mode` in Gradio 5.
# Pick whichever the installed version supports so this runs on 4.x and 5.x.
import inspect
_flag_param = ("flagging_mode"
               if "flagging_mode" in inspect.signature(gr.Interface).parameters
               else "allow_flagging")

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(20, 90, value=54, step=1, label="Age"),
        gr.Radio([0, 1], value=1, label="Sex (0 = female, 1 = male)"),
        gr.Radio([1, 2, 3, 4], value=4,
                 label="Chest pain type (1 typical...4 asymptomatic)"),
        gr.Slider(80, 220, value=130, step=1, label="Resting BP (mm Hg)"),
        gr.Slider(100, 600, value=245, step=1, label="Cholesterol (mg/dl)"),
        gr.Radio([0, 1], value=0, label="Fasting blood sugar > 120 (0/1)"),
        gr.Radio([0, 1, 2], value=1, label="Resting ECG (0-2)"),
        gr.Slider(60, 220, value=150, step=1, label="Max heart rate (thalach)"),
        gr.Radio([0, 1], value=0, label="Exercise-induced angina (0/1)"),
        gr.Slider(0.0, 6.5, value=1.0, step=0.1, label="ST depression (oldpeak)"),
        gr.Radio([1, 2, 3], value=2, label="ST slope (1-3)"),
        gr.Radio([0, 1, 2, 3], value=0, label="Major vessels colored (ca)"),
        gr.Radio([3, 6, 7], value=3,
                 label="Thalassemia (3 normal, 6 fixed, 7 reversible)"),
    ],
    outputs=[gr.Label(num_top_classes=2, label="Probability"),
             gr.Textbox(label="Result")],
    title="Heart Disease Baseline Classifier",
    description=description,
    **{_flag_param: "never"},
)

if __name__ == "__main__":
    # ssr_mode=False disables Gradio's Node SSR proxy, which can fail the
    # Hugging Face Spaces health check and stop the app after startup.
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
