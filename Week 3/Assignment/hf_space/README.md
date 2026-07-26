---
title: Heart Disease Baseline
emoji: 🫀
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.20.0
app_file: app.py
pinned: false
license: mit
---

# Heart Disease Baseline Classifier

Week 3 capstone baseline (MSAI 699). A logistic-regression model trained on the
UCI Cleveland Heart Disease dataset (303 patients, 13 clinical features),
wrapped in a Gradio UI. The model is trained at startup and reports held-out
test accuracy and ROC-AUC in the app description.

**For educational use only. Not medical advice.**

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

## Data

Loaded from the UCI repository at runtime (or a bundled
`processed.cleveland.data` if placed alongside `app.py`).
