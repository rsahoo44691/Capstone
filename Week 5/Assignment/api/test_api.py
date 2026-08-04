"""
Tests for the heart disease prediction API.

Run:
    python -m pytest test_api.py -v
or, with no pytest installed:
    python test_api.py
"""
from fastapi.testclient import TestClient

import main

VALID = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


def test_health_reports_loaded_model():
    with TestClient(main.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_version"]


def test_predict_returns_calibrated_traceable_response():
    with TestClient(main.app) as client:
        r = client.post("/predict", json=VALID)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["probability"] <= 1.0
        assert body["prediction"] in (0, 1)
        assert body["model_version"]
        # The decision must agree with the threshold it reports.
        assert body["prediction"] == int(body["probability"] >= body["threshold"])


def test_out_of_range_value_is_rejected():
    with TestClient(main.app) as client:
        bad = VALID | {"trestbps": 9000}
        assert client.post("/predict", json=bad).status_code == 422


def test_invalid_category_code_is_rejected():
    with TestClient(main.app) as client:
        bad = VALID | {"thal": 5}          # only 3, 6, 7 are valid codes
        assert client.post("/predict", json=bad).status_code == 422


def test_missing_field_is_rejected():
    with TestClient(main.app) as client:
        bad = {k: v for k, v in VALID.items() if k != "age"}
        assert client.post("/predict", json=bad).status_code == 422


def test_key_order_does_not_change_the_prediction():
    """Guards the training-serving skew fix: column order comes from the
    artifact, not from the caller's JSON key order."""
    with TestClient(main.app) as client:
        a = client.post("/predict", json=VALID).json()
        b = client.post("/predict",
                        json=dict(reversed(list(VALID.items())))).json()
        assert a["probability"] == b["probability"]


def test_model_info_exposes_provenance():
    with TestClient(main.app) as client:
        body = client.get("/model-info").json()
        assert body["test_metrics"]["roc_auc"] > 0.9
        assert len(body["features"]) == 13


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
                passed += 1
            except AssertionError as exc:
                print(f"FAIL {name}: {exc}")
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
