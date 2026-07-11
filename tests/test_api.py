"""End-to-end API tests using FastAPI's TestClient.

The TestClient runs the lifespan handler, so the model is trained/loaded exactly
as it would be in production before any request is served.
"""

from fastapi.testclient import TestClient
from sklearn.datasets import load_digits

from app.main import app


def test_healthz():
    with TestClient(app) as client:
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


def test_readyz_ready_after_startup():
    with TestClient(app) as client:
        resp = client.get("/readyz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"


def test_metadata():
    with TestClient(app) as client:
        resp = client.get("/metadata")
        assert resp.status_code == 200
        body = resp.json()
        assert body["n_features"] == 64
        assert body["labels"] == list(range(10))
        assert body["training_accuracy"] > 0.9


def test_predict_known_sample():
    digits = load_digits()
    # Use the first sample, whose true label we know.
    sample = digits.data[0].tolist()
    expected = int(digits.target[0])

    with TestClient(app) as client:
        resp = client.post("/predict", json={"pixels": sample})
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] == expected
        assert 0 <= body["prediction"] <= 9
        assert len(body["probabilities"]) == 10
        assert abs(sum(body["probabilities"]) - 1.0) < 1e-6


def test_predict_rejects_wrong_length():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"pixels": [0.0] * 10})
        assert resp.status_code == 422
