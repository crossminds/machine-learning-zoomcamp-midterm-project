"""API tests for predict.py (requires models/model.bin - run train.py first)."""

from fastapi.testclient import TestClient

from predict import EXAMPLE, app

client = TestClient(app)

LEGIT = {**{f"V{i}": 0.0 for i in range(1, 29)}, "Time": 50_000.0, "Amount": 25.0}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "xgboost"}


def test_known_fraud_is_flagged():
    # EXAMPLE is a real fraud from the held-out test set
    result = client.post("/predict", json=EXAMPLE).json()
    assert result["is_fraud"] is True
    assert result["fraud_probability"] >= result["threshold"]


def test_average_transaction_is_legit():
    # all PCA components at their mean (0), ordinary amount and daytime hour
    result = client.post("/predict", json=LEGIT).json()
    assert result["is_fraud"] is False
    assert 0 <= result["fraud_probability"] < 0.01


def test_missing_field_is_rejected():
    payload = {k: v for k, v in EXAMPLE.items() if k != "V14"}
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_amount_is_rejected():
    assert client.post("/predict", json={**EXAMPLE, "Amount": -5}).status_code == 422
