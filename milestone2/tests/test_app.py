from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"features": [1, 2, 3]}
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "version" in data
    assert data["version"] == "v1.0"
