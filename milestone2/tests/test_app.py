from fastapi.testclient import TestClient
import pytest
from app.app import app

client = TestClient(app)


def test_predict_endpoint():
    """Test successful prediction with valid input"""
    response = client.post(
        "/predict",
        json={"features": [1, 2, 3]}
    )

    assert response.status_code == 200

    data = response.json()
    assert "prediction" in data
    assert "version" in data
    assert data["version"] == "v1.0"


def test_predict_with_different_features():
    """Test prediction with different feature values"""
    response = client.post(
        "/predict",
        json={"features": [5, 10, 15]}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], (int, float))


def test_predict_response_type():
    """Test that prediction returns correct data types"""
    response = client.post(
        "/predict",
        json={"features": [1, 2, 3]}
    )

    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data["prediction"], (int, float))
    assert isinstance(data["version"], str)


def test_predict_invalid_json():
    """Test prediction with invalid JSON"""
    response = client.post(
        "/predict",
        json={"invalid_field": [1, 2, 3]}
    )

    assert response.status_code == 422  # Validation error


def test_predict_missing_features():
    """Test prediction without features field"""
    response = client.post(
        "/predict",
        json={}
    )

    assert response.status_code == 422


def test_predict_wrong_method():
    """Test using wrong HTTP method"""
    response = client.get("/predict")
    
    assert response.status_code == 405  # Method not allowed


def test_predict_with_negative_features():
    """Test prediction with negative feature values"""
    response = client.post(
        "/predict",
        json={"features": [-1, -2, -3]}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_predict_with_float_features():
    """Test prediction with float feature values"""
    response = client.post(
        "/predict",
        json={"features": [1.5, 2.7, 3.14]}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_predict_with_zero_features():
    """Test prediction with zero values"""
    response = client.post(
        "/predict",
        json={"features": [0, 0, 0]}
    )

    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data


def test_predict_version_consistency():
    """Test that version is consistent across multiple requests"""
    response1 = client.post("/predict", json={"features": [1, 2, 3]})
    response2 = client.post("/predict", json={"features": [4, 5, 6]})

    assert response1.json()["version"] == response2.json()["version"]


def test_predict_prediction_value_range():
    """Test that prediction is a reasonable value"""
    response = client.post(
        "/predict",
        json={"features": [1, 2, 3]}
    )

    assert response.status_code == 200
    data = response.json()
    prediction = data["prediction"]
    
    # Prediction should be a finite number (not NaN or Inf)
    assert prediction == prediction  # NaN check: NaN != NaN
    assert abs(prediction) != float('inf')
