# Milestone 1 – Web & Serverless Model Serving

## Project Overview
This project demonstrates end-to-end model serving using two deployment patterns:
1. A containerized FastAPI service deployed on Google Cloud Run
2. A serverless Google Cloud Function for inference

The objective is to understand how trained ML model artifacts are deployed and served via APIs, and to compare architectural trade-offs between container-based and serverless deployments.

---

## Model Description
A lightweight scikit-learn Linear Regression model was trained using numerical input features.  
The trained model is serialized as a deterministic artifact (`model.pkl`) using `joblib`.

The same model artifact is reused across:
- Local FastAPI serving
- Cloud Run deployment
- Cloud Function deployment

---

## ML Lifecycle Context
This project focuses on the **model serving and deployment stage** of the ML lifecycle.

Lifecycle flow:
1. Model training and artifact creation (offline)
2. Model serialization (`model.pkl`)
3. Model loading by the serving layer
4. API-based inference via HTTP endpoints
5. Consumers invoke predictions through REST calls

Monitoring and logging would typically be added at the serving layer to track latency, errors, and prediction behavior.

---

## Local FastAPI Service
A FastAPI application exposes a `/predict` endpoint with:
- Pydantic request schema for input validation
- Pydantic response schema for structured output
- Deterministic model loading at application startup

The model is loaded once when the application starts, avoiding repeated disk reads per request.

### API Example

**Request**
```json
POST /predict
{
  "features": [1, 2, 3]
}