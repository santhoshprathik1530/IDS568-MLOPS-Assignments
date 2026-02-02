# main.py
import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os

# ---------------------------
# App initialization
# ---------------------------
app = FastAPI(title="ML Model Serving API")

# ---------------------------
# Load model ONCE at startup
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

model = joblib.load(MODEL_PATH)

# ---------------------------
# Request / Response Schemas
# ---------------------------
class PredictRequest(BaseModel):
    features: List[float]

class PredictResponse(BaseModel):
    prediction: float
    version: str

# ---------------------------
# Prediction Endpoint
# ---------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    prediction = round(model.predict([request.features])[0], 2)
    return PredictResponse(
        prediction=prediction,
        version="v1.0"
    )