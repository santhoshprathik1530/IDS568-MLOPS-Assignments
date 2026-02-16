# Milestone 2 – ML Model Serving API

This milestone deploys a trained ML model as a FastAPI service.

## API
POST `/predict`

Request:
```json
{
  "features": [1, 2, 3]
}
