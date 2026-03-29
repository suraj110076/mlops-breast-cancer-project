from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict
from src.config import MODEL_PATH

app = FastAPI(title="Breast Cancer Prediction API", version="1.0.0")


class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=30, max_length=30)


@app.get("/")
def home():
    return {"message": "MLOps Breast Cancer Classifier API is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": Path(MODEL_PATH).exists(),
    }


@app.post("/predict")
def make_prediction(request: PredictionRequest):
    if not Path(MODEL_PATH).exists():
        raise HTTPException(
            status_code=400,
            detail="Model not found. Please run `python run_pipeline.py` first.",
        )

    result = predict(request.features)
    return result
