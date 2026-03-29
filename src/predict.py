import joblib
import pandas as pd

from src.config import MODEL_PATH


def load_model():
    return joblib.load(MODEL_PATH)


def predict(features: list[float]):
    model = load_model()
    data = pd.DataFrame([features])
    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])
    return {"prediction": prediction, "probability": probability}
