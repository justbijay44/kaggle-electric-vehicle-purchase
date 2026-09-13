import pandas as pd

from fastapi import FastAPI

from src.predict import predict
from src.schemas import EVFeatures

app = FastAPI()

@app.post("/predict")
def predict_endpoint(features: EVFeatures):
    df = pd.DataFrame([features.model_dump()])
    probability = predict(df)[0]
    return {"probability": float(probability)}