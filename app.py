from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "churn_pipeline.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "Model file not found. Run `python train.py` first."
    )

pipeline = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    version="1.0.0",
    description="Predicts whether a telecom customer is likely to churn.",
)


class CustomerRequest(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: Literal["Yes", "No"]
    churn_probability: float


@app.get("/")
def root():
    return {"message": "Telco Customer Churn Prediction API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerRequest):
    data = customer.model_dump()
    input_df = pd.DataFrame([data])

    prediction_value = int(pipeline.predict(input_df)[0])
    probability = float(pipeline.predict_proba(input_df)[0][1])

    return PredictionResponse(
        prediction="Yes" if prediction_value == 1 else "No",
        churn_probability=round(probability, 4),
    )
