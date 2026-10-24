from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(
    title="SMARD Price Predictor API",
    description="API to predict electricity prices and spike volatility based on generation data."
)

CLF_PATH = os.path.join("models", "price_spike_model.pkl")
REG_PATH = os.path.join("models", "price_value_model.pkl")
FEATURES_PATH = os.path.join("models", "model_features.pkl")

clf_model = joblib.load(CLF_PATH)
reg_model = joblib.load(REG_PATH)
feature_names = joblib.load(FEATURES_PATH)

class PredictionRequest(BaseModel):
    wind_onshore: float
    wind_offshore: float
    solar_pv: float
    fossil_gas: float

@app.post("/predict")
def predict_market(data: PredictionRequest):
    input_df = pd.DataFrame([{
        'Wind onshore [MWh] Calculated resolutions': data.wind_onshore,
        'Wind offshore [MWh] Calculated resolutions': data.wind_offshore,
        'Photovoltaics [MWh] Calculated resolutions': data.solar_pv,
        'Fossil gas [MWh] Calculated resolutions': data.fossil_gas
    }])
    
    # Volatility Classification
    spike_prediction = int(clf_model.predict(input_df)[0])
    spike_probability = float(clf_model.predict_proba(input_df)[0][1])
    
    # Exact Price Regression
    exact_price = float(reg_model.predict(input_df)[0])
    
    return {
        "predicted_price": round(exact_price, 2),
        "spike_prediction": spike_prediction,
        "spike_probability": round(spike_probability * 100, 1),
        "status": "Price Spike Expected (>100 €/MWh)" if spike_prediction == 1 else "Normal Price Expected (<100 €/MWh)"
    }