from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Load the "Brain" you just trained
model = joblib.load("fraud_model.pkl")

# Define what a transaction looks like
class Transaction(BaseModel):
    distance_from_home: float
    purchase_price_ratio: float
    online_order: int

@app.get("/")
def home():
    return {"message": "Fraud Detection AI is Online"}

@app.post("/predict")
def predict(data: Transaction):
    # Convert incoming data to a DataFrame for the model
    input_df = pd.DataFrame([data.dict()])
    
    # Get the prediction (0 or 1) and the probability
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    return {
        "is_fraud": bool(prediction),
        "confidence": float(probability)
    }