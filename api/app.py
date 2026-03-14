from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.predict import RiskModelPredictor

app = FastAPI(title="Financial Document Risk Analyzer API")

# Initialize model predictor. Will throw error on startup if model doesn't exist.
try:
    predictor = RiskModelPredictor()
except FileNotFoundError as e:
    print(f"Warning: {e}")
    predictor = None

class FinancialDataRecord(BaseModel):
    income: float
    debt: float
    credit_score: float
    employment_years: float

class RiskPredictionResponse(BaseModel):
    risk_level: str
    risk_probability: float
    explanations: list[str]

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/analyze", response_model=RiskPredictionResponse)
def analyze_risk(record: FinancialDataRecord):
    if not predictor:
        raise HTTPException(status_code=503, detail="Risk Model is not loaded. Please train it first.")
        
    df = pd.DataFrame([record.dict()])
    
    try:
        results = predictor.predict(df)
        return results[0]  # Return the single record's results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
