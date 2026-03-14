import os
import joblib
import pandas as pd
from .feature_engineering import extract_features, scale_features
from .explain_model import get_shap_explanations

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")

class RiskModelPredictor:
    def __init__(self):
        model_path = os.path.join(MODEL_DIR, "rf_risk_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Missing model/scaler artifacts in {MODEL_DIR}. Train the model first.")
            
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
    def predict(self, data: pd.DataFrame):
        """
        Receives raw data, applies feature engineering and scaling, returns Risk Probabilities, Classes, and SHAP rules.
        """
        # Feature Engineering
        X_engineered = extract_features(data)
        
        # Scaling
        X_scaled = scale_features(X_engineered, scaler=self.scaler, is_training=False)
        
        # Predict
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = self.model.predict(X_scaled)
        
        # SHAP Explanations
        shap_reasons = get_shap_explanations(self.model, X_scaled, list(X_scaled.columns))
        
        results = []
        for i in range(len(preds)):
            results.append({
                "risk_probability": round(float(probs[i]), 4),
                "risk_level": "HIGH" if preds[i] == 1 else "LOW",
                "explanations": shap_reasons[i]
            })
        return results

if __name__ == "__main__":
    predictor = RiskModelPredictor()
    test_df = pd.DataFrame([{
        "income": 65000,
        "debt": 45000,
        "credit_score": 580,
        "employment_years": 1
    }])
    import json
    print(json.dumps(predictor.predict(test_df), indent=2))
