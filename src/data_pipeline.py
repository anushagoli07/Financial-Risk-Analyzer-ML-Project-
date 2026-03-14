import pandas as pd
import numpy as np

def load_training_data(filepath=None, num_samples=1000, random_state=42):
    """
    Loads historical training data. If no filepath is provided, generates
    a synthetic dataset of financial records for training the risk model.
    """
    if filepath:
        return pd.read_csv(filepath)
    
    # Generate synthetic data
    np.random.seed(random_state)
    
    # Feature distributions
    income = np.random.normal(loc=65000, scale=20000, size=num_samples).clip(min=20000)
    debt = np.random.normal(loc=30000, scale=15000, size=num_samples).clip(min=0)
    credit_score = np.random.normal(loc=680, scale=80, size=num_samples).clip(min=300, max=850)
    employment_years = np.random.exponential(scale=5, size=num_samples).clip(min=0, max=40)
    
    # Artificial Risk Target (1 = High Risk / Default, 0 = Low Risk)
    # Higher debt/income ratio, lower credit score -> higher risk
    base_risk_score = (debt / income) * 100 - (credit_score - 600) * 0.5 - employment_years * 2
    
    # Introduce some noise
    risk_score = base_risk_score + np.random.normal(loc=0, scale=10, size=num_samples)
    
    # Threshold for High Risk (approx 20-30% high risk)
    threshold = np.percentile(risk_score, 75)
    target = (risk_score >= threshold).astype(int)
    
    df = pd.DataFrame({
        "income": income,
        "debt": debt,
        "credit_score": credit_score,
        "employment_years": employment_years,
        "default": target
    })
    
    return df

def fetch_reference_dataset(data: pd.DataFrame):
    """
    Returns the reference dataset baseline needed for data drift monitoring.
    For simplicity, could be the historical data.
    """
    return data
