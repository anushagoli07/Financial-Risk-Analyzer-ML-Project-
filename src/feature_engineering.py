import pandas as pd
from sklearn.preprocessing import StandardScaler

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes derived features like Debt-to-Income (DTI) ratio.
    Expects columns: ['income', 'debt', 'credit_score', 'employment_years']
    """
    # Create a copy to avoid SettingWithCopy warning
    features = df.copy()
    
    # Avoid division by zero
    features['debt_to_income_ratio'] = features['debt'] / (features['income'] + 1e-5)
    
    return features

def scale_features(features: pd.DataFrame, scaler: StandardScaler = None, is_training: bool = True):
    """
    Scales features to 0 mean and unit variance for model consumption.
    If training, fits the scaler and returns it.
    If inference, transforms using the provided scaler.
    """
    if is_training:
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(features)
        return pd.DataFrame(scaled_array, columns=features.columns), scaler
    else:
        if scaler is None:
            raise ValueError("Scaler must be provided for inference scaling.")
        scaled_array = scaler.transform(features)
        return pd.DataFrame(scaled_array, columns=features.columns)
