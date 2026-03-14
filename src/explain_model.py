import shap
import pandas as pd
from typing import Dict, Any

def get_shap_explanations(model, scaled_features_df: pd.DataFrame, feature_names: list) -> list:
    """
    Given a tree-based model and scaled input features, returns top contributing factors using SHAP.
    feature_names should be the raw column names matching the scaled df.
    """
    explainer = shap.TreeExplainer(model)
    
    # Calculate shap values for positive class
    # For a RandomForest with 2 classes, explainer.shap_values usually returns a list of 2 arrays.
    # index 1 represents probability of positive "Default / HIGH" class.
    shap_values = explainer.shap_values(scaled_features_df)
    
    # Handle single sample vs batch
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        # shap >= 0.40 returns Explanation objects for some models
        if len(shap_values.shape) == 3:
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values

    explanations = []
    
    for row_idx in range(len(scaled_features_df)):
        row_shap = shap_values_pos[row_idx]
        
        # Pair feature names with their shap impact
        impacts = list(zip(feature_names, row_shap))
        
        # Sort by absolute impact magnitude to explain the biggest drivers
        impacts.sort(key=lambda x: abs(x[1]), reverse=True)
        
        reasons = []
        for feat, val in impacts[:3]: # Top 3 features
            direction = "increased" if val > 0 else "decreased"
            reasons.append(f"{feat.replace('_', ' ').capitalize()} {direction} the risk score by {abs(val):.3f}")
            
        explanations.append(reasons)
        
    return explanations
