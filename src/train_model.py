import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from .data_pipeline import load_training_data
from .feature_engineering import extract_features, scale_features

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models")

def train_and_save_model():
    """
    Loads data, extracts features, trains a Random Forest Classifier to predict
    financial risk, and saves the serialized model pipelines for Serving.
    """
    print("Loading synthetic financial training data...")
    df = load_training_data(num_samples=2000)
    
    X_raw = df[['income', 'debt', 'credit_score', 'employment_years']]
    y = df['default']
    
    # Feature Engineering
    X_engineered = extract_features(X_raw)
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_engineered, y, test_size=0.2, random_state=42)
    
    # Scale Data
    X_train_scaled, scaler = scale_features(X_train, is_training=True)
    X_test_scaled = scale_features(X_test, scaler=scaler, is_training=False)
    
    # Model architecture
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    predictions = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Evaluation ->")
    print(classification_report(y_test, predictions))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # Create directory and save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "rf_risk_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel & Scaler saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_and_save_model()
