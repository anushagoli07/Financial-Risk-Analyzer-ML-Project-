import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
from .data_pipeline import load_training_data, fetch_reference_dataset
from .feature_engineering import extract_features

REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")

def generate_drift_report(current_data: pd.DataFrame):
    """
    Generates an Evidently Data Drift report comparing historical data against the incoming batch.
    """
    # 1. Load Reference Data
    # In a real system, you'd fetch the actual data the model was trained on
    historical_raw = load_training_data(num_samples=2000)
    reference_data = fetch_reference_dataset(historical_raw)
    reference_features = extract_features(reference_data)
    
    # 2. Process Current Data
    current_features = extract_features(current_data)
    
    # 3. Create Report
    data_drift_report = Report(metrics=[
        DataDriftPreset(),
    ])
    
    # We only care about feature columns, not target for drift detection of inputs
    columns = ['income', 'debt', 'credit_score', 'employment_years', 'debt_to_income_ratio']
    
    print("Generating Evidently Data Drift Report...")
    data_drift_report.run(reference_data=reference_features[columns], current_data=current_features[columns])
    
    os.makedirs(REPORT_DIR, exist_ok=True)
    report_path = os.path.join(REPORT_DIR, "data_drift_report.html")
    data_drift_report.save_html(report_path)
    
    print(f"Drift report generated at: {report_path}")

if __name__ == "__main__":
    # Simulate a drifted batch of incoming loan applications
    # e.g., sudden influx of low-income, high-debt applications
    drifted_df = pd.DataFrame([{
        "income": 30000,
        "debt": 80000,
        "credit_score": 450,
        "employment_years": 0.5
    } for _ in range(500)])
    
    generate_drift_report(drifted_df)
