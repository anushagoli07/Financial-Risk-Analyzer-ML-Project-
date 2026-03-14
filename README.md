# Intelligent Financial Document Risk Analyzer

An end-to-end intelligent document processing and risk analysis system utilizing Scikit-learn, SHAP, LlamaIndex, FastAPI, Streamlit, and Evidently data drift monitoring.

## System Architecture
Financial Documents -> LlamaIndex Extraction -> Feature Engineering (DTI) -> Random Forest ML Classification -> SHAP Explanations -> FastAPI Backend -> Streamlit Interactive Dashboard.

## Setup Instructions
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the machine learning model:
   ```bash
   python -m src.train_model
   ```
3. Run the API Server:
   ```bash
   uvicorn api.app:app --reload
   ```
4. Run the Streamlit UI (in a new terminal):
   ```bash
   streamlit run frontend/dashboard.py
   ```

## MLOps Features
- **Explainability**: SHAP attributes the exact reason a document was flagged HIGH RISK.
- **Monitoring**: `src/monitoring.py` analyzes data drift using Evidently AI.
- **CI/CD**: `.github/workflows/ci.yml` runs automated checks using GitHub Actions.
