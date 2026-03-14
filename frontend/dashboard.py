import streamlit as st
import requests

API_URL = "http://127.0.0.0:8000"

st.set_page_config(page_title="Risk Analyzer Dashboard", page_icon="🏦", layout="wide")

st.title("Intelligent Financial Document Risk Analyzer")
st.markdown("Automated risk assessment powered by Scikit-Learn, SHAP, and LLM Extraction.")

# Sidebar for manual input
st.sidebar.header("Manual Data Entry")
income = st.sidebar.number_input("Income ($)", min_value=0.0, value=65000.0)
debt = st.sidebar.number_input("Debt ($)", min_value=0.0, value=45000.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=300, max_value=850, value=580)
employment_years = st.sidebar.number_input("Employment (Years)", min_value=0.0, value=1.0)

if st.sidebar.button("Analyze Risk"):
    payload = {
        "income": income,
        "debt": debt,
        "credit_score": credit_score,
        "employment_years": employment_years
    }
    
    with st.spinner("Analyzing profile..."):
        try:
            # Change out localhost if running inside docker network
            response = requests.post("http://127.0.0.1:8000/analyze", json=payload)
            if response.status_code == 200:
                data = response.json()
                
                # Display Results
                col1, col2 = st.columns(2)
                with col1:
                    risk_lvl = data.get("risk_level", "UNKNOWN")
                    color = "red" if risk_lvl == "HIGH" else "green"
                    st.markdown(f"### Risk Level: <span style='color:{color}'>{risk_lvl}</span>", unsafe_allow_html=True)
                    st.metric("Default Probability", f"{data.get('risk_probability', 0):.2%}")
                    
                with col2:
                    st.subheader("💡 Why this decision?")
                    for reason in data.get("explanations", []):
                        st.markdown(f"- {reason}")
            else:
                st.error(f"API Error: {response.text}")
                
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")

st.divider()
st.subheader("Upload Financial Document (PDF)")
st.caption("LLM Parser (LlamaIndex) pending integration. For now, use the manual data entry form.")
uploaded_file = st.file_uploader("Upload a Loan Application Statement", type=["pdf", "txt", "docx"])
if uploaded_file:
    st.info("Document uploaded. Parsing mechanism via LlamaIndex will route extracted fields to the risk model.")

