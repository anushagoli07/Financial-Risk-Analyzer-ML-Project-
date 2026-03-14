FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Assume the model should be shipped or trained at build time. We will train it at build time.
RUN python -m src.train_model

# Expose Streamlit and FastAPI ports
EXPOSE 8000 8501

# Command to run both the API and Streamlit with a simple shell script
# (In true production, use docker-compose with separate services)
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port 8000 & streamlit run frontend/dashboard.py --server.port 8501 --server.address 0.0.0.0"]
