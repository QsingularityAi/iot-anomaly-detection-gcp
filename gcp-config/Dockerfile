# Multi-stage Dockerfile for IoT Anomaly Detection System
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy

# Copy source code
COPY . .

# Set development environment
ENV ENVIRONMENT=development

# Expose port for development server
EXPOSE 8080

# Default command for development
CMD ["python", "src/iot_device_simulator_temp_vibration.py", "--help"]

# Production stage
FROM base as production

# Copy only necessary files
COPY src/iot_device_simulator_temp_vibration.py ./src/
COPY src/data_consumer.py ./src/
COPY src/custom_ml_models_temp_vibration.py ./src/
COPY src/model_evaluation.py ./src/
COPY src/dataflow_pipeline_temp_vibration.py ./src/
COPY gcp-config/config/ ./config/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
USER app

# Set production environment
ENV ENVIRONMENT=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "src/data_consumer.py"]

# Dataflow stage (for Dataflow pipeline)
FROM apache/beam_python3.9_sdk:latest as dataflow

# Copy pipeline code
COPY src/dataflow_pipeline_temp_vibration.py /opt/apache/beam/

# Set working directory
WORKDIR /opt/apache/beam

# Default command for Dataflow
CMD ["python", "dataflow_pipeline_temp_vibration.py"]

# ML Model Serving stage
FROM python:3.9-slim as ml-serving

# Install serving dependencies
RUN pip install --no-cache-dir \
    flask==2.3.3 \
    gunicorn==21.2.0 \
    scikit-learn==1.3.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    joblib==1.3.2 \
    google-cloud-storage==2.10.0

# Create app directory
WORKDIR /app

# Copy model serving code
COPY gcp-config/deploy_vertex_ai.py .
COPY src/custom_ml_models_temp_vibration.py .

# Create serving script
RUN echo '#!/usr/bin/env python3\n\
import os\n\
import json\n\
import joblib\n\
import numpy as np\n\
import pandas as pd\n\
from flask import Flask, request, jsonify\n\
from google.cloud import storage\n\
\n\
app = Flask(__name__)\n\
\n\
# Global variables for model\n\
model = None\n\
scaler = None\n\
\n\
def load_model():\n\
    global model, scaler\n\
    # Load model from local files or Cloud Storage\n\
    # Implementation depends on deployment method\n\
    pass\n\
\n\
@app.route("/predict", methods=["POST"])\n\
def predict():\n\
    try:\n\
        data = request.get_json()\n\
        # Implement prediction logic\n\
        return jsonify({"prediction": "anomaly", "score": 0.8})\n\
    except Exception as e:\n\
        return jsonify({"error": str(e)}), 500\n\
\n\
@app.route("/health", methods=["GET"])\n\
def health():\n\
    return jsonify({"status": "healthy"})\n\
\n\
if __name__ == "__main__":\n\
    load_model()\n\
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))\n\
' > serve.py

# Expose port
EXPOSE 8080

# Run with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "serve:app"]

