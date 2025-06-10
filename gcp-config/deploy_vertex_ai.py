#!/usr/bin/env python3
"""
Vertex AI Model Deployment Script

This script deploys custom ML models to Vertex AI for real-time inference.
"""

import argparse
import json
import os
from google.cloud import aiplatform
from google.cloud import storage
import joblib
import tensorflow as tf

class VertexAIModelDeployer:
    """Deploy custom ML models to Vertex AI."""
    
    def __init__(self, project_id: str, region: str = "us-central1"):
        self.project_id = project_id
        self.region = region
        aiplatform.init(project=project_id, location=region)
        
    def create_custom_container_image(self, model_path: str, image_uri: str) -> None:
        """Create a custom container image for model serving."""
        # This would typically involve building a Docker image with the model
        # For now, we'll create a simple serving script
        
        serving_script = '''
import os
import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from google.cloud import storage

app = Flask(__name__)

# Load model on startup
model = None
scaler = None

def load_model():
    global model, scaler
    # Download model from Cloud Storage
    client = storage.Client()
    bucket = client.bucket(os.environ['MODEL_BUCKET'])
    
    # Download model files
    blob = bucket.blob('ensemble_anomaly_detector/isolation_forest.pkl')
    blob.download_to_filename('/tmp/isolation_forest.pkl')
    
    blob = bucket.blob('ensemble_anomaly_detector/one_class_svm.pkl')
    blob.download_to_filename('/tmp/one_class_svm.pkl')
    
    blob = bucket.blob('ensemble_anomaly_detector/scaler.pkl')
    blob.download_to_filename('/tmp/scaler.pkl')
    
    # Load models
    isolation_forest = joblib.load('/tmp/isolation_forest.pkl')
    one_class_svm = joblib.load('/tmp/one_class_svm.pkl')
    scaler = joblib.load('/tmp/scaler.pkl')
    
    model = {
        'isolation_forest': isolation_forest,
        'one_class_svm': one_class_svm
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data
        data = request.get_json()
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Prepare features
        features = ['temperature', 'humidity', 'pressure', 'vibration_level', 'power_consumption', 'heat_index']
        X = df[features].fillna(0).values
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = {}
        scores = {}
        
        for name, model_obj in model.items():
            pred = model_obj.predict(X_scaled)
            predictions[name] = (pred == -1).astype(int).tolist()
            
            if hasattr(model_obj, 'decision_function'):
                scores[name] = (-model_obj.decision_function(X_scaled)).tolist()
        
        # Ensemble prediction
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        predictions['ensemble'] = (ensemble_pred > 0.5).astype(int).tolist()
        
        ensemble_score = np.mean(list(scores.values()), axis=0)
        scores['ensemble'] = ensemble_score.tolist()
        
        return jsonify({
            'predictions': predictions,
            'scores': scores,
            'is_anomaly': bool(predictions['ensemble'][0]),
            'anomaly_score': float(scores['ensemble'][0])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
'''
        
        # Save serving script
        with open(os.path.join(model_path, 'serve.py'), 'w') as f:
            f.write(serving_script)
        
        # Create Dockerfile
        dockerfile = '''
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY serve.py .

EXPOSE 8080

CMD ["python", "serve.py"]
'''
        
        with open(os.path.join(model_path, 'Dockerfile'), 'w') as f:
            f.write(dockerfile)
        
        # Create requirements.txt for serving
        requirements = '''
flask==2.3.3
google-cloud-storage==2.10.0
scikit-learn==1.3.0
pandas==2.0.3
numpy==1.24.3
joblib==1.3.2
'''
        
        with open(os.path.join(model_path, 'requirements.txt'), 'w') as f:
            f.write(requirements)
    
    def upload_model_to_gcs(self, local_model_path: str, bucket_name: str, model_name: str) -> str:
        """Upload model artifacts to Google Cloud Storage."""
        client = storage.Client(project=self.project_id)
        
        # Create bucket if it doesn't exist
        try:
            bucket = client.create_bucket(bucket_name)
            print(f"Created bucket {bucket_name}")
        except Exception:
            bucket = client.bucket(bucket_name)
            print(f"Using existing bucket {bucket_name}")
        
        # Upload model files
        for root, dirs, files in os.walk(local_model_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_model_path)
                blob_name = f"{model_name}/{relative_path}"
                
                blob = bucket.blob(blob_name)
                blob.upload_from_filename(local_file_path)
                print(f"Uploaded {local_file_path} to gs://{bucket_name}/{blob_name}")
        
        return f"gs://{bucket_name}/{model_name}"
    
    def deploy_model(self, model_display_name: str, model_uri: str, container_image_uri: str) -> str:
        """Deploy model to Vertex AI endpoint."""
        
        # Upload model
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_uri,
            serving_container_image_uri=container_image_uri,
            serving_container_predict_route="/predict",
            serving_container_health_route="/health",
            serving_container_ports=[8080],
        )
        
        print(f"Model uploaded: {model.resource_name}")
        
        # Create endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name=f"{model_display_name}-endpoint"
        )
        
        print(f"Endpoint created: {endpoint.resource_name}")
        
        # Deploy model to endpoint
        model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=model_display_name,
            machine_type="n1-standard-2",
            min_replica_count=1,
            max_replica_count=3,
        )
        
        print(f"Model deployed to endpoint: {endpoint.resource_name}")
        return endpoint.resource_name


def main():
    """Main function to deploy models to Vertex AI."""
    parser = argparse.ArgumentParser(description='Deploy ML models to Vertex AI')
    parser.add_argument('--project-id', required=True, help='Google Cloud Project ID')
    parser.add_argument('--region', default='us-central1', help='Google Cloud region')
    parser.add_argument('--model-path', required=True, help='Local path to model artifacts')
    parser.add_argument('--bucket-name', required=True, help='GCS bucket name for model storage')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--container-image', help='Custom container image URI')
    
    args = parser.parse_args()
    
    deployer = VertexAIModelDeployer(args.project_id, args.region)
    
    # Create serving container files
    deployer.create_custom_container_image(args.model_path, args.container_image)
    
    # Upload model to GCS
    model_uri = deployer.upload_model_to_gcs(
        args.model_path, 
        args.bucket_name, 
        args.model_name
    )
    
    # Use pre-built serving container if custom image not provided
    container_image = args.container_image or "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
    
    # Deploy model
    endpoint_name = deployer.deploy_model(
        args.model_name,
        model_uri,
        container_image
    )
    
    print(f"\nDeployment completed!")
    print(f"Endpoint: {endpoint_name}")
    print(f"Model URI: {model_uri}")

if __name__ == "__main__":
    main()

