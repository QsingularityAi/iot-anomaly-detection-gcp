# 🚀 IoT Anomaly Detection System - Deployment Guide

This guide provides step-by-step instructions for deploying the IoT Anomaly Detection System on Google Cloud Platform.

## 📋 Prerequisites

### Required Tools
- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) (gcloud CLI)
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose
- [Terraform](https://www.terraform.io/downloads.html) >= 1.0
- Python 3.9+

### GCP Requirements
- Google Cloud Project with billing enabled
- Required APIs enabled (will be enabled by Terraform):
  - Pub/Sub API
  - BigQuery API
  - Dataflow API
  - Cloud Storage API
  - Vertex AI API
  - Cloud Functions API
  - Cloud Monitoring API

## 🔧 Setup Instructions

### 1. Clone and Initialize

```bash
git clone <repository-url>
cd iot-anomaly-detection-gcp
./setup.sh
```

### 2. Google Cloud Authentication

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login

# Set your project
gcloud config set project YOUR_PROJECT_ID
```

### 3. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create iot-anomaly-detection \
    --display-name="IoT Anomaly Detection Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:iot-anomaly-detection@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/bigquery.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:iot-anomaly-detection@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/pubsub.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:iot-anomaly-detection@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/dataflow.admin"

gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:iot-anomaly-detection@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"

# Create and download key
gcloud iam service-accounts keys create credentials.json \
    --iam-account=iot-anomaly-detection@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### 4. Configure Environment

```bash
# Copy and edit environment file
cp .env.template .env
```

Edit `.env` file:
```bash
PROJECT_ID=your-gcp-project-id
REGION=us-central1
ZONE=us-central1-a
TOPIC_NAME=iot-sensor-data
SUBSCRIPTION_NAME=dataflow-subscription
BIGQUERY_DATASET=iot_anomaly_detection
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

### 5. Configure Terraform

```bash
# Copy and edit Terraform variables
cp gcp-config/terraform/terraform.tfvars.template gcp-config/terraform/terraform.tfvars
```

Edit `terraform.tfvars`:
```hcl
project_id = "your-gcp-project-id"
region     = "us-central1"
dataset_id = "iot_anomaly_detection"
topic_name = "iot-sensor-data"
```

## 🏗️ Infrastructure Deployment

### 1. Deploy with Terraform

```bash
cd gcp-config/terraform

# Initialize Terraform
terraform init

# Review the deployment plan
terraform plan

# Deploy infrastructure
terraform apply
```

This will create:
- Pub/Sub topics and subscriptions
- BigQuery dataset and tables
- Cloud Storage buckets
- Service accounts and IAM roles
- Cloud Functions for alerting
- Monitoring alert policies

### 2. Verify Infrastructure

```bash
# Check Pub/Sub topics
gcloud pubsub topics list

# Check BigQuery datasets
bq ls

# Check Cloud Storage buckets
gsutil ls
```

## 🤖 BigQuery ML Models

### 1. Create ML Models

```bash
# Run BigQuery ML model creation
bq query --use_legacy_sql=false < gcp-config/bigquery_ml_models_temp_vibration.sql
```

### 2. Verify Models

```bash
# List models in BigQuery
bq ls -m iot_anomaly_detection
```

## 📊 Deploy Dataflow Pipeline

### 1. Build and Deploy

```bash
# Make deployment script executable
chmod +x scripts/deploy_dataflow.sh

# Deploy Dataflow pipeline
./scripts/deploy_dataflow.sh
```

### 2. Monitor Pipeline

```bash
# Check Dataflow jobs
gcloud dataflow jobs list --region=us-central1
```

## 🐳 Local Development Environment

### 1. Start Services

```bash
cd gcp-config
docker-compose up -d
```

This starts:
- IoT Device Simulator (port 8080)
- Data Consumer
- ML Model Trainer
- Streamlit Dashboard (port 8501)
- Jupyter Notebook (port 8888)
- Grafana Dashboard (port 3000)
- Prometheus (port 9090)

### 2. Access Services

- **Streamlit Dashboard**: http://localhost:8501
- **Jupyter Notebook**: http://localhost:8888
- **Grafana Dashboard**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## 🧪 Testing

### 1. Run System Tests

```bash
# Run comprehensive tests
python test_system.py
```

### 2. Test Individual Components

```bash
# Test ML models
python -c "from src.custom_ml_models_temp_vibration import TemperatureVibrationAnomalyDetector; detector = TemperatureVibrationAnomalyDetector(); data = detector.generate_synthetic_data(100); print('✅ ML models working')"

# Test IoT simulator
python src/iot_device_simulator_temp_vibration.py --help

# Test data consumer
python src/data_consumer.py --help
```

## 📈 Production Deployment

### 1. Deploy to Cloud Run (Optional)

```bash
# Build and deploy Streamlit dashboard
gcloud run deploy iot-anomaly-dashboard \
    --source=./streamlit-app \
    --region=us-central1 \
    --allow-unauthenticated
```

### 2. Set up Monitoring

```bash
# Create monitoring dashboard
gcloud monitoring dashboards create --config-from-file=gcp-config/monitoring-dashboard.json
```

### 3. Configure Alerting

```bash
# Alerting policies are created by Terraform
# Verify they're active
gcloud alpha monitoring policies list
```

## 🔍 Monitoring and Troubleshooting

### Key Metrics to Monitor

1. **Data Flow Metrics**:
   - Pub/Sub message throughput
   - Dataflow job status
   - BigQuery insert rates

2. **Anomaly Detection Metrics**:
   - Anomaly detection rate
   - Model prediction accuracy
   - Alert generation frequency

3. **System Health**:
   - Container resource usage
   - API response times
   - Error rates

### Common Issues

#### 1. Pub/Sub Connection Issues
```bash
# Check topic exists
gcloud pubsub topics describe iot-sensor-data

# Check subscription
gcloud pubsub subscriptions describe dataflow-subscription

# Test publishing
gcloud pubsub topics publish iot-sensor-data --message='{"test": "message"}'
```

#### 2. BigQuery Access Issues
```bash
# Check dataset permissions
bq show iot_anomaly_detection

# Test query
bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM `iot_anomaly_detection.sensor_data`'
```

#### 3. Dataflow Pipeline Issues
```bash
# Check job logs
gcloud dataflow jobs describe JOB_ID --region=us-central1

# View logs
gcloud logging read "resource.type=dataflow_job"
```

#### 4. Docker Issues
```bash
# Check container status
docker-compose ps

# View logs
docker-compose logs iot-simulator

# Restart services
docker-compose restart
```

## 🔄 Updates and Maintenance

### 1. Update Infrastructure

```bash
cd gcp-config/terraform
terraform plan
terraform apply
```

### 2. Update ML Models

```bash
# Retrain models with new data
bq query --use_legacy_sql=false < gcp-config/bigquery_ml_models_temp_vibration.sql
```

### 3. Update Application Code

```bash
# Rebuild containers
docker-compose build

# Restart services
docker-compose up -d
```

## 🔒 Security Best Practices

### 1. Service Account Security
- Use least privilege principle
- Rotate service account keys regularly
- Use Workload Identity when possible

### 2. Network Security
- Use VPC networks for isolation
- Configure firewall rules appropriately
- Enable audit logging

### 3. Data Security
- Enable encryption at rest and in transit
- Use Cloud KMS for key management
- Implement data retention policies

## 📚 Additional Resources

- [Google Cloud Pub/Sub Documentation](https://cloud.google.com/pubsub/docs)
- [BigQuery ML Documentation](https://cloud.google.com/bigquery-ml/docs)
- [Dataflow Documentation](https://cloud.google.com/dataflow/docs)
- [Terraform Google Provider](https://registry.terraform.io/providers/hashicorp/google/latest/docs)

## 🆘 Support

For issues and questions:

1. Check the [troubleshooting section](#monitoring-and-troubleshooting)
2. Review logs using the commands above
3. Run the test suite: `python test_system.py`
4. Create an issue in the repository

## 📝 Changelog

- **v1.0**: Initial deployment guide
- **v1.1**: Added BigQuery ML integration
- **v1.2**: Enhanced monitoring and alerting
- **v1.3**: Added comprehensive testing