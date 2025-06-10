#!/bin/bash

# Master Deployment Script for IoT Anomaly Detection System
# This script deploys the complete IoT anomaly detection system on Google Cloud Platform

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${1:-""}
REGION=${2:-"us-central1"}
ZONE=${3:-"us-central1-a"}

# Derived variables
DATASET_ID="iot_anomaly_detection"
TOPIC_NAME="iot-temp-vibration-data"
SUBSCRIPTION_NAME="dataflow-subscription"
BUCKET_PREFIX="${PROJECT_ID}-iot-anomaly"
TEMP_BUCKET="${BUCKET_PREFIX}-temp"
STAGING_BUCKET="${BUCKET_PREFIX}-staging"
MODEL_BUCKET="${BUCKET_PREFIX}-models"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required. Usage: $0 <PROJECT_ID> [REGION] [ZONE]"
        exit 1
    fi
    
    if ! command_exists gcloud; then
        print_error "gcloud CLI is not installed. Please install Google Cloud SDK."
        exit 1
    fi
    
    if ! command_exists python3; then
        print_error "Python 3 is not installed."
        exit 1
    fi
    
    # Check if user is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Function to enable required APIs
enable_apis() {
    print_status "Enabling required Google Cloud APIs..."
    
    apis=(
        "compute.googleapis.com"
        "pubsub.googleapis.com"
        "bigquery.googleapis.com"
        "dataflow.googleapis.com"
        "cloudfunctions.googleapis.com"
        "monitoring.googleapis.com"
        "logging.googleapis.com"
        "storage.googleapis.com"
        "aiplatform.googleapis.com"
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        print_status "Enabling $api..."
        gcloud services enable "$api" --project="$PROJECT_ID"
    done
    
    print_success "All APIs enabled"
}

# Function to create Cloud Storage buckets
create_storage_buckets() {
    print_status "Creating Cloud Storage buckets..."
    
    buckets=("$TEMP_BUCKET" "$STAGING_BUCKET" "$MODEL_BUCKET")
    
    for bucket in "${buckets[@]}"; do
        if gsutil ls "gs://$bucket/" 2>/dev/null; then
            print_warning "Bucket gs://$bucket already exists"
        else
            print_status "Creating bucket gs://$bucket..."
            gsutil mb -p "$PROJECT_ID" -l "$REGION" "gs://$bucket/"
            print_success "Created bucket gs://$bucket"
        fi
    done
}

# Function to set up Pub/Sub resources
setup_pubsub() {
    print_status "Setting up Pub/Sub resources..."
    
    # Create topic
    if gcloud pubsub topics describe "$TOPIC_NAME" --project="$PROJECT_ID" 2>/dev/null; then
        print_warning "Topic $TOPIC_NAME already exists"
    else
        print_status "Creating topic $TOPIC_NAME..."
        gcloud pubsub topics create "$TOPIC_NAME" --project="$PROJECT_ID"
        print_success "Created topic $TOPIC_NAME"
    fi
    
    # Create subscription
    if gcloud pubsub subscriptions describe "$SUBSCRIPTION_NAME" --project="$PROJECT_ID" 2>/dev/null; then
        print_warning "Subscription $SUBSCRIPTION_NAME already exists"
    else
        print_status "Creating subscription $SUBSCRIPTION_NAME..."
        gcloud pubsub subscriptions create "$SUBSCRIPTION_NAME" \
            --topic="$TOPIC_NAME" \
            --project="$PROJECT_ID"
        print_success "Created subscription $SUBSCRIPTION_NAME"
    fi
}

# Function to set up BigQuery resources
setup_bigquery() {
    print_status "Setting up BigQuery resources..."
    
    # Create dataset
    if bq ls -d "$PROJECT_ID:$DATASET_ID" 2>/dev/null; then
        print_warning "Dataset $DATASET_ID already exists"
    else
        print_status "Creating dataset $DATASET_ID..."
        bq mk --location="$REGION" "$PROJECT_ID:$DATASET_ID"
        print_success "Created dataset $DATASET_ID"
    fi
    
    # Run BigQuery setup script
    print_status "Creating BigQuery tables..."
    python3 ../gcp-config/setup_bigquery.py --project-id "$PROJECT_ID" --dataset-id "$DATASET_ID"
    print_success "BigQuery tables created"
}

# Function to deploy Dataflow pipeline
deploy_dataflow() {
    print_status "Deploying Dataflow pipeline..."
    
    # Install required Python packages
    pip3 install apache-beam[gcp] --quiet
    
    # Deploy the pipeline
    ./deploy_dataflow.sh "$PROJECT_ID" "$REGION" \
        "gs://$TEMP_BUCKET" "gs://$STAGING_BUCKET" \
        "projects/$PROJECT_ID/subscriptions/$SUBSCRIPTION_NAME" \
        "$PROJECT_ID:$DATASET_ID.sensor_data"
    
    print_success "Dataflow pipeline deployed"
}

# Function to train BigQuery ML models
train_bigquery_models() {
    print_status "Training BigQuery ML models..."
    
    # Execute BigQuery ML model training scripts
    bq query --use_legacy_sql=false --project_id="$PROJECT_ID" < ../gcp-config/bigquery_ml_models_temp_vibration.sql
    
    print_success "BigQuery ML models training initiated"
}

# Function to set up monitoring and alerting
setup_monitoring() {
    print_status "Setting up monitoring and alerting..."
    
    # Create alerting policies (this would typically be done via Terraform or gcloud)
    print_status "Creating alerting policies..."
    
    # Example: Create an alert for high anomaly detection rate
    cat > anomaly_alert_policy.json << EOF
{
  "displayName": "High Anomaly Detection Rate",
  "documentation": {
    "content": "Alert when anomaly detection rate exceeds 10% over 5 minutes"
  },
  "conditions": [
    {
      "displayName": "Anomaly rate condition",
      "conditionThreshold": {
        "filter": "resource.type=\"pubsub_topic\" AND resource.labels.topic_id=\"$TOPIC_NAME\"",
        "comparison": "COMPARISON_GREATER_THAN",
        "thresholdValue": 0.1,
        "duration": "300s",
        "aggregations": [
          {
            "alignmentPeriod": "60s",
            "perSeriesAligner": "ALIGN_RATE",
            "crossSeriesReducer": "REDUCE_MEAN"
          }
        ]
      }
    }
  ],
  "enabled": true
}
EOF
    
    # Create the alert policy
    gcloud alpha monitoring policies create --policy-from-file=anomaly_alert_policy.json --project="$PROJECT_ID"
    
    print_success "Monitoring and alerting configured"
}

# Function to create service accounts and IAM roles
setup_iam() {
    print_status "Setting up IAM roles and service accounts..."
    
    # Create service account for Dataflow
    SA_NAME="iot-anomaly-dataflow"
    SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"
    
    if gcloud iam service-accounts describe "$SA_EMAIL" --project="$PROJECT_ID" 2>/dev/null; then
        print_warning "Service account $SA_EMAIL already exists"
    else
        print_status "Creating service account $SA_EMAIL..."
        gcloud iam service-accounts create "$SA_NAME" \
            --display-name="IoT Anomaly Detection Dataflow Service Account" \
            --project="$PROJECT_ID"
        print_success "Created service account $SA_EMAIL"
    fi
    
    # Grant necessary roles
    roles=(
        "roles/dataflow.worker"
        "roles/bigquery.dataEditor"
        "roles/pubsub.subscriber"
        "roles/storage.objectAdmin"
    )
    
    for role in "${roles[@]}"; do
        print_status "Granting role $role to $SA_EMAIL..."
        gcloud projects add-iam-policy-binding "$PROJECT_ID" \
            --member="serviceAccount:$SA_EMAIL" \
            --role="$role"
    done
    
    print_success "IAM configuration completed"
}

# Function to deploy Cloud Functions for alerting
deploy_cloud_functions() {
    print_status "Deploying Cloud Functions for alerting..."
    
    # Create a simple alerting function
    mkdir -p cloud_functions/anomaly_alerter
    
    cat > cloud_functions/anomaly_alerter/main.py << 'EOF'
import json
import logging
from google.cloud import pubsub_v1
from google.cloud import bigquery

def anomaly_alerter(event, context):
    """Cloud Function triggered by Pub/Sub messages containing anomalies."""
    
    # Decode the Pub/Sub message
    pubsub_message = json.loads(base64.b64decode(event['data']).decode('utf-8'))
    
    # Check if this is an anomaly
    if pubsub_message.get('is_anomaly', False):
        device_id = pubsub_message.get('device_id')
        anomaly_score = pubsub_message.get('anomaly_score', 0)
        timestamp = pubsub_message.get('timestamp')
        
        # Log the anomaly
        logging.warning(f"Anomaly detected: Device {device_id}, Score: {anomaly_score}, Time: {timestamp}")
        
        # Here you could send notifications via email, Slack, etc.
        # For now, we'll just log it
        
        return f"Anomaly processed for device {device_id}"
    
    return "No anomaly detected"
EOF
    
    cat > cloud_functions/anomaly_alerter/requirements.txt << 'EOF'
google-cloud-pubsub==2.18.1
google-cloud-bigquery==3.11.4
EOF
    
    # Deploy the function
    gcloud functions deploy anomaly-alerter \
        --runtime python39 \
        --trigger-topic "$TOPIC_NAME" \
        --source cloud_functions/anomaly_alerter \
        --project "$PROJECT_ID" \
        --region "$REGION"
    
    print_success "Cloud Functions deployed"
}

# Function to create deployment summary
create_deployment_summary() {
    print_status "Creating deployment summary..."
    
    cat > deployment_summary.md << EOF
# IoT Anomaly Detection System - Deployment Summary

## Deployment Information
- **Project ID**: $PROJECT_ID
- **Region**: $REGION
- **Zone**: $ZONE
- **Deployment Date**: $(date)

## Resources Created

### Cloud Storage Buckets
- **Temp Bucket**: gs://$TEMP_BUCKET
- **Staging Bucket**: gs://$STAGING_BUCKET
- **Model Bucket**: gs://$MODEL_BUCKET

### Pub/Sub Resources
- **Topic**: $TOPIC_NAME
- **Subscription**: $SUBSCRIPTION_NAME

### BigQuery Resources
- **Dataset**: $DATASET_ID
- **Tables**: sensor_data, anomaly_alerts, model_performance

### Dataflow Pipeline
- **Pipeline**: iot-anomaly-detection-$(date +%Y%m%d)
- **Input**: projects/$PROJECT_ID/subscriptions/$SUBSCRIPTION_NAME
- **Output**: $PROJECT_ID:$DATASET_ID.sensor_data

### Service Accounts
- **Dataflow SA**: iot-anomaly-dataflow@$PROJECT_ID.iam.gserviceaccount.com

### Cloud Functions
- **Anomaly Alerter**: anomaly-alerter

## Next Steps

1. **Start Data Simulation**:
   \`\`\`bash
   python3 iot_device_simulator.py --project-id $PROJECT_ID --topic-name $TOPIC_NAME
   \`\`\`

2. **Monitor the Pipeline**:
   - Dataflow Console: https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID
   - BigQuery Console: https://console.cloud.google.com/bigquery?project=$PROJECT_ID

3. **Train Custom ML Models**:
   \`\`\`bash
   python3 custom_ml_models.py
   \`\`\`

4. **Evaluate Model Performance**:
   \`\`\`bash
   python3 model_evaluation.py --project-id $PROJECT_ID --simulate
   \`\`\`

## Monitoring URLs
- **Cloud Console**: https://console.cloud.google.com/home/dashboard?project=$PROJECT_ID
- **Monitoring**: https://console.cloud.google.com/monitoring?project=$PROJECT_ID
- **Logging**: https://console.cloud.google.com/logs?project=$PROJECT_ID

## Troubleshooting
- Check Dataflow job logs for pipeline issues
- Monitor Pub/Sub metrics for message flow
- Review BigQuery job history for ML model training
- Check Cloud Function logs for alerting issues

EOF
    
    print_success "Deployment summary created: deployment_summary.md"
}

# Main deployment function
main() {
    echo "=========================================="
    echo "IoT Anomaly Detection System Deployment"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    
    print_status "Starting deployment for project: $PROJECT_ID"
    print_status "Region: $REGION"
    print_status "Zone: $ZONE"
    echo ""
    
    # Set the project
    gcloud config set project "$PROJECT_ID"
    
    # Execute deployment steps
    enable_apis
    setup_iam
    create_storage_buckets
    setup_pubsub
    setup_bigquery
    deploy_dataflow
    train_bigquery_models
    setup_monitoring
    deploy_cloud_functions
    create_deployment_summary
    
    echo ""
    print_success "=========================================="
    print_success "Deployment completed successfully!"
    print_success "=========================================="
    echo ""
    print_status "Check deployment_summary.md for next steps and monitoring URLs."
}

# Run main function
main "$@"

