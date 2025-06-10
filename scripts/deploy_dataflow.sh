#!/bin/bash

# Dataflow Pipeline Deployment Script
# This script deploys the IoT anomaly detection Dataflow pipeline to Google Cloud

set -e

# Configuration
PROJECT_ID=${1:-"your-project-id"}
REGION=${2:-"us-central1"}
TEMP_LOCATION=${3:-"gs://${PROJECT_ID}-dataflow-temp"}
STAGING_LOCATION=${4:-"gs://${PROJECT_ID}-dataflow-staging"}
SUBSCRIPTION=${5:-"projects/${PROJECT_ID}/subscriptions/temp-vibration-subscription"}
OUTPUT_TABLE=${6:-"${PROJECT_ID}:iot_anomaly_detection.sensor_data"}
JOB_NAME="iot-anomaly-detection-$(date +%Y%m%d-%H%M%S)"

echo "Deploying Dataflow pipeline..."
echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Job Name: $JOB_NAME"
echo "Input Subscription: $SUBSCRIPTION"
echo "Output Table: $OUTPUT_TABLE"

# Check if required buckets exist, create if not
echo "Checking/creating Cloud Storage buckets..."
gsutil ls gs://${PROJECT_ID}-dataflow-temp/ 2>/dev/null || gsutil mb gs://${PROJECT_ID}-dataflow-temp/
gsutil ls gs://${PROJECT_ID}-dataflow-staging/ 2>/dev/null || gsutil mb gs://${PROJECT_ID}-dataflow-staging/

# Deploy the pipeline
python ../src/dataflow_pipeline_temp_vibration.py \
    --runner DataflowRunner \
    --project $PROJECT_ID \
    --region $REGION \
    --temp_location $TEMP_LOCATION \
    --staging_location $STAGING_LOCATION \
    --input_subscription $SUBSCRIPTION \
    --output_table $OUTPUT_TABLE \
    --job_name $JOB_NAME \
    --window_size 60 \
    --max_num_workers 10 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --enable_streaming_engine \
    --use_public_ips false

echo "Dataflow pipeline deployed successfully!"
echo "Job Name: $JOB_NAME"
echo "Monitor the job at: https://console.cloud.google.com/dataflow/jobs/$REGION/$JOB_NAME?project=$PROJECT_ID"

