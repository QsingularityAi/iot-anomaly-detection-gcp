#!/bin/bash

# Cleanup Script for IoT Anomaly Detection System
# This script removes all resources created by the deployment

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

# Derived variables
DATASET_ID="iot_anomaly_detection"
TOPIC_NAME="iot-sensor-data"
SUBSCRIPTION_NAME="dataflow-subscription"
BUCKET_PREFIX="${PROJECT_ID}-iot-anomaly"

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

# Function to check prerequisites
check_prerequisites() {
    if [ -z "$PROJECT_ID" ]; then
        print_error "Project ID is required. Usage: $0 <PROJECT_ID> [REGION]"
        exit 1
    fi
    
    if ! command -v gcloud >/dev/null 2>&1; then
        print_error "gcloud CLI is not installed."
        exit 1
    fi
    
    # Set the project
    gcloud config set project "$PROJECT_ID"
}

# Function to stop Dataflow jobs
stop_dataflow_jobs() {
    print_status "Stopping Dataflow jobs..."
    
    # Get all running Dataflow jobs
    jobs=$(gcloud dataflow jobs list --status=active --format="value(id)" --project="$PROJECT_ID" --region="$REGION" 2>/dev/null || true)
    
    if [ -n "$jobs" ]; then
        for job in $jobs; do
            print_status "Stopping Dataflow job: $job"
            gcloud dataflow jobs cancel "$job" --project="$PROJECT_ID" --region="$REGION" || true
        done
        
        # Wait for jobs to stop
        print_status "Waiting for Dataflow jobs to stop..."
        sleep 30
    else
        print_status "No active Dataflow jobs found"
    fi
}

# Function to delete Cloud Functions
delete_cloud_functions() {
    print_status "Deleting Cloud Functions..."
    
    functions=$(gcloud functions list --format="value(name)" --project="$PROJECT_ID" --region="$REGION" 2>/dev/null || true)
    
    if [ -n "$functions" ]; then
        for func in $functions; do
            if [[ "$func" == *"anomaly"* ]]; then
                print_status "Deleting Cloud Function: $func"
                gcloud functions delete "$func" --project="$PROJECT_ID" --region="$REGION" --quiet || true
            fi
        done
    else
        print_status "No Cloud Functions found"
    fi
}

# Function to delete Vertex AI endpoints and models
delete_vertex_ai_resources() {
    print_status "Deleting Vertex AI resources..."
    
    # Delete endpoints
    endpoints=$(gcloud ai endpoints list --format="value(name)" --project="$PROJECT_ID" --region="$REGION" 2>/dev/null || true)
    
    if [ -n "$endpoints" ]; then
        for endpoint in $endpoints; do
            print_status "Deleting Vertex AI endpoint: $endpoint"
            gcloud ai endpoints delete "$endpoint" --project="$PROJECT_ID" --region="$REGION" --quiet || true
        done
    fi
    
    # Delete models
    models=$(gcloud ai models list --format="value(name)" --project="$PROJECT_ID" --region="$REGION" 2>/dev/null || true)
    
    if [ -n "$models" ]; then
        for model in $models; do
            if [[ "$model" == *"anomaly"* ]]; then
                print_status "Deleting Vertex AI model: $model"
                gcloud ai models delete "$model" --project="$PROJECT_ID" --region="$REGION" --quiet || true
            fi
        done
    fi
}

# Function to delete BigQuery resources
delete_bigquery_resources() {
    print_status "Deleting BigQuery resources..."
    
    # Delete ML models
    models=$(bq ls -m --format=csv --project_id="$PROJECT_ID" "$DATASET_ID" 2>/dev/null | tail -n +2 | cut -d, -f1 || true)
    
    if [ -n "$models" ]; then
        for model in $models; do
            print_status "Deleting BigQuery ML model: $model"
            bq rm -f -m "$PROJECT_ID:$DATASET_ID.$model" || true
        done
    fi
    
    # Delete dataset (this will delete all tables)
    if bq ls -d "$PROJECT_ID:$DATASET_ID" >/dev/null 2>&1; then
        print_status "Deleting BigQuery dataset: $DATASET_ID"
        bq rm -r -f -d "$PROJECT_ID:$DATASET_ID" || true
    else
        print_status "BigQuery dataset $DATASET_ID not found"
    fi
}

# Function to delete Pub/Sub resources
delete_pubsub_resources() {
    print_status "Deleting Pub/Sub resources..."
    
    # Delete subscriptions
    subscriptions=$(gcloud pubsub subscriptions list --format="value(name)" --project="$PROJECT_ID" 2>/dev/null || true)
    
    if [ -n "$subscriptions" ]; then
        for sub in $subscriptions; do
            if [[ "$sub" == *"$SUBSCRIPTION_NAME"* ]] || [[ "$sub" == *"anomaly"* ]]; then
                print_status "Deleting Pub/Sub subscription: $sub"
                gcloud pubsub subscriptions delete "$sub" --project="$PROJECT_ID" --quiet || true
            fi
        done
    fi
    
    # Delete topics
    topics=$(gcloud pubsub topics list --format="value(name)" --project="$PROJECT_ID" 2>/dev/null || true)
    
    if [ -n "$topics" ]; then
        for topic in $topics; do
            if [[ "$topic" == *"$TOPIC_NAME"* ]] || [[ "$topic" == *"anomaly"* ]]; then
                print_status "Deleting Pub/Sub topic: $topic"
                gcloud pubsub topics delete "$topic" --project="$PROJECT_ID" --quiet || true
            fi
        done
    fi
}

# Function to delete Cloud Storage buckets
delete_storage_buckets() {
    print_status "Deleting Cloud Storage buckets..."
    
    buckets=$(gsutil ls -p "$PROJECT_ID" 2>/dev/null | grep "$BUCKET_PREFIX" || true)
    
    if [ -n "$buckets" ]; then
        for bucket in $buckets; do
            print_status "Deleting Cloud Storage bucket: $bucket"
            gsutil -m rm -r "$bucket" || true
        done
    else
        print_status "No matching Cloud Storage buckets found"
    fi
}

# Function to delete monitoring resources
delete_monitoring_resources() {
    print_status "Deleting monitoring resources..."
    
    # Delete alert policies
    policies=$(gcloud alpha monitoring policies list --format="value(name)" --project="$PROJECT_ID" 2>/dev/null || true)
    
    if [ -n "$policies" ]; then
        for policy in $policies; do
            if [[ "$policy" == *"anomaly"* ]] || [[ "$policy" == *"iot"* ]]; then
                print_status "Deleting monitoring policy: $policy"
                gcloud alpha monitoring policies delete "$policy" --project="$PROJECT_ID" --quiet || true
            fi
        done
    fi
}

# Function to delete service accounts
delete_service_accounts() {
    print_status "Deleting service accounts..."
    
    service_accounts=$(gcloud iam service-accounts list --format="value(email)" --project="$PROJECT_ID" 2>/dev/null || true)
    
    if [ -n "$service_accounts" ]; then
        for sa in $service_accounts; do
            if [[ "$sa" == *"iot-anomaly"* ]] || [[ "$sa" == *"dataflow"* ]]; then
                print_status "Deleting service account: $sa"
                gcloud iam service-accounts delete "$sa" --project="$PROJECT_ID" --quiet || true
            fi
        done
    fi
}

# Function to clean up local files
cleanup_local_files() {
    print_status "Cleaning up local files..."
    
    # Remove generated files
    files_to_remove=(
        "deployment_summary.md"
        "model_performance_report.md"
        "model_performance_comparison.png"
        "anomaly_alert_policy.json"
        "models/"
        "cloud_functions/"
        "__pycache__/"
        "*.pyc"
        "*.log"
    )
    
    for file in "${files_to_remove[@]}"; do
        if [ -e "$file" ]; then
            print_status "Removing: $file"
            rm -rf "$file" || true
        fi
    done
}

# Function to create cleanup summary
create_cleanup_summary() {
    print_status "Creating cleanup summary..."
    
    cat > cleanup_summary.md << EOF
# IoT Anomaly Detection System - Cleanup Summary

## Cleanup Information
- **Project ID**: $PROJECT_ID
- **Region**: $REGION
- **Cleanup Date**: $(date)

## Resources Removed

### Dataflow Jobs
- All active Dataflow jobs stopped and cancelled

### Cloud Functions
- All anomaly-related Cloud Functions deleted

### Vertex AI Resources
- All Vertex AI endpoints deleted
- All anomaly detection models deleted

### BigQuery Resources
- Dataset: $DATASET_ID (including all tables and ML models)

### Pub/Sub Resources
- Topic: $TOPIC_NAME and related topics
- Subscriptions: $SUBSCRIPTION_NAME and related subscriptions

### Cloud Storage Buckets
- All buckets with prefix: $BUCKET_PREFIX

### Monitoring Resources
- All anomaly-related alert policies deleted

### Service Accounts
- All IoT anomaly detection service accounts deleted

### Local Files
- Generated files and directories cleaned up

## Verification Steps

1. **Check remaining resources**:
   \`\`\`bash
   # Check for any remaining resources
   gcloud dataflow jobs list --project=$PROJECT_ID
   gcloud pubsub topics list --project=$PROJECT_ID
   bq ls --project_id=$PROJECT_ID
   gsutil ls -p $PROJECT_ID
   \`\`\`

2. **Verify billing**:
   - Check the billing console to ensure no unexpected charges
   - Review resource usage reports

## Notes
- Some resources may take a few minutes to be fully deleted
- Billing for usage up to the cleanup time will still apply
- If you encounter any issues, check the Cloud Console for remaining resources

EOF
    
    print_success "Cleanup summary created: cleanup_summary.md"
}

# Main cleanup function
main() {
    echo "=========================================="
    echo "IoT Anomaly Detection System Cleanup"
    echo "=========================================="
    echo ""
    
    check_prerequisites
    
    print_warning "This will delete ALL resources for the IoT Anomaly Detection System in project: $PROJECT_ID"
    print_warning "This action cannot be undone!"
    echo ""
    
    read -p "Are you sure you want to continue? (yes/no): " confirm
    
    if [ "$confirm" != "yes" ]; then
        print_status "Cleanup cancelled by user"
        exit 0
    fi
    
    print_status "Starting cleanup for project: $PROJECT_ID"
    echo ""
    
    # Execute cleanup steps in order
    stop_dataflow_jobs
    delete_cloud_functions
    delete_vertex_ai_resources
    delete_bigquery_resources
    delete_pubsub_resources
    delete_storage_buckets
    delete_monitoring_resources
    delete_service_accounts
    cleanup_local_files
    create_cleanup_summary
    
    echo ""
    print_success "=========================================="
    print_success "Cleanup completed successfully!"
    print_success "=========================================="
    echo ""
    print_status "Check cleanup_summary.md for details and verification steps."
}

# Run main function
main "$@"

