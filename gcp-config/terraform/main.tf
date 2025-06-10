# Terraform configuration for IoT Anomaly Detection System
# This file defines the infrastructure as code for the complete system

terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Variables
variable "project_id" {
  description = "Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "Google Cloud Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "Google Cloud Zone"
  type        = string
  default     = "us-central1-a"
}

variable "dataset_id" {
  description = "BigQuery dataset ID"
  type        = string
  default     = "iot_anomaly_detection"
}

variable "topic_name" {
  description = "Pub/Sub topic name"
  type        = string
  default     = "iot-sensor-data"
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "compute.googleapis.com",
    "pubsub.googleapis.com",
    "bigquery.googleapis.com",
    "dataflow.googleapis.com",
    "cloudfunctions.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "run.googleapis.com"
  ])

  service = each.value
  project = var.project_id

  disable_dependent_services = true
}

# Cloud Storage Buckets
resource "google_storage_bucket" "temp_bucket" {
  name     = "${var.project_id}-iot-anomaly-temp"
  location = var.region

  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 7
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "staging_bucket" {
  name     = "${var.project_id}-iot-anomaly-staging"
  location = var.region

  uniform_bucket_level_access = true
  
  lifecycle_rule {
    condition {
      age = 30
    }
    action {
      type = "Delete"
    }
  }

  depends_on = [google_project_service.required_apis]
}

resource "google_storage_bucket" "model_bucket" {
  name     = "${var.project_id}-iot-anomaly-models"
  location = var.region

  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }

  depends_on = [google_project_service.required_apis]
}

# Pub/Sub Topic
resource "google_pubsub_topic" "iot_sensor_data" {
  name = var.topic_name

  message_retention_duration = "86400s" # 24 hours

  depends_on = [google_project_service.required_apis]
}

# Pub/Sub Subscription for Dataflow
resource "google_pubsub_subscription" "dataflow_subscription" {
  name  = "dataflow-subscription"
  topic = google_pubsub_topic.iot_sensor_data.name

  ack_deadline_seconds = 600 # 10 minutes for Dataflow processing

  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "600s"
  }

  dead_letter_policy {
    dead_letter_topic     = google_pubsub_topic.dead_letter_topic.id
    max_delivery_attempts = 5
  }

  depends_on = [google_project_service.required_apis]
}

# Dead Letter Topic
resource "google_pubsub_topic" "dead_letter_topic" {
  name = "${var.topic_name}-dead-letter"

  depends_on = [google_project_service.required_apis]
}

# Pub/Sub Subscription for Alerting
resource "google_pubsub_subscription" "alerting_subscription" {
  name  = "alerting-subscription"
  topic = google_pubsub_topic.iot_sensor_data.name

  filter = "attributes.is_anomaly=\"true\""

  depends_on = [google_project_service.required_apis]
}

# BigQuery Dataset
resource "google_bigquery_dataset" "iot_anomaly_detection" {
  dataset_id  = var.dataset_id
  location    = var.region
  description = "Dataset for IoT anomaly detection system"

  delete_contents_on_destroy = false

  depends_on = [google_project_service.required_apis]
}

# BigQuery Tables
resource "google_bigquery_table" "sensor_data" {
  dataset_id = google_bigquery_dataset.iot_anomaly_detection.dataset_id
  table_id   = "sensor_data"

  time_partitioning {
    type  = "DAY"
    field = "timestamp"
  }

  clustering = ["device_id", "device_type"]

  schema = jsonencode([
    {
      name = "device_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "device_type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "location"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "processed_at"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "temperature"
      type = "FLOAT"
      mode = "REQUIRED"
    },
    {
      name = "humidity"
      type = "FLOAT"
      mode = "REQUIRED"
    },
    {
      name = "pressure"
      type = "FLOAT"
      mode = "REQUIRED"
    },
    {
      name = "vibration_level"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "power_consumption"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "is_anomaly"
      type = "BOOLEAN"
      mode = "REQUIRED"
    },
    {
      name = "anomaly_type"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "anomaly_score"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "rule_based_anomaly"
      type = "BOOLEAN"
      mode = "NULLABLE"
    },
    {
      name = "ml_anomaly_score"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "ml_anomaly_prediction"
      type = "BOOLEAN"
      mode = "NULLABLE"
    }
  ])
}

resource "google_bigquery_table" "anomaly_alerts" {
  dataset_id = google_bigquery_dataset.iot_anomaly_detection.dataset_id
  table_id   = "anomaly_alerts"

  time_partitioning {
    type  = "DAY"
    field = "alert_timestamp"
  }

  schema = jsonencode([
    {
      name = "alert_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "device_id"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "alert_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "anomaly_type"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "severity"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "anomaly_score"
      type = "FLOAT"
      mode = "REQUIRED"
    },
    {
      name = "sensor_values"
      type = "JSON"
      mode = "NULLABLE"
    },
    {
      name = "alert_status"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "acknowledged_by"
      type = "STRING"
      mode = "NULLABLE"
    },
    {
      name = "acknowledged_at"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    },
    {
      name = "resolved_at"
      type = "TIMESTAMP"
      mode = "NULLABLE"
    }
  ])
}

resource "google_bigquery_table" "model_performance" {
  dataset_id = google_bigquery_dataset.iot_anomaly_detection.dataset_id
  table_id   = "model_performance"

  schema = jsonencode([
    {
      name = "model_name"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "model_version"
      type = "STRING"
      mode = "REQUIRED"
    },
    {
      name = "evaluation_timestamp"
      type = "TIMESTAMP"
      mode = "REQUIRED"
    },
    {
      name = "precision"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "recall"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "f1_score"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "accuracy"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "auc_roc"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "false_positive_rate"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "false_negative_rate"
      type = "FLOAT"
      mode = "NULLABLE"
    },
    {
      name = "training_data_size"
      type = "INTEGER"
      mode = "NULLABLE"
    },
    {
      name = "evaluation_data_size"
      type = "INTEGER"
      mode = "NULLABLE"
    }
  ])
}

# Service Account for Dataflow
resource "google_service_account" "dataflow_sa" {
  account_id   = "iot-anomaly-dataflow"
  display_name = "IoT Anomaly Detection Dataflow Service Account"
}

# IAM roles for Dataflow service account
resource "google_project_iam_member" "dataflow_worker" {
  project = var.project_id
  role    = "roles/dataflow.worker"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "bigquery_data_editor" {
  project = var.project_id
  role    = "roles/bigquery.dataEditor"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "pubsub_subscriber" {
  project = var.project_id
  role    = "roles/pubsub.subscriber"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

resource "google_project_iam_member" "storage_object_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.dataflow_sa.email}"
}

# IAM role for Cloud Functions service account to read Artifact Registry
resource "google_project_iam_member" "cloudfunctions_artifactregistry_reader" {
  project = var.project_id
  role    = "roles/artifactregistry.reader"
  member  = "serviceAccount:${var.project_id}@gcf-admin-robot.iam.gserviceaccount.com"
}

# Cloud Function for Anomaly Alerting
resource "google_cloudfunctions_function" "anomaly_alerter" {
  name        = "anomaly-alerter"
  description = "Function to handle anomaly alerts"
  runtime     = "python39"

  available_memory_mb   = 256
  source_archive_bucket = google_storage_bucket.staging_bucket.name
  source_archive_object = "anomaly-alerter.zip"
  entry_point          = "anomaly_alerter"

  event_trigger {
    event_type = "providers/cloud.pubsub/eventTypes/topic.publish"
    resource   = google_pubsub_topic.iot_sensor_data.name
  }

  depends_on = [google_project_service.required_apis]
}

# Monitoring Alert Policy
resource "google_monitoring_alert_policy" "high_anomaly_rate" {
  display_name = "High Anomaly Detection Rate"
  combiner     = "OR"

  conditions {
    display_name = "Anomaly rate condition"

    condition_threshold {
      filter          = "metric.type=\"pubsub.googleapis.com/topic/message_count\" AND resource.type=\"pubsub_topic\" AND resource.labels.topic_id=\"${var.topic_name}\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.1

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_RATE"
      }
    }
  }

  notification_channels = []

  depends_on = [google_project_service.required_apis]
}

# Outputs
output "project_id" {
  description = "Google Cloud Project ID"
  value       = var.project_id
}

output "region" {
  description = "Google Cloud Region"
  value       = var.region
}

output "pubsub_topic" {
  description = "Pub/Sub topic name"
  value       = google_pubsub_topic.iot_sensor_data.name
}

output "bigquery_dataset" {
  description = "BigQuery dataset ID"
  value       = google_bigquery_dataset.iot_anomaly_detection.dataset_id
}

output "storage_buckets" {
  description = "Cloud Storage bucket names"
  value = {
    temp    = google_storage_bucket.temp_bucket.name
    staging = google_storage_bucket.staging_bucket.name
    models  = google_storage_bucket.model_bucket.name
  }
}

output "service_account_email" {
  description = "Dataflow service account email"
  value       = google_service_account.dataflow_sa.email
}
