# IoT Anomaly Detection System on Google Cloud Platform

A comprehensive IoT anomaly detection system built on Google Cloud Platform, featuring real-time data processing, machine learning-based anomaly detection, and automated alerting for temperature and vibration sensors.

## 🏗️ Architecture Overview

This system processes IoT sensor data through a complete pipeline:

- **Data Ingestion**: IoT devices publish sensor data to Google Cloud Pub/Sub
- **Stream Processing**: Apache Beam/Dataflow processes data in real-time
- **Storage**: BigQuery stores processed data and anomaly results
- **ML Models**: Custom anomaly detection models for temperature and vibration data
- **Monitoring**: Real-time dashboards and alerting system
- **Infrastructure**: Terraform-managed GCP resources

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Terraform >= 1.0
- Google Cloud SDK (gcloud)
- Google Cloud Project with billing enabled

### 1. Clone and Setup

```bash
git clone <repository-url>
cd iot-anomaly-detection-gcp
./setup.sh
```

### 2. Configure Environment

Edit `.env` file with your GCP project details:

```bash
PROJECT_ID=your-gcp-project-id
REGION=us-central1
TOPIC_NAME=iot-sensor-data
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json
```

### 3. Configure Terraform

Edit `gcp-config/terraform/terraform.tfvars`:

```hcl
project_id = "your-gcp-project-id"
region     = "us-central1"
dataset_id = "iot_anomaly_detection"
topic_name = "iot-sensor-data"
```

### 4. Deploy Infrastructure

```bash
cd gcp-config/terraform
terraform plan
terraform apply
```

### 5. Start Local Development

```bash
cd gcp-config
docker-compose up
```

## 📁 Project Structure

```
iot-anomaly-detection-gcp/
├── src/                                    # Source code
│   ├── iot_device_simulator_temp_vibration.py
│   ├── data_consumer.py
│   ├── dataflow_pipeline_temp_vibration.py
│   ├── custom_ml_models_temp_vibration.py
│   └── model_evaluation.py
├── gcp-config/                            # GCP configuration
│   ├── terraform/                         # Infrastructure as Code
│   │   ├── main.tf
│   │   └── terraform.tfvars.template
│   ├── Dockerfile                         # Multi-stage Docker build
│   ├── docker-compose.yml                # Local development environment
│   └── config/                           # Configuration files
├── streamlit-app/                         # Dashboard application
│   ├── app.py
│   └── requirements.txt
├── scripts/                               # Deployment scripts
├── requirements.txt                       # Python dependencies
├── setup.sh                             # Setup script
└── README.md
```

## 🐳 Docker Services

The system includes several containerized services:

- **iot-simulator**: Generates synthetic IoT sensor data
- **data-consumer**: Consumes and processes Pub/Sub messages
- **ml-trainer**: Trains anomaly detection models
- **ml-server**: Serves ML models via REST API
- **jupyter**: Jupyter notebook for data analysis
- **grafana**: Monitoring dashboard
- **prometheus**: Metrics collection
- **redis**: Caching and session storage

## 🏗️ Infrastructure Components

### Google Cloud Resources

- **Pub/Sub**: Message queuing for IoT data
- **BigQuery**: Data warehouse for sensor data and results
- **Dataflow**: Stream processing pipeline
- **Cloud Storage**: Model artifacts and temporary files
- **Cloud Functions**: Serverless alerting
- **Vertex AI**: ML model deployment
- **Monitoring**: Alerting and observability

### Terraform Modules

The Terraform configuration creates:

- Pub/Sub topics and subscriptions
- BigQuery datasets and tables
- Cloud Storage buckets
- IAM roles and service accounts
- Monitoring alert policies
- Cloud Functions for alerting

## 🤖 Machine Learning Models

### Anomaly Detection Approaches

1. **Statistical Methods**: Z-score and threshold-based detection
2. **Isolation Forest**: Unsupervised anomaly detection
3. **One-Class SVM**: Support vector machine for outlier detection
4. **LSTM Autoencoder**: Deep learning for time series anomalies

### Model Features

- Temperature sensor readings
- Vibration sensor readings
- Statistical features (z-scores, moving averages)
- Time-based features

## 📊 Monitoring and Alerting

### Dashboards

- Real-time sensor data visualization
- Anomaly detection results
- System performance metrics
- Model performance tracking

### Alerting

- Threshold-based alerts for sensor values
- ML model-based anomaly alerts
- System health monitoring
- Email/SMS notifications via Cloud Functions

## 🔧 Development

### Local Development

```bash
# Start all services
docker-compose up

# Start specific service
docker-compose up iot-simulator

# View logs
docker-compose logs -f iot-simulator

# Stop services
docker-compose down
```

### Testing

```bash
# Run tests in container
docker-compose exec ml-trainer python -m pytest

# Run specific test
docker-compose exec ml-trainer python -m pytest tests/test_models.py
```

### Code Quality

```bash
# Format code
docker-compose exec ml-trainer black src/

# Lint code
docker-compose exec ml-trainer flake8 src/

# Type checking
docker-compose exec ml-trainer mypy src/
```

## 🚀 Deployment

### Production Deployment

1. **Infrastructure**: Deploy with Terraform
2. **Dataflow Pipeline**: Deploy using provided scripts
3. **ML Models**: Deploy to Vertex AI
4. **Monitoring**: Configure alerts and dashboards

### CI/CD Pipeline

The system supports automated deployment through:

- GitHub Actions / Cloud Build
- Automated testing
- Infrastructure validation
- Model deployment

## 📈 Scaling

### Horizontal Scaling

- Dataflow auto-scaling based on message volume
- Multiple IoT simulator instances
- Load-balanced ML serving endpoints

### Performance Optimization

- BigQuery partitioning and clustering
- Pub/Sub message batching
- Model serving optimization
- Caching strategies

## 🔒 Security

### Best Practices

- Service account with minimal permissions
- VPC network isolation
- Encrypted data in transit and at rest
- Secret management with Google Secret Manager

### Access Control

- IAM roles for different components
- API authentication and authorization
- Network security policies

## 🐛 Troubleshooting

### Common Issues

1. **Docker Build Fails**: Check Docker daemon and disk space
2. **Terraform Apply Fails**: Verify GCP permissions and quotas
3. **Pub/Sub Connection Issues**: Check service account credentials
4. **BigQuery Access Denied**: Verify IAM roles

### Debugging

```bash
# Check container logs
docker-compose logs service-name

# Access container shell
docker-compose exec service-name bash

# Check Terraform state
terraform show

# Validate Terraform config
terraform validate
```

## 📚 Documentation

- [Architecture Design](docs/architecture.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)
- [Troubleshooting Guide](docs/troubleshooting.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the troubleshooting guide
- Review the documentation

## 🔄 Version History

- **v1.0.0**: Initial release with temperature and vibration detection
- **v1.1.0**: Added Docker containerization
- **v1.2.0**: Terraform infrastructure automation
- **v1.3.0**: Enhanced monitoring and alerting