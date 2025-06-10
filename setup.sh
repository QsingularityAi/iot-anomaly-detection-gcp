#!/bin/bash

# IoT Anomaly Detection System Setup Script
# This script helps set up the development environment

set -e

echo "🚀 Setting up IoT Anomaly Detection System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Terraform is installed
if ! command -v terraform &> /dev/null; then
    print_error "Terraform is not installed. Please install Terraform first."
    exit 1
fi

print_status "Docker and Terraform are installed ✓"

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    print_status "Creating .env file from template..."
    cp .env.template .env
    print_warning "Please edit .env file with your configuration before proceeding"
else
    print_status ".env file already exists ✓"
fi

# Create terraform.tfvars file from template if it doesn't exist
if [ ! -f gcp-config/terraform/terraform.tfvars ]; then
    print_status "Creating terraform.tfvars file from template..."
    cp gcp-config/terraform/terraform.tfvars.template gcp-config/terraform/terraform.tfvars
    print_warning "Please edit gcp-config/terraform/terraform.tfvars with your configuration"
else
    print_status "terraform.tfvars file already exists ✓"
fi

# Initialize Terraform
print_status "Initializing Terraform..."
cd gcp-config/terraform
terraform init
cd ../..

print_status "Validating Terraform configuration..."
cd gcp-config/terraform
terraform validate
cd ../..

# Check Docker daemon
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running. Please start Docker first."
    exit 1
fi

print_status "Docker daemon is running ✓"

# Build Docker images (optional)
read -p "Do you want to build Docker images now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Building Docker development image..."
    docker build -f gcp-config/Dockerfile --target development -t iot-anomaly-dev .
    print_status "Docker image built successfully ✓"
fi

print_status "Setup completed! 🎉"
echo
echo "Next steps:"
echo "1. Edit .env file with your GCP project configuration"
echo "2. Edit gcp-config/terraform/terraform.tfvars with your Terraform variables"
echo "3. Set up Google Cloud credentials"
echo "4. Run 'terraform plan' in gcp-config/terraform/ to review infrastructure changes"
echo "5. Run 'terraform apply' to create GCP resources"
echo "6. Use 'docker-compose up' in gcp-config/ to start local development environment"
echo
echo "For more information, check the README.md file."