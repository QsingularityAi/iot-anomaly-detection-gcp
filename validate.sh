#!/bin/bash

# Validation script for IoT Anomaly Detection System
# This script validates the setup and configuration

set -e

echo "🔍 Validating IoT Anomaly Detection System setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Docker
if command -v docker &> /dev/null; then
    if docker info &> /dev/null; then
        print_status "Docker is installed and running"
    else
        print_error "Docker is installed but not running"
        exit 1
    fi
else
    print_error "Docker is not installed"
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    print_status "Docker Compose is available"
else
    print_warning "Docker Compose not found, using 'docker compose' plugin"
fi

# Check Terraform
if command -v terraform &> /dev/null; then
    TERRAFORM_VERSION=$(terraform version -json | grep -o '"terraform_version":"[^"]*' | cut -d'"' -f4)
    print_status "Terraform is installed (version: $TERRAFORM_VERSION)"
else
    print_error "Terraform is not installed"
    exit 1
fi

# Check required files
echo
echo "📁 Checking required files..."

required_files=(
    "requirements.txt"
    "gcp-config/Dockerfile"
    "gcp-config/docker-compose.yml"
    "gcp-config/terraform/main.tf"
    ".env.template"
    "gcp-config/terraform/terraform.tfvars.template"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file exists"
    else
        print_error "$file is missing"
        exit 1
    fi
done

# Check source files
echo
echo "🐍 Checking Python source files..."

src_files=(
    "src/iot_device_simulator_temp_vibration.py"
    "src/data_consumer.py"
    "src/dataflow_pipeline_temp_vibration.py"
    "src/custom_ml_models_temp_vibration.py"
    "src/model_evaluation.py"
)

for file in "${src_files[@]}"; do
    if [ -f "$file" ]; then
        print_status "$file exists"
    else
        print_error "$file is missing"
        exit 1
    fi
done

# Validate Terraform configuration
echo
echo "🏗️ Validating Terraform configuration..."

cd gcp-config/terraform
if terraform validate &> /dev/null; then
    print_status "Terraform configuration is valid"
else
    print_error "Terraform configuration is invalid"
    terraform validate
    exit 1
fi
cd ../..

# Check Docker build context
echo
echo "🐳 Checking Docker build context..."

if [ -f ".dockerignore" ]; then
    print_status ".dockerignore file exists"
else
    print_warning ".dockerignore file is missing (recommended)"
fi

# Validate docker-compose.yml
echo
echo "📦 Validating Docker Compose configuration..."

cd gcp-config
if docker-compose config &> /dev/null; then
    print_status "Docker Compose configuration is valid"
else
    print_warning "Docker Compose validation failed (may need environment variables)"
fi
cd ..

# Check Python syntax
echo
echo "🐍 Checking Python syntax..."

python_files=$(find src/ -name "*.py" 2>/dev/null || true)
if [ -n "$python_files" ]; then
    for file in $python_files; do
        if python3 -m py_compile "$file" 2>/dev/null; then
            print_status "$file syntax is valid"
        else
            print_error "$file has syntax errors"
            exit 1
        fi
    done
else
    print_warning "No Python files found in src/"
fi

echo
print_status "All validations passed! 🎉"
echo
echo "Next steps:"
echo "1. Configure .env file with your GCP project settings"
echo "2. Configure gcp-config/terraform/terraform.tfvars"
echo "3. Set up Google Cloud credentials"
echo "4. Run 'terraform plan' to review infrastructure"
echo "5. Run 'docker-compose up' to start development environment"