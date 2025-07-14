#!/bin/bash

# AI Vehicle Monitoring System - AWS EC2 Deployment Script
# This script deploys the system to AWS EC2 as specified in the guidelines

set -e  # Exit on any error

# Configuration
APP_NAME="ai-vehicle-system"
DOCKER_IMAGE="$APP_NAME:latest"
EC2_USER="ubuntu"
EC2_HOST=""  # Set this to your EC2 instance IP/hostname
KEY_PATH=""  # Set this to your SSH key path

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    # Check if AWS CLI is installed
    if ! command -v aws &> /dev/null; then
        warn "AWS CLI is not installed. Some features may not work."
    fi
    
    # Check if SSH key exists
    if [[ -n "$KEY_PATH" && ! -f "$KEY_PATH" ]]; then
        error "SSH key not found at $KEY_PATH"
    fi
    
    # Check if EC2 host is set
    if [[ -z "$EC2_HOST" ]]; then
        error "EC2_HOST is not set. Please configure your EC2 instance details."
    fi
    
    log "Prerequisites check completed"
}

# Build Docker image
build_image() {
    log "Building Docker image..."
    
    # Build the production image
    docker build -t $DOCKER_IMAGE --target production .
    
    if [[ $? -eq 0 ]]; then
        log "Docker image built successfully"
    else
        error "Failed to build Docker image"
    fi
}

# Test image locally
test_image() {
    log "Testing Docker image locally..."
    
    # Run a quick test
    docker run --rm $DOCKER_IMAGE python -c "
import sys
sys.path.append('/app')
from src.utils.config import get_settings
print('Configuration loaded successfully')
"
    
    if [[ $? -eq 0 ]]; then
        log "Docker image test passed"
    else
        error "Docker image test failed"
    fi
}

# Deploy to EC2
deploy_to_ec2() {
    log "Deploying to EC2 instance: $EC2_HOST"
    
    # Create deployment directory on EC2
    ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "mkdir -p ~/ai-vehicle-system"
    
    # Copy docker-compose and configuration files
    scp -i "$KEY_PATH" docker-compose.yml $EC2_USER@$EC2_HOST:~/ai-vehicle-system/
    scp -i "$KEY_PATH" .env.example $EC2_USER@$EC2_HOST:~/ai-vehicle-system/.env
    
    # Copy deployment scripts
    scp -i "$KEY_PATH" deployment/ec2-setup.sh $EC2_USER@$EC2_HOST:~/ai-vehicle-system/
    
    # Save and transfer Docker image
    log "Transferring Docker image..."
    docker save $DOCKER_IMAGE | gzip | ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "
        cd ~/ai-vehicle-system && 
        gunzip | docker load
    "
    
    # Run setup script on EC2
    ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "
        cd ~/ai-vehicle-system && 
        chmod +x ec2-setup.sh && 
        ./ec2-setup.sh
    "
    
    log "Deployment to EC2 completed"
}

# Setup monitoring
setup_monitoring() {
    log "Setting up monitoring..."
    
    ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "
        cd ~/ai-vehicle-system &&
        docker-compose up -d prometheus grafana
    "
    
    log "Monitoring setup completed"
    log "Grafana available at: http://$EC2_HOST:3000 (admin/admin)"
    log "Prometheus available at: http://$EC2_HOST:9090"
}

# Health check
health_check() {
    log "Performing health check..."
    
    # Wait for service to start
    sleep 30
    
    # Check if API is responding
    response=$(ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health/")
    
    if [[ "$response" == "200" ]]; then
        log "Health check passed - API is responding"
    else
        warn "Health check failed - API returned status: $response"
    fi
    
    # Check Docker containers
    ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "docker-compose ps"
}

# Rollback function
rollback() {
    log "Rolling back deployment..."
    
    ssh -i "$KEY_PATH" $EC2_USER@$EC2_HOST "
        cd ~/ai-vehicle-system &&
        docker-compose down &&
        docker-compose up -d
    "
    
    log "Rollback completed"
}

# Main deployment function
main() {
    log "Starting deployment of AI Vehicle Monitoring System"
    
    case "${1:-deploy}" in
        "build")
            check_prerequisites
            build_image
            test_image
            ;;
        "deploy")
            check_prerequisites
            build_image
            test_image
            deploy_to_ec2
            setup_monitoring
            health_check
            ;;
        "monitor")
            setup_monitoring
            ;;
        "health")
            health_check
            ;;
        "rollback")
            rollback
            ;;
        *)
            echo "Usage: $0 {build|deploy|monitor|health|rollback}"
            echo ""
            echo "Commands:"
            echo "  build    - Build Docker image only"
            echo "  deploy   - Full deployment (default)"
            echo "  monitor  - Setup monitoring only"
            echo "  health   - Run health check"
            echo "  rollback - Rollback deployment"
            exit 1
            ;;
    esac
    
    log "Deployment script completed successfully"
}

# Configuration validation
if [[ -z "$EC2_HOST" ]]; then
    echo "Please configure EC2_HOST in this script before running"
    echo "Example: EC2_HOST=\"ec2-xxx-xxx-xxx-xxx.compute-1.amazonaws.com\""
    exit 1
fi

if [[ -z "$KEY_PATH" ]]; then
    echo "Please configure KEY_PATH in this script before running"
    echo "Example: KEY_PATH=\"~/.ssh/my-ec2-key.pem\""
    exit 1
fi

# Run main function
main "$@"
