#!/bin/bash

# EC2 Instance Setup Script for AI Vehicle Monitoring System
# This script sets up the EC2 instance with required dependencies

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

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

# Update system packages
update_system() {
    log "Updating system packages..."
    sudo apt-get update -y
    sudo apt-get upgrade -y
}

# Install Docker
install_docker() {
    log "Installing Docker..."
    
    # Remove old versions
    sudo apt-get remove -y docker docker-engine docker.io containerd runc || true
    
    # Install dependencies
    sudo apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg \
        lsb-release
    
    # Add Docker's official GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Set up stable repository
    echo \
        "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker Engine
    sudo apt-get update -y
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io
    
    # Add user to docker group
    sudo usermod -aG docker $USER
    
    # Start and enable Docker
    sudo systemctl start docker
    sudo systemctl enable docker
    
    log "Docker installed successfully"
}

# Install Docker Compose
install_docker_compose() {
    log "Installing Docker Compose..."
    
    # Download and install Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # Create symlink
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    log "Docker Compose installed successfully"
}

# Install monitoring tools
install_monitoring() {
    log "Installing monitoring tools..."
    
    # Install htop, iotop, and other monitoring utilities
    sudo apt-get install -y htop iotop nethogs ncdu tree
    
    log "Monitoring tools installed"
}

# Configure firewall
configure_firewall() {
    log "Configuring firewall..."
    
    # Install ufw if not present
    sudo apt-get install -y ufw
    
    # Reset firewall rules
    sudo ufw --force reset
    
    # Default policies
    sudo ufw default deny incoming
    sudo ufw default allow outgoing
    
    # Allow SSH
    sudo ufw allow ssh
    
    # Allow application ports
    sudo ufw allow 8000/tcp  # API
    sudo ufw allow 3000/tcp  # Grafana
    sudo ufw allow 9090/tcp  # Prometheus
    
    # Enable firewall
    sudo ufw --force enable
    
    log "Firewall configured"
}

# Setup application directories
setup_directories() {
    log "Setting up application directories..."
    
    # Create data directories
    mkdir -p ~/ai-vehicle-system/data/faiss_index
    mkdir -p ~/ai-vehicle-system/data/chromadb
    mkdir -p ~/ai-vehicle-system/logs
    
    # Set permissions
    chmod 755 ~/ai-vehicle-system/data
    chmod 755 ~/ai-vehicle-system/logs
    
    log "Directories created"
}

# Configure environment
configure_environment() {
    log "Configuring environment..."
    
    # Check if .env exists, if not copy from example
    if [[ ! -f ~/ai-vehicle-system/.env ]]; then
        cp ~/ai-vehicle-system/.env.example ~/ai-vehicle-system/.env
        warn "Please edit ~/ai-vehicle-system/.env with your configuration"
    fi
    
    # Set up log rotation
    sudo tee /etc/logrotate.d/ai-vehicle-system > /dev/null <<EOF
~/ai-vehicle-system/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 ubuntu ubuntu
}
EOF
    
    log "Environment configured"
}

# Start services
start_services() {
    log "Starting services..."
    
    cd ~/ai-vehicle-system
    
    # Pull any missing images
    docker-compose pull
    
    # Start services
    docker-compose up -d
    
    # Wait for services to start
    sleep 10
    
    # Show status
    docker-compose ps
    
    log "Services started"
}

# Setup system monitoring
setup_system_monitoring() {
    log "Setting up system monitoring..."
    
    # Create monitoring script
    cat > ~/monitor-system.sh << 'EOF'
#!/bin/bash
# System monitoring script

echo "=== System Status $(date) ==="
echo "Uptime: $(uptime)"
echo "Memory Usage:"
free -h
echo "Disk Usage:"
df -h
echo "Docker Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo "Recent Logs:"
docker-compose logs --tail=10
EOF
    
    chmod +x ~/monitor-system.sh
    
    # Add to crontab for regular monitoring
    (crontab -l 2>/dev/null; echo "*/15 * * * * ~/monitor-system.sh >> ~/system-monitor.log 2>&1") | crontab -
    
    log "System monitoring configured"
}

# Main setup function
main() {
    log "Starting EC2 setup for AI Vehicle Monitoring System"
    
    update_system
    install_docker
    install_docker_compose
    install_monitoring
    configure_firewall
    setup_directories
    configure_environment
    start_services
    setup_system_monitoring
    
    log "EC2 setup completed successfully!"
    log ""
    log "Next steps:"
    log "1. Edit ~/ai-vehicle-system/.env with your configuration"
    log "2. Restart services: cd ~/ai-vehicle-system && docker-compose restart"
    log "3. Check logs: docker-compose logs -f"
    log "4. Access API: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):8000"
    log "5. Access Grafana: http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):3000"
}

# Run main function
main "$@"
