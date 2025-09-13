#!/bin/bash

set -e

echo "🚀 OBS Live Translator Deployment Script"
echo "========================================"

# Configuration
DOCKER_IMAGE="obs-live-translator"
DOCKER_TAG="${1:-latest}"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found. Please install Docker first."
        exit 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found. Please install Docker Compose first."
        exit 1
    fi

    # Check NVIDIA Docker runtime
    if ! docker info | grep -q nvidia; then
        log_warn "NVIDIA Docker runtime not found. GPU acceleration may not work."
    fi

    log_info "All requirements met ✓"
}

build_image() {
    log_info "Building Docker image..."
    docker build -t ${DOCKER_IMAGE}:${DOCKER_TAG} .

    if [ $? -eq 0 ]; then
        log_info "Docker image built successfully ✓"
    else
        log_error "Failed to build Docker image"
        exit 1
    fi
}

create_env_file() {
    if [ ! -f ${ENV_FILE} ]; then
        log_info "Creating environment file..."
        cat > ${ENV_FILE} <<EOL
# OBS Live Translator Environment Variables
RUST_LOG=info
OBS_TRANSLATOR_PORT=8080
MONITORING_PORT=8081
MAX_CONCURRENT_STREAMS=20
CACHE_SIZE_MB=2048
ENABLE_GPU_ACCELERATION=true
ENABLE_CACHE_WARMING=true
ENABLE_PERFORMANCE_MONITORING=true
NVIDIA_VISIBLE_DEVICES=all
EOL
        log_info "Environment file created ✓"
    else
        log_info "Using existing environment file"
    fi
}

start_services() {
    log_info "Starting services..."
    docker-compose -f ${COMPOSE_FILE} up -d

    if [ $? -eq 0 ]; then
        log_info "Services started successfully ✓"
    else
        log_error "Failed to start services"
        exit 1
    fi
}

wait_for_health() {
    log_info "Waiting for services to be healthy..."

    local max_attempts=30
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        if curl -f http://localhost:8080/health &> /dev/null; then
            log_info "Services are healthy ✓"
            return 0
        fi

        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done

    log_error "Services failed to become healthy"
    return 1
}

show_status() {
    log_info "Service Status:"
    docker-compose ps

    echo ""
    log_info "Access URLs:"
    echo "  • Main API:        http://localhost:8080"
    echo "  • Health Check:    http://localhost:8080/health"
    echo "  • Metrics:         http://localhost:8081/metrics"
    echo "  • Grafana:         http://localhost:3000"
    echo "  • Prometheus:      http://localhost:9091"
}

# Main execution
main() {
    check_requirements
    create_env_file
    build_image
    start_services
    wait_for_health
    show_status

    log_info "Deployment completed successfully! 🎉"
}

# Handle different commands
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        log_info "Stopping services..."
        docker-compose down
        ;;
    restart)
        log_info "Restarting services..."
        docker-compose restart
        ;;
    logs)
        docker-compose logs -f translator
        ;;
    status)
        show_status
        ;;
    clean)
        log_warn "Cleaning up all containers and volumes..."
        docker-compose down -v
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|logs|status|clean}"
        exit 1
        ;;
esac