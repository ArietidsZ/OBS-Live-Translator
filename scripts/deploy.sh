#!/bin/bash

# OBS Live Translator Deployment Script
# Automated deployment for production environments

set -euo pipefail

# Configuration
APP_NAME="obs-live-translator"
DEPLOY_USER="obs-translator"
DEPLOY_GROUP="obs-translator"
INSTALL_DIR="/opt/${APP_NAME}"
CONFIG_DIR="/etc/${APP_NAME}"
DATA_DIR="/var/lib/${APP_NAME}"
LOG_DIR="/var/log/${APP_NAME}"
MODELS_DIR="${INSTALL_DIR}/models"

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
    exit 1
}

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root"
    fi
}

# Detect OS
detect_os() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        VER=$VERSION_ID
    else
        log_error "Cannot detect OS"
    fi
    
    log_info "Detected OS: $OS $VER"
}

# Install dependencies
install_dependencies() {
    log_info "Installing dependencies..."
    
    case $OS in
        ubuntu|debian)
            apt-get update
            apt-get install -y \
                curl \
                wget \
                tar \
                gzip \
                libssl-dev \
                ca-certificates
            ;;
        centos|rhel|fedora)
            yum install -y \
                curl \
                wget \
                tar \
                gzip \
                openssl-devel \
                ca-certificates
            ;;
        *)
            log_warn "Unknown OS, skipping dependency installation"
            ;;
    esac
}

# Create user and directories
setup_environment() {
    log_info "Setting up environment..."
    
    # Create user if not exists
    if ! id "$DEPLOY_USER" &>/dev/null; then
        useradd -r -s /bin/false "$DEPLOY_USER"
        log_info "Created user: $DEPLOY_USER"
    fi
    
    # Create directories
    mkdir -p "$INSTALL_DIR" "$CONFIG_DIR" "$DATA_DIR" "$LOG_DIR" "$MODELS_DIR"
    
    # Set permissions
    chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$INSTALL_DIR" "$DATA_DIR" "$LOG_DIR"
    chmod 755 "$INSTALL_DIR" "$CONFIG_DIR"
    chmod 750 "$DATA_DIR" "$LOG_DIR"
}

# Download and install binary
install_binary() {
    local VERSION=${1:-"latest"}
    log_info "Installing $APP_NAME version: $VERSION"
    
    # Detect architecture
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            ARCH_SUFFIX="x64"
            ;;
        aarch64|arm64)
            ARCH_SUFFIX="arm64"
            ;;
        *)
            log_error "Unsupported architecture: $ARCH"
            ;;
    esac
    
    # Download binary
    DOWNLOAD_URL="https://github.com/yourusername/$APP_NAME/releases/$VERSION/download/${APP_NAME}-linux-${ARCH_SUFFIX}.tar.gz"
    log_info "Downloading from: $DOWNLOAD_URL"
    
    wget -q --show-progress -O "/tmp/${APP_NAME}.tar.gz" "$DOWNLOAD_URL" || \
        log_error "Failed to download binary"
    
    # Extract and install
    tar -xzf "/tmp/${APP_NAME}.tar.gz" -C "$INSTALL_DIR"
    chmod +x "${INSTALL_DIR}/${APP_NAME}"
    
    # Create symlink
    ln -sf "${INSTALL_DIR}/${APP_NAME}" "/usr/local/bin/${APP_NAME}"
    
    log_info "Binary installed successfully"
}

# Download models
download_models() {
    local PROFILE=${1:-"medium"}
    log_info "Downloading models for profile: $PROFILE"
    
    # Model URLs based on profile
    declare -A MODELS
    
    case $PROFILE in
        low)
            MODELS["whisper"]="https://huggingface.co/models/whisper_tiny.onnx"
            MODELS["translation"]="https://huggingface.co/models/m2m_small.onnx"
            ;;
        medium)
            MODELS["whisper"]="https://huggingface.co/models/whisper_small.onnx"
            MODELS["translation"]="https://huggingface.co/models/nllb_medium.onnx"
            ;;
        high)
            MODELS["whisper"]="https://huggingface.co/models/parakeet.onnx"
            MODELS["translation"]="https://huggingface.co/models/nllb_large.onnx"
            ;;
        *)
            log_error "Invalid profile: $PROFILE"
            ;;
    esac
    
    # Download each model
    for model_name in "${!MODELS[@]}"; do
        log_info "Downloading $model_name model..."
        wget -q --show-progress -O "${MODELS_DIR}/${model_name}.onnx" "${MODELS[$model_name]}" || \
            log_warn "Failed to download $model_name model"
    done
    
    # Set permissions
    chown -R "$DEPLOY_USER:$DEPLOY_GROUP" "$MODELS_DIR"
    chmod 644 "${MODELS_DIR}"/*.onnx
    
    log_info "Models downloaded successfully"
}

# Generate configuration
generate_config() {
    local PROFILE=${1:-"medium"}
    log_info "Generating configuration for profile: $PROFILE"
    
    cat > "${CONFIG_DIR}/config.toml" <<EOF
[general]
profile = "$PROFILE"
log_level = "info"
data_dir = "$DATA_DIR"

[audio]
sample_rate = 16000
frame_size_ms = 30.0
enable_vad = true
enable_noise_reduction = true
enable_agc = true

[asr]
model_type = "whisper"
model_path = "${MODELS_DIR}/whisper.onnx"
device = "cpu"
batch_size = 1

[translation]
model_type = "nllb"
model_path = "${MODELS_DIR}/translation.onnx"
source_language = "auto"
target_language = "en"

[streaming]
host = "0.0.0.0"
port = 8080
buffer_size = 8192
max_connections = 100

[monitoring]
enable = true
metrics_port = 9090
health_check_port = 8081
EOF
    
    chmod 644 "${CONFIG_DIR}/config.toml"
    log_info "Configuration generated"
}

# Setup systemd service
setup_systemd() {
    log_info "Setting up systemd service..."
    
    cat > "/etc/systemd/system/${APP_NAME}.service" <<EOF
[Unit]
Description=OBS Live Translator
After=network.target
Requires=network.target

[Service]
Type=simple
User=$DEPLOY_USER
Group=$DEPLOY_GROUP
WorkingDirectory=$INSTALL_DIR
ExecStart=${INSTALL_DIR}/${APP_NAME} --config ${CONFIG_DIR}/config.toml
ExecReload=/bin/kill -HUP \$MAINPID
KillMode=mixed
KillSignal=SIGTERM
Restart=always
RestartSec=10
StandardOutput=append:${LOG_DIR}/stdout.log
StandardError=append:${LOG_DIR}/stderr.log

# Security
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=${DATA_DIR} ${LOG_DIR}

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    log_info "Systemd service configured"
}

# Setup log rotation
setup_logrotate() {
    log_info "Setting up log rotation..."
    
    cat > "/etc/logrotate.d/${APP_NAME}" <<EOF
${LOG_DIR}/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 $DEPLOY_USER $DEPLOY_GROUP
    sharedscripts
    postrotate
        systemctl reload $APP_NAME 2>/dev/null || true
    endscript
}
EOF
    
    chmod 644 "/etc/logrotate.d/${APP_NAME}"
    log_info "Log rotation configured"
}

# Configure firewall
setup_firewall() {
    log_info "Configuring firewall..."
    
    if command -v ufw &>/dev/null; then
        ufw allow 8080/tcp comment "OBS Live Translator WebSocket"
        ufw allow 8081/tcp comment "OBS Live Translator Health Check"
        log_info "UFW rules added"
    elif command -v firewall-cmd &>/dev/null; then
        firewall-cmd --permanent --add-port=8080/tcp
        firewall-cmd --permanent --add-port=8081/tcp
        firewall-cmd --reload
        log_info "Firewalld rules added"
    else
        log_warn "No firewall detected, skipping firewall configuration"
    fi
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    # Check binary
    if ! command -v "$APP_NAME" &>/dev/null; then
        log_error "Binary not found in PATH"
    fi
    
    # Check version
    VERSION=$($APP_NAME --version 2>/dev/null || echo "unknown")
    log_info "Installed version: $VERSION"
    
    # Verify models
    if ! $APP_NAME verify-models &>/dev/null; then
        log_warn "Model verification failed"
    fi
    
    # Test configuration
    if ! $APP_NAME --config "${CONFIG_DIR}/config.toml" --dry-run &>/dev/null; then
        log_warn "Configuration test failed"
    fi
    
    log_info "Installation verified"
}

# Start service
start_service() {
    log_info "Starting service..."
    
    systemctl enable "$APP_NAME"
    systemctl start "$APP_NAME"
    
    sleep 3
    
    if systemctl is-active --quiet "$APP_NAME"; then
        log_info "Service started successfully"
    else
        log_error "Failed to start service"
    fi
    
    # Check health endpoint
    if curl -s -f "http://localhost:8081/health" &>/dev/null; then
        log_info "Health check passed"
    else
        log_warn "Health check failed"
    fi
}

# Uninstall function
uninstall() {
    log_info "Uninstalling $APP_NAME..."
    
    # Stop service
    systemctl stop "$APP_NAME" 2>/dev/null || true
    systemctl disable "$APP_NAME" 2>/dev/null || true
    
    # Remove files
    rm -f "/etc/systemd/system/${APP_NAME}.service"
    rm -f "/etc/logrotate.d/${APP_NAME}"
    rm -f "/usr/local/bin/${APP_NAME}"
    rm -rf "$INSTALL_DIR"
    rm -rf "$CONFIG_DIR"
    rm -rf "$DATA_DIR"
    rm -rf "$LOG_DIR"
    
    # Remove user
    userdel -r "$DEPLOY_USER" 2>/dev/null || true
    
    systemctl daemon-reload
    
    log_info "Uninstallation complete"
}

# Main function
main() {
    local ACTION=${1:-"install"}
    local PROFILE=${2:-"medium"}
    local VERSION=${3:-"latest"}
    
    case $ACTION in
        install)
            check_root
            detect_os
            install_dependencies
            setup_environment
            install_binary "$VERSION"
            download_models "$PROFILE"
            generate_config "$PROFILE"
            setup_systemd
            setup_logrotate
            setup_firewall
            verify_installation
            start_service
            
            log_info "Installation complete!"
            log_info "Service status: systemctl status $APP_NAME"
            log_info "Logs: journalctl -u $APP_NAME -f"
            log_info "WebSocket endpoint: ws://localhost:8080"
            log_info "Health check: http://localhost:8081/health"
            log_info "Metrics: http://localhost:9090/metrics"
            ;;
        
        uninstall)
            check_root
            uninstall
            ;;
        
        upgrade)
            check_root
            log_info "Upgrading $APP_NAME..."
            systemctl stop "$APP_NAME"
            install_binary "$VERSION"
            systemctl start "$APP_NAME"
            log_info "Upgrade complete"
            ;;
        
        *)
            echo "Usage: $0 {install|uninstall|upgrade} [profile] [version]"
            echo "  Profiles: low, medium, high"
            echo "  Version: latest or specific version (e.g., v1.0.0)"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"