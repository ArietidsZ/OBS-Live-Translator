# Deployment Guide

## System Requirements

### Minimum Requirements (Low Profile)
- **CPU**: Dual-core x86_64 or ARM64 processor
- **RAM**: 2GB
- **Storage**: 500MB for binaries and models
- **OS**: Linux (Ubuntu 20.04+), macOS 11+, Windows 10+

### Recommended Requirements (Medium Profile)
- **CPU**: Quad-core processor with AVX2 support
- **RAM**: 4GB
- **GPU**: Optional (NVIDIA with CUDA 11.8+)
- **Storage**: 2GB for models and cache
- **OS**: Linux (Ubuntu 22.04), macOS 12+, Windows 11

### High Performance (High Profile)
- **CPU**: 8+ cores with AVX-512 support
- **RAM**: 8GB+
- **GPU**: NVIDIA RTX 3060+ with CUDA 12.0+
- **Storage**: 5GB for all models
- **OS**: Linux with real-time kernel preferred

## Installation

### Binary Installation

#### Linux
```bash
# Download latest release
wget https://github.com/yourusername/obs-live-translator/releases/latest/download/obs-live-translator-linux-x64.tar.gz

# Extract
tar -xzf obs-live-translator-linux-x64.tar.gz

# Install
sudo ./install.sh

# Verify installation
obs-live-translator --version
```

#### macOS
```bash
# Using Homebrew
brew tap yourusername/obs-live-translator
brew install obs-live-translator

# Or download directly
curl -L https://github.com/yourusername/obs-live-translator/releases/latest/download/obs-live-translator-macos.tar.gz | tar -xz
sudo ./install.sh
```

#### Windows
```powershell
# Download installer
Invoke-WebRequest -Uri "https://github.com/yourusername/obs-live-translator/releases/latest/download/obs-live-translator-setup.exe" -OutFile "obs-live-translator-setup.exe"

# Run installer
.\obs-live-translator-setup.exe

# Or use Chocolatey
choco install obs-live-translator
```

### Docker Deployment

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libssl-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY obs-live-translator /usr/local/bin/
COPY models /opt/models

ENV OBS_MODELS_PATH=/opt/models
ENV RUST_LOG=info

EXPOSE 8080

CMD ["obs-live-translator", "--config", "/config/config.toml"]
```

```bash
# Build and run
docker build -t obs-live-translator .
docker run -p 8080:8080 -v ./config:/config obs-live-translator
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: obs-live-translator
spec:
  replicas: 3
  selector:
    matchLabels:
      app: obs-live-translator
  template:
    metadata:
      labels:
        app: obs-live-translator
    spec:
      containers:
      - name: translator
        image: obs-live-translator:latest
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "2Gi"
            cpu: "2"
            nvidia.com/gpu: "1"  # For GPU nodes
          limits:
            memory: "4Gi"
            cpu: "4"
        env:
        - name: PROFILE
          value: "medium"
        - name: RUST_LOG
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /config
        - name: models
          mountPath: /opt/models
      volumes:
      - name: config
        configMap:
          name: translator-config
      - name: models
        persistentVolumeClaim:
          claimName: models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: obs-live-translator
spec:
  selector:
    app: obs-live-translator
  ports:
  - port: 8080
    targetPort: 8080
  type: LoadBalancer
```

## Configuration

### Configuration File (config.toml)

```toml
[general]
profile = "medium"
log_level = "info"
data_dir = "/var/lib/obs-translator"

[audio]
sample_rate = 16000
frame_size_ms = 30.0
enable_vad = true
enable_noise_reduction = true

[asr]
model_type = "whisper_small"
model_path = "/opt/models/whisper_small.onnx"
device = "cuda"  # or "cpu"
batch_size = 4

[translation]
model_type = "nllb"
model_path = "/opt/models/nllb.onnx"
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
```

### Environment Variables

```bash
# Core settings
export OBS_PROFILE=medium
export OBS_LOG_LEVEL=info
export OBS_DATA_DIR=/var/lib/obs-translator

# Model paths
export OBS_MODELS_PATH=/opt/models
export OBS_WHISPER_MODEL=/opt/models/whisper_small.onnx
export OBS_TRANSLATION_MODEL=/opt/models/nllb.onnx

# Performance
export OBS_THREADS=4
export OBS_BATCH_SIZE=4
export OBS_USE_GPU=true

# Networking
export OBS_HOST=0.0.0.0
export OBS_PORT=8080
export OBS_MAX_CONNECTIONS=100
```

## Model Management

### Automatic Model Download

```bash
# Download all required models
obs-live-translator download-models --profile medium

# Download specific model
obs-live-translator download-models --model whisper_small

# Verify models
obs-live-translator verify-models
```

### Manual Model Setup

```bash
# Create models directory
mkdir -p /opt/models

# Download models
wget https://huggingface.co/models/whisper_small.onnx -O /opt/models/whisper_small.onnx
wget https://huggingface.co/models/nllb.onnx -O /opt/models/nllb.onnx

# Set permissions
chmod 644 /opt/models/*.onnx
```

## Service Management

### Systemd Service (Linux)

```ini
# /etc/systemd/system/obs-live-translator.service
[Unit]
Description=OBS Live Translator
After=network.target

[Service]
Type=simple
User=obs-translator
Group=obs-translator
WorkingDirectory=/opt/obs-translator
ExecStart=/usr/local/bin/obs-live-translator --config /etc/obs-translator/config.toml
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

# Performance tuning
CPUAffinity=0-3
Nice=-10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable obs-live-translator
sudo systemctl start obs-live-translator
sudo systemctl status obs-live-translator
```

### Health Checks

```bash
# HTTP health endpoint
curl http://localhost:8081/health

# Metrics endpoint (Prometheus format)
curl http://localhost:9090/metrics

# WebSocket test
wscat -c ws://localhost:8080
```

## Performance Tuning

### Linux Kernel Parameters

```bash
# /etc/sysctl.d/99-obs-translator.conf
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
vm.swappiness = 10
```

### CPU Affinity

```bash
# Pin to specific CPU cores
taskset -c 0-3 obs-live-translator

# Or use numactl for NUMA systems
numactl --cpubind=0 --membind=0 obs-live-translator
```

### GPU Configuration

```bash
# NVIDIA GPU
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0
export CUDNN_BENCHMARK=1

# Monitor GPU usage
nvidia-smi dmon -i 0
```

## Monitoring

### Prometheus Integration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'obs-translator'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 10s
```

### Grafana Dashboard

Import the provided dashboard from `monitoring/grafana-dashboard.json`:

1. Open Grafana
2. Go to Dashboards → Import
3. Upload the JSON file
4. Select Prometheus data source

### Logging

```bash
# Configure logging
export RUST_LOG=obs_live_translator=debug,info

# Log to file
obs-live-translator 2>&1 | tee /var/log/obs-translator.log

# Log rotation
# /etc/logrotate.d/obs-translator
/var/log/obs-translator.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 644 obs-translator obs-translator
    postrotate
        systemctl reload obs-live-translator
    endscript
}
```

## Security

### TLS/SSL Configuration

```toml
# config.toml
[tls]
enable = true
cert_path = "/etc/ssl/certs/obs-translator.crt"
key_path = "/etc/ssl/private/obs-translator.key"
ca_path = "/etc/ssl/certs/ca-bundle.crt"
```

### Authentication

```toml
[auth]
enable = true
type = "jwt"  # or "api_key"
jwt_secret = "your-secret-key"
token_expiry = 3600
```

### Firewall Rules

```bash
# Allow WebSocket port
sudo ufw allow 8080/tcp

# Allow metrics port (internal only)
sudo ufw allow from 10.0.0.0/8 to any port 9090

# Allow health check port
sudo ufw allow 8081/tcp
```

## Troubleshooting

### Common Issues

1. **Model loading failures**
   ```bash
   # Check model files
   ls -la /opt/models/
   
   # Verify model compatibility
   obs-live-translator verify-models --verbose
   ```

2. **High latency**
   ```bash
   # Check CPU throttling
   cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   
   # Set to performance mode
   echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
   ```

3. **Memory issues**
   ```bash
   # Monitor memory usage
   watch -n 1 'ps aux | grep obs-live-translator'
   
   # Adjust memory limits
   export OBS_MAX_MEMORY=4096
   ```

4. **GPU not detected**
   ```bash
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   
   # Verify ONNX Runtime GPU support
   obs-live-translator check-gpu
   ```

### Debug Mode

```bash
# Enable debug logging
export RUST_LOG=debug
export RUST_BACKTRACE=full

# Run with verbose output
obs-live-translator --verbose --debug

# Generate debug report
obs-live-translator debug-report > debug.txt
```

## Backup and Recovery

### Configuration Backup

```bash
# Backup script
#!/bin/bash
BACKUP_DIR="/backup/obs-translator"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /etc/obs-translator/
tar -czf $BACKUP_DIR/models_$DATE.tar.gz /opt/models/

# Keep only last 7 backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
```

### Disaster Recovery

```bash
# Restore from backup
tar -xzf /backup/obs-translator/config_latest.tar.gz -C /
tar -xzf /backup/obs-translator/models_latest.tar.gz -C /

# Verify restoration
obs-live-translator verify-config
obs-live-translator verify-models

# Restart service
sudo systemctl restart obs-live-translator
```

## Support

For additional support:
- GitHub Issues: https://github.com/yourusername/obs-live-translator/issues
- Documentation: https://docs.obs-live-translator.io
- Community Forum: https://forum.obs-live-translator.io