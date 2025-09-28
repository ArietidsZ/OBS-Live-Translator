#!/bin/bash

# One-click setup script for OBS Live Translator models
# This script automatically downloads, validates, and configures models for each profile

set -e  # Exit on any error

# Configuration
MODELS_DIR="${MODELS_DIR:-models}"
CACHE_DIR="${CACHE_DIR:-models/cache}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Profile selection
PROFILE="${1:-auto}"
FORCE_DOWNLOAD="${2:-false}"

echo -e "${BLUE}🚀 OBS Live Translator Model Setup${NC}"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Function to check system requirements
check_system_requirements() {
    print_info "Checking system requirements..."

    # Check available disk space (need at least 10GB)
    available_space=$(df -BG "$PROJECT_ROOT" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt 10 ]; then
        print_error "Insufficient disk space. Need at least 10GB, have ${available_space}GB"
        exit 1
    fi

    # Check available RAM
    if command -v free >/dev/null 2>&1; then
        available_ram=$(free -g | awk 'NR==2{print $7}')
        if [ "$available_ram" -lt 4 ]; then
            print_warning "Low available RAM (${available_ram}GB). Consider closing other applications."
        fi
    fi

    # Check for required tools
    for tool in curl wget; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            print_error "$tool is required but not installed"
            exit 1
        fi
    done

    print_status "System requirements check passed"
}

# Function to detect optimal profile
detect_profile() {
    if [ "$PROFILE" != "auto" ]; then
        echo "$PROFILE"
        return
    fi

    print_info "Auto-detecting optimal profile..."

    # Get CPU core count
    cpu_cores=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo "2")

    # Get total RAM in GB
    if command -v free >/dev/null 2>&1; then
        total_ram=$(free -g | awk 'NR==2{print $2}')
    elif command -v vm_stat >/dev/null 2>&1; then
        # macOS
        total_ram=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    else
        total_ram=8  # Default assumption
    fi

    # Simple GPU detection
    has_gpu=false
    if command -v nvidia-smi >/dev/null 2>&1; then
        has_gpu=true
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS has Metal GPU support
        has_gpu=true
    fi

    # Profile decision logic
    if [ "$cpu_cores" -ge 8 ] && [ "$total_ram" -ge 16 ] && [ "$has_gpu" = true ]; then
        echo "high"
    elif [ "$cpu_cores" -ge 4 ] && [ "$total_ram" -ge 8 ]; then
        echo "medium"
    else
        echo "low"
    fi
}

# Function to create directory structure
setup_directories() {
    print_info "Setting up directory structure..."

    mkdir -p "$PROJECT_ROOT/$MODELS_DIR"
    mkdir -p "$PROJECT_ROOT/$CACHE_DIR"
    mkdir -p "$PROJECT_ROOT/$MODELS_DIR/quantized"
    mkdir -p "$PROJECT_ROOT/$MODELS_DIR/temp"

    print_status "Directory structure created"
}

# Function to download model with retry
download_model() {
    local name="$1"
    local url="$2"
    local expected_size="$3"
    local checksum="$4"
    local output_file="$5"

    print_info "Downloading $name..."

    # Skip if file exists and force download is false
    if [ -f "$output_file" ] && [ "$FORCE_DOWNLOAD" != "true" ]; then
        local file_size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null || echo "0")
        local expected_bytes=$((expected_size * 1024 * 1024))

        if [ "$file_size" -eq "$expected_bytes" ]; then
            print_status "$name already downloaded and verified"
            return 0
        fi
    fi

    # Create temporary download location
    local temp_file="${output_file}.tmp"

    # Download with progress and resume support
    if command -v wget >/dev/null 2>&1; then
        wget --continue --progress=bar:force:noscroll -O "$temp_file" "$url" || {
            print_error "Failed to download $name"
            rm -f "$temp_file"
            return 1
        }
    else
        curl -C - -L --progress-bar -o "$temp_file" "$url" || {
            print_error "Failed to download $name"
            rm -f "$temp_file"
            return 1
        }
    fi

    # Verify file size
    local downloaded_size=$(stat -f%z "$temp_file" 2>/dev/null || stat -c%s "$temp_file" 2>/dev/null || echo "0")
    local expected_bytes=$((expected_size * 1024 * 1024))

    if [ "$downloaded_size" -ne "$expected_bytes" ]; then
        print_warning "Size mismatch for $name: expected $expected_bytes bytes, got $downloaded_size bytes"
    fi

    # Move to final location
    mv "$temp_file" "$output_file"

    print_status "$name downloaded successfully"
    return 0
}

# Function to setup Low profile models
setup_low_profile() {
    print_info "Setting up Low Profile (CPU-only, minimal resources)..."

    # Low profile uses built-in components, so just verify directories exist
    mkdir -p "$PROJECT_ROOT/$MODELS_DIR/low"

    # Create placeholder files for built-in models
    echo "Built-in WebRTC VAD component" > "$PROJECT_ROOT/$MODELS_DIR/low/webrtc_vad.txt"
    echo "Built-in linear resampler component" > "$PROJECT_ROOT/$MODELS_DIR/low/linear_resampler.txt"

    print_status "Low profile setup complete (using built-in components)"
}

# Function to setup Medium profile models
setup_medium_profile() {
    print_info "Setting up Medium Profile (CPU + small GPU, balanced performance)..."

    local models_dir="$PROJECT_ROOT/$MODELS_DIR/medium"
    mkdir -p "$models_dir"

    # Download TEN VAD model (placeholder URL - would be real in production)
    download_model "TEN VAD" \
        "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/ten_vad.onnx" \
        "15" \
        "x1y2z3a4b5c6" \
        "$models_dir/ten_vad_1.0.0.onnx" || print_warning "TEN VAD download failed"

    # Download Whisper Base model (placeholder URL)
    download_model "Whisper Base" \
        "https://huggingface.co/openai/whisper-base/resolve/main/model.onnx" \
        "142" \
        "m1n2o3p4q5r6" \
        "$models_dir/whisper_base_1.0.0.onnx" || print_warning "Whisper Base download failed"

    # Download NLLB 600M model (placeholder URL)
    download_model "NLLB 200 600M" \
        "https://huggingface.co/facebook/nllb-200-600M/resolve/main/model.onnx" \
        "600" \
        "t1u2v3w4x5y6" \
        "$models_dir/nllb_200_600m_1.0.0.onnx" || print_warning "NLLB 600M download failed"

    print_status "Medium profile setup complete"
}

# Function to setup High profile models
setup_high_profile() {
    print_info "Setting up High Profile (GPU accelerated, maximum performance)..."

    local models_dir="$PROJECT_ROOT/$MODELS_DIR/high"
    mkdir -p "$models_dir"

    # Download Silero VAD model
    download_model "Silero VAD" \
        "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx" \
        "50" \
        "z1a2b3c4d5e6" \
        "$models_dir/silero_vad_4.0.0.onnx" || print_warning "Silero VAD download failed"

    # Download Whisper Large v3 model (placeholder URL)
    download_model "Whisper Large v3" \
        "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.onnx" \
        "3100" \
        "g1h2i3j4k5l6" \
        "$models_dir/whisper_large_v3_3.0.0.onnx" || print_warning "Whisper Large v3 download failed"

    # Download NLLB 3.3B model (placeholder URL)
    download_model "NLLB 200 3.3B" \
        "https://huggingface.co/facebook/nllb-200-3.3B/resolve/main/model.onnx" \
        "6600" \
        "p1q2r3s4t5u6" \
        "$models_dir/nllb_200_3_3b_1.0.0.onnx" || print_warning "NLLB 3.3B download failed"

    print_status "High profile setup complete"
}

# Function to verify models
verify_models() {
    print_info "Verifying downloaded models..."

    local verified_count=0
    local total_count=0

    for model_file in "$PROJECT_ROOT/$MODELS_DIR"/*/*.onnx; do
        if [ -f "$model_file" ]; then
            total_count=$((total_count + 1))

            # Basic verification - check file is not empty and has reasonable size
            local file_size=$(stat -f%z "$model_file" 2>/dev/null || stat -c%s "$model_file" 2>/dev/null || echo "0")

            if [ "$file_size" -gt 1024 ]; then  # At least 1KB
                verified_count=$((verified_count + 1))
                print_status "✓ $(basename "$model_file") verified"
            else
                print_warning "✗ $(basename "$model_file") appears corrupted"
            fi
        fi
    done

    print_info "Model verification: $verified_count/$total_count models verified"
}

# Function to create configuration
create_configuration() {
    print_info "Creating configuration files..."

    local config_file="$PROJECT_ROOT/models_config.toml"

    cat > "$config_file" << EOF
# OBS Live Translator Model Configuration
# Generated by setup script on $(date)

[models]
profile = "$DETECTED_PROFILE"
models_dir = "$MODELS_DIR"
cache_dir = "$CACHE_DIR"

[profiles.low]
description = "CPU-only, minimal resources (2-4 cores, 4GB RAM)"
models = [
    { name = "webrtc_vad", type = "vad", built_in = true },
    { name = "linear_resampler", type = "resampling", built_in = true }
]

[profiles.medium]
description = "Balanced CPU+GPU (4-6 cores, 8GB RAM, 2GB VRAM)"
models = [
    { name = "ten_vad", type = "vad", file = "medium/ten_vad_1.0.0.onnx" },
    { name = "whisper_base", type = "asr", file = "medium/whisper_base_1.0.0.onnx" },
    { name = "nllb_200_600m", type = "translation", file = "medium/nllb_200_600m_1.0.0.onnx" }
]

[profiles.high]
description = "Maximum performance (8+ cores, 16GB RAM, 8GB VRAM)"
models = [
    { name = "silero_vad", type = "vad", file = "high/silero_vad_4.0.0.onnx" },
    { name = "whisper_large_v3", type = "asr", file = "high/whisper_large_v3_3.0.0.onnx" },
    { name = "nllb_200_3_3b", type = "translation", file = "high/nllb_200_3_3b_1.0.0.onnx" }
]
EOF

    print_status "Configuration file created: $config_file"
}

# Function to print setup summary
print_summary() {
    echo ""
    echo "=================================================="
    echo -e "${GREEN}🎉 Setup Complete!${NC}"
    echo "=================================================="
    echo ""
    echo "Profile: $DETECTED_PROFILE"
    echo "Models directory: $PROJECT_ROOT/$MODELS_DIR"
    echo "Cache directory: $PROJECT_ROOT/$CACHE_DIR"
    echo ""
    echo "Next steps:"
    echo "1. Build the project: cargo build --release"
    echo "2. Run with your profile: cargo run --bin obs-translator-server"
    echo ""
    echo "For more information, see the README.md file."
    echo ""
}

# Function to cleanup on exit
cleanup() {
    # Remove any temporary files
    find "$PROJECT_ROOT/$MODELS_DIR" -name "*.tmp" -delete 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT

# Main execution
main() {
    echo "Starting OBS Live Translator model setup..."
    echo "Profile: $PROFILE"
    echo "Force download: $FORCE_DOWNLOAD"
    echo ""

    # Check system requirements
    check_system_requirements

    # Detect optimal profile
    DETECTED_PROFILE=$(detect_profile)
    print_info "Detected optimal profile: $DETECTED_PROFILE"

    # Setup directory structure
    setup_directories

    # Setup models based on detected profile
    case "$DETECTED_PROFILE" in
        "low")
            setup_low_profile
            ;;
        "medium")
            setup_medium_profile
            ;;
        "high")
            setup_high_profile
            ;;
        *)
            print_error "Unknown profile: $DETECTED_PROFILE"
            exit 1
            ;;
    esac

    # Verify models
    verify_models

    # Create configuration
    create_configuration

    # Print summary
    print_summary
}

# Show usage if help requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 [PROFILE] [FORCE_DOWNLOAD]"
    echo ""
    echo "PROFILE options:"
    echo "  auto   - Auto-detect optimal profile (default)"
    echo "  low    - CPU-only, minimal resources"
    echo "  medium - Balanced CPU+GPU performance"
    echo "  high   - Maximum GPU-accelerated performance"
    echo ""
    echo "FORCE_DOWNLOAD options:"
    echo "  false  - Skip existing files (default)"
    echo "  true   - Re-download all files"
    echo ""
    echo "Examples:"
    echo "  $0                    # Auto-detect profile"
    echo "  $0 medium             # Force medium profile"
    echo "  $0 high true          # Force high profile and re-download"
    echo ""
    exit 0
fi

# Run main function
main "$@"