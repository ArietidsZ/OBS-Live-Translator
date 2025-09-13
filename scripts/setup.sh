#!/bin/bash
# High-Performance OBS Live Translator Setup Script
set -e

echo "🚀 Setting up OBS Live Translator v2.0 (Rust Edition)"

# Check for required tools
check_requirements() {
    echo "📋 Checking requirements..."
    
    if ! command -v cargo &> /dev/null; then
        echo "❌ Rust/Cargo not found. Installing..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
        source ~/.cargo/env
    fi
    
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js not found. Please install Node.js 18+ first."
        exit 1
    fi
    
    echo "✅ Requirements check passed"
}

# Check for CUDA support
check_cuda() {
    echo "🖥️  Checking for CUDA support..."
    
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        CUDA_AVAILABLE=true
    else
        echo "⚠️  NVIDIA GPU not detected, will use CPU fallback"
        CUDA_AVAILABLE=false
    fi
}

# Build Rust core
build_rust_core() {
    echo "🦀 Building optimized Rust core..."
    
    if [ "$CUDA_AVAILABLE" = true ]; then
        echo "Building with CUDA acceleration..."
        cargo build --release --features cuda,tensorrt,simd
    else
        echo "Building with CPU-only optimizations..."
        cargo build --release --features simd
    fi
    
    echo "✅ Rust core built successfully"
}

# Install Node.js dependencies
install_node_deps() {
    echo "📦 Installing Node.js dependencies..."
    npm install --production
    echo "✅ Node.js dependencies installed"
}

# Download AI models
download_models() {
    echo "🧠 Setting up AI models..."
    
    mkdir -p models
    
    # Download optimized models for different performance modes
    echo "Downloading Whisper models..."
    curl -L "https://huggingface.co/openai/whisper-small/resolve/main/pytorch_model.bin" -o models/whisper-small.bin || echo "⚠️  Model download failed, will download at runtime"
    
    echo "Downloading NLLB translation models..."
    curl -L "https://huggingface.co/facebook/nllb-200-distilled-600M/resolve/main/pytorch_model.bin" -o models/nllb-600m.bin || echo "⚠️  Model download failed, will download at runtime"
    
    echo "✅ Models setup complete"
}

# Setup development environment
setup_dev_env() {
    if [ "$1" = "--dev" ]; then
        echo "🔧 Setting up development environment..."
        
        # Install development dependencies
        cargo install cargo-watch
        npm install --include=dev
        
        # Setup pre-commit hooks
        echo "Setting up git hooks..."
        echo "#!/bin/bash\ncargo test && cargo fmt --check && cargo clippy" > .git/hooks/pre-commit
        chmod +x .git/hooks/pre-commit
        
        echo "✅ Development environment ready"
    fi
}

# Run benchmarks
run_benchmarks() {
    if [ "$1" = "--benchmark" ]; then
        echo "📊 Running performance benchmarks..."
        
        cargo run --release --bin obs-translator-server -- --benchmark
        
        echo "✅ Benchmarks complete"
    fi
}

# Main setup flow
main() {
    check_requirements
    check_cuda
    build_rust_core
    install_node_deps
    download_models
    setup_dev_env "$1"
    run_benchmarks "$1"
    
    echo ""
    echo "🎉 OBS Live Translator v2.0 setup complete!"
    echo ""
    echo "🚀 Quick Start:"
    echo "  1. Start server: npm start"
    echo "  2. Open OBS Studio"
    echo "  3. Add Browser Source: http://localhost:8080/overlay"
    echo "  4. Set size to 1920x1080"
    echo ""
    echo "📊 Performance Dashboard: http://localhost:8080/dashboard"
    echo "📖 Documentation: README.md"
    echo ""
    echo "Happy streaming! 🎮✨"
}

# Parse command line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    echo "OBS Live Translator Setup Script"
    echo ""
    echo "Usage: ./setup.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --dev        Setup development environment"
    echo "  --benchmark  Run performance benchmarks after setup"
    echo "  --help       Show this help message"
    echo ""
    exit 0
fi

main "$1"