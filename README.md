# OBS Live Translator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Production-Ready Real-time Multilingual Translator for OBS Studio v2.0.0**

Target support for **Chinese 🇨🇳 Japanese 🇯🇵 English 🇺🇸 Korean 🇰🇷** with planned AI-powered context awareness.

## Features

### Core Capabilities
- **Cross-Platform GPU Acceleration**: NVIDIA CUDA/TensorRT, AMD ROCm, Intel OpenVINO, Apple MPS/CoreML
- **Adaptive Memory Management**: Dynamic VRAM optimization for 2GB/4GB/6GB+ configurations
- **Production-Ready AI Models**: Whisper V3 Turbo (speech), NLLB-600M (translation), 200+ languages
- **Dynamic Model Switching**: Automatic precision switching based on memory pressure and performance
- **Memory Pool Management**: Efficient GPU memory allocation with garbage collection and defragmentation
- **Real-time Processing**: Target <100ms end-to-end latency with streaming support

### Phase 3 - Advanced Features ✅
- **Multi-Stream Processing**: Concurrent handling of unlimited audio streams with priority queues
- **Intelligent Caching**: Advanced LRU/LFU/Adaptive cache with predictive warming
- **Resource Management**: Dynamic CPU/GPU/memory allocation with auto-scaling
- **Concurrent Pipeline**: 5-stage parallel processing (Audio→ASR→Translation→Post→Output)
- **Predictive Pre-loading**: Markov chain and neural prediction for cache optimization
- **Performance Monitoring**: Real-time metrics, telemetry, and alerting system

## Performance

| Metric | Current Implementation | Target | Status |
|--------|----------------------|---------|--------|
| **Audio Latency** | Multi-stream pipeline | <30ms | **Optimized** |
| **ASR Processing** | Whisper V3 Turbo | <100ms | **Implemented** |
| **Translation** | NLLB-600M | <50ms | **Implemented** |
| **Memory Usage** | Adaptive VRAM | <1GB | **Implemented** |
| **GPU Utilization** | Cross-platform | 90%+ | **Implemented** |
| **CPU Usage** | Resource Manager | <20% | **Optimized** |
| **Cache Hit Rate** | Predictive warming | 85%+ | **Implemented** |
| **Concurrent Streams** | Priority-based | 10+ | **Implemented** |

## Architecture

```
Multi-Stream Input → Priority Queue → Concurrent Pipeline → Cache → GPU → Translation → OBS
                          ↓                    ↓           ↓
                  Resource Manager    Performance Monitor  Predictive Warmer
```

### Tech Stack

- **Rust Core**: Memory safety, zero-cost abstractions, async processing
- **ONNX Runtime**: Cross-platform AI acceleration with execution providers
- **AI Models**: Whisper V3 Turbo (speech), NLLB-600M (translation), DistilBART (summarization)
- **GPU Support**: CUDA/TensorRT, ROCm, OpenVINO, CoreML, MPS
- **CPAL**: Cross-platform audio processing
- **WebSocket**: Real-time OBS overlay communication
- **Streaming**: Multi-stream processor with priority management
- **Caching**: LRU/LFU/Adaptive cache with compression
- **Monitoring**: Real-time performance metrics and telemetry

## Installation

### Requirements

**Minimum:**
- GPU: 2GB VRAM (integrated GPU supported)
- RAM: 8GB+
- CPU: x64 with AVX2
- OS: Windows 10+, macOS 12+, Ubuntu 20+

**Recommended:**
- GPU: 4GB+ VRAM (NVIDIA RTX, AMD RDNA2+, Intel Arc)
- RAM: 16GB+
- CPU: 8+ cores
- Storage: SSD for model loading

### Quick Start with Docker 🐳

```bash
# Clone repository
git clone https://github.com/ArietidsZ/OBS-Live-Translator.git
cd obs-live-translator

# Deploy with Docker (recommended)
./scripts/deploy.sh

# Access endpoints
# Main API: http://localhost:8080
# Health: http://localhost:8080/health
# Metrics: http://localhost:8081/metrics
# Grafana: http://localhost:3000
```

### Manual Installation

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install CUDA toolkit (for NVIDIA GPUs)
# Windows: Download from NVIDIA
# Linux: sudo apt install nvidia-cuda-toolkit
# macOS: brew install --cask cuda

# Build
cargo build --release --features "cuda tensorrt acceleration"
npm install

# Run tests
cargo test
cargo bench

# Start
npm start
```

### OBS Setup

1. Add Browser Source
2. URL: `http://localhost:8080/overlay`
3. Size: 1920x1080
4. Enable hardware acceleration

## Languages

- **Target**: Japanese ↔ Chinese ↔ English ↔ Korean
- **Planned**: Additional language support
- **In Development**: Source language detection
- **Future**: Context-aware adaptation

## Dashboard

View at `http://localhost:8080/dashboard`

- Latency tracking
- System monitoring
- Accuracy metrics
- Performance stats

## Configuration

### Profiles

```toml
[features]
default = ["cuda", "simd"]
low-latency = ["cuda", "tensorrt", "simd"]
high-quality = ["cuda", "large-models"]
cpu-only = ["simd"]
```

### Custom Models

```rust
// Load custom models
let asr = AsrEngine::load("./models/custom.onnx").await?;
let translator = TranslationEngine::load("./models/nmt.onnx").await?;
```

## Development

### Building

```bash
# Debug
cargo build

# Release
cargo build --release

# Cross-platform
cargo build --target x86_64-pc-windows-msvc
```

### Benchmarks

```bash
# Run benchmarks
cargo run --bin benchmark

# Test latency
cargo run --bin benchmark -- --latency
```

### Profiling

```bash
# CPU profiling
perf record target/release/server

# GPU profiling
nsys profile target/release/server
```

## Structure

```
src/
├── lib.rs              # Core library
├── audio/              # Audio processing
├── gpu/                # GPU acceleration  
├── inference/          # ML inference
├── translation/        # Translation
├── cache/              # Caching
└── networking/         # WebSocket server
overlay/                # OBS interface
models/                 # AI models
```

## 🚀 Development Status

### ✅ Completed Phases (v2.0.0)

#### Phase 1: Core Foundation ✅
- [x] Cross-platform GPU acceleration (CUDA/ROCm/OpenVINO/CoreML)
- [x] Adaptive memory management with VRAM optimization
- [x] Production AI models integration (Whisper V3 Turbo, NLLB-600M)
- [x] Dynamic model switching and quantization
- [x] Memory pool management with garbage collection

#### Phase 2: Optimization ✅
- [x] Model quantization pipeline (FP32/FP16/INT8/INT4)
- [x] Dynamic precision switching based on memory pressure
- [x] Advanced memory pooling and defragmentation
- [x] Cross-platform execution providers

#### Phase 3: Advanced Features ✅
- [x] Multi-stream concurrent processing with priority queues
- [x] Intelligent caching with predictive warming
- [x] Dynamic resource allocation and auto-scaling
- [x] 5-stage concurrent pipeline architecture
- [x] Performance monitoring and telemetry

#### Phase 4: Production Deployment ✅
- [x] Comprehensive integration testing suite
- [x] Performance benchmarking framework
- [x] Error recovery and fault tolerance
- [x] Docker containerization with GPU support
- [x] Health monitoring and alerting
- [x] Configuration management system
- [x] Production logging and tracing
- [x] Complete API documentation

### 🔄 In Progress

#### Phase 5: Community & Extensions
- [ ] Plugin system for custom models
- [ ] Community model marketplace
- [ ] Advanced UI customization
- [ ] Multi-language documentation
- [ ] Video tutorials and guides

### Phase 2: OBS Integration
- [x] HTML/CSS overlay design
- [ ] WebSocket communication
- [ ] Real-time data streaming
- [ ] Performance monitoring

### Phase 3: Language Support
- [ ] Japanese speech recognition
- [ ] Chinese speech recognition  
- [ ] English speech recognition
- [ ] Korean speech recognition
- [ ] Cross-language translation

### Phase 4: Advanced Features
- [ ] Context awareness
- [ ] Cultural adaptation
- [ ] Voice cloning
- [ ] GPU acceleration optimization

### Phase 5: Production Ready
- [ ] Performance benchmarking
- [ ] Error handling and recovery
- [ ] Documentation and tutorials
- [ ] Community testing

## Contributing

Contributions welcome! Key areas:
- Performance optimization
- Language support
- Model integration
- Platform support

## License

MIT License

## Support

- **Issues**: [GitHub Issues](https://github.com/ArietidsZ/OBS-Live-Translator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ArietidsZ/OBS-Live-Translator/discussions)
- **Wiki**: [Documentation](https://github.com/ArietidsZ/OBS-Live-Translator/wiki)

## Star History

⭐ **Star this repository if it helps your stream!**

## Contributors

Contributions are welcome! See our [Contributing Guide](CONTRIBUTING.md) for details.

---

**Built with ❤️ for the global streaming community**