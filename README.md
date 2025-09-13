# OBS Live Translator

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Rust](https://img.shields.io/badge/rust-1.70+-orange.svg)](https://www.rust-lang.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)

**Real-time multilingual translator for OBS Studio (In Development)**

Target support for **Chinese 🇨🇳 Japanese 🇯🇵 English 🇺🇸 Korean 🇰🇷** with planned AI-powered context awareness.

## Features

- **Target <50ms Latency**: Real-time translation pipeline (in development)
- **Rust Architecture**: High-performance core (under construction)
- **GPU Acceleration**: CUDA and TensorRT support (planned)
- **Memory Optimization**: Zero-copy audio processing (in progress)
- **Streaming Integration**: OBS overlay system (testing)

## Performance

| Metric | v1.0 (JavaScript) | v2.0 (Rust Core) | Improvement |
|--------|------------------|------------------|-------------|
| **Audio Latency** | TBD | Target <30ms | **Testing** |
| **ASR Processing** | TBD | Target <100ms | **In Progress** |
| **Translation** | TBD | Target <50ms | **Planned** |
| **Memory Usage** | TBD | Target <1GB | **Optimizing** |
| **GPU Utilization** | TBD | Target 90%+ | **Developing** |
| **CPU Usage** | TBD | Target <20% | **Benchmarking** |

## Architecture

```
Audio → Rust Core → GPU → Translation → OBS
```

### Tech Stack

- **Rust Core**: Memory safety, SIMD optimization
- **CUDA**: GPU acceleration
- **Candle ML**: ML inference
- **CPAL**: Audio processing
- **WebSocket**: Real-time streaming

## Installation

### Requirements
- GPU: NVIDIA RTX with 8GB+ VRAM
- RAM: 16GB+
- CPU: x64 with AVX2
- OS: Windows 10+, macOS 12+, Ubuntu 20+

### Setup

```bash
git clone https://github.com/your-username/obs-live-translator.git
cd obs-live-translator

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install CUDA toolkit
# Windows: Download from NVIDIA
# Linux: sudo apt install nvidia-cuda-toolkit
# macOS: brew install --cask cuda

# Build
cargo build --release
npm install

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

## Development Status

### Phase 1: Core Foundation (Current)
- [x] Project structure and architecture
- [x] Rust core framework setup
- [ ] Basic audio processing pipeline
- [ ] Speech recognition integration
- [ ] Translation engine implementation

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