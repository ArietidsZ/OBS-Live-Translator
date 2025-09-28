# OBS Live Translator - Project Structure

## Overview
A high-performance real-time speech translation system with dynamic profile-based optimization for OBS streaming. The system automatically adapts to hardware capabilities with three performance profiles (Low/Medium/High) and provides sub-500ms latency translation.

## Core Architecture

```
obs-live-translator/
├── src/
│   ├── lib.rs                      # Main library entry point with Translator API
│   ├── bin/
│   │   └── server.rs                # WebSocket server binary
│   │
│   ├── profile/                    # Profile-based system management
│   │   ├── mod.rs                   # Profile enum and core logic
│   │   ├── manager.rs               # Profile lifecycle management
│   │   ├── components.rs            # Profile-aware component registry
│   │   └── monitor.rs               # Resource monitoring for profile switching
│   │
│   ├── audio/                       # Audio processing pipeline
│   │   ├── mod.rs                   # Audio module coordination
│   │   ├── processor_trait.rs       # AudioProcessor trait definition
│   │   ├── pipeline.rs              # Unified audio pipeline
│   │   ├── vad/                     # Voice Activity Detection
│   │   │   ├── mod.rs               # VAD trait and factory
│   │   │   ├── webrtc_vad.rs       # Low profile: WebRTC VAD
│   │   │   ├── ten_vad.rs          # Medium profile: TEN VAD (ONNX)
│   │   │   ├── silero_vad.rs       # High profile: Silero VAD (GPU)
│   │   │   └── adaptive_vad.rs     # Adaptive VAD with confidence scoring
│   │   ├── resampling/              # Audio resampling
│   │   │   ├── mod.rs               # Resampler trait and factory
│   │   │   ├── linear_resampler.rs  # Low profile: SIMD linear
│   │   │   ├── cubic_resampler.rs   # Medium profile: Cubic Hermite
│   │   │   ├── soxr_resampler.rs    # High profile: libsoxr FFI
│   │   │   └── adaptive_resampler.rs # Adaptive quality selection
│   │   └── features/                # Feature extraction
│   │       ├── mod.rs               # Feature extractor trait
│   │       ├── rustfft_extractor.rs # Low profile: RustFFT basic
│   │       ├── enhanced_extractor.rs # Medium profile: Enhanced RustFFT
│   │       ├── ipp_extractor.rs     # High profile: Intel IPP
│   │       └── adaptive_extractor.rs # Adaptive feature extraction
│   │
│   ├── asr/                         # Speech Recognition
│   │   ├── mod.rs                   # ASR engine trait and factory
│   │   ├── whisper_tiny.rs          # Low profile: Whisper-tiny INT8
│   │   ├── whisper_small.rs         # Medium profile: Whisper-small FP16
│   │   ├── parakeet_streaming.rs    # High profile: Parakeet-TDT streaming
│   │   ├── inference_engine.rs      # ONNX Runtime integration
│   │   └── adaptive_asr.rs          # Adaptive model selection
│   │
│   ├── language_detection/          # Language Detection
│   │   ├── mod.rs                   # Language detector trait
│   │   ├── fasttext_detector.rs     # FastText-based detection
│   │   ├── language_fusion.rs       # Audio+text fusion (high profile)
│   │   └── pipeline.rs              # Detection pipeline
│   │
│   ├── translation/                 # Neural Machine Translation
│   │   ├── mod.rs                   # Translation engine trait
│   │   ├── marian_translator.rs     # Low profile: MarianNMT INT8
│   │   ├── m2m_translator.rs        # Medium profile: M2M-100 FP16
│   │   ├── nllb_translator.rs       # High profile: NLLB-200
│   │   ├── cache.rs                 # Translation caching
│   │   └── translation_pipeline.rs  # Unified translation pipeline
│   │
│   ├── inference/                   # ML Inference Infrastructure
│   │   ├── mod.rs                   # Inference coordination
│   │   ├── engine.rs                # Base inference engine
│   │   ├── unified_framework.rs     # Profile-aware model loading
│   │   ├── acceleration.rs          # Hardware acceleration (TensorRT, ONNX)
│   │   ├── batch.rs                 # Batch processing optimization
│   │   └── tokenizer.rs             # Tokenization utilities
│   │
│   ├── streaming/                   # Real-time Streaming
│   │   ├── mod.rs                   # Streaming module coordination
│   │   ├── websocket.rs             # WebSocket server (Axum)
│   │   ├── optimized_websocket.rs   # Binary protocol optimization
│   │   ├── enhanced_session.rs      # Session management
│   │   ├── opus_codec.rs            # Opus audio codec integration
│   │   ├── protocol_optimizer.rs    # Protocol optimization
│   │   └── realtime_pipeline.rs     # Real-time processing pipeline
│   │
│   ├── memory/                      # Memory Management
│   │   ├── mod.rs                   # Memory allocator setup (MiMalloc)
│   │   └── audio_buffers.rs         # Zero-copy audio buffers
│   │
│   ├── models/                      # Model Management
│   │   ├── mod.rs                   # Model coordination
│   │   ├── downloader.rs            # Model downloading with progress
│   │   ├── validator.rs             # Model integrity validation
│   │   ├── cache.rs                 # Model caching system
│   │   └── quantizer.rs             # Model quantization (INT8/FP16)
│   │
│   ├── config/                      # Configuration System
│   │   ├── mod.rs                   # Configuration traits
│   │   ├── profile_config.rs        # Profile-aware configuration
│   │   └── setup_manager.rs         # One-click setup automation
│   │
│   ├── monitoring/                  # Performance Monitoring
│   │   ├── mod.rs                   # Monitoring coordination
│   │   ├── metrics.rs               # Metrics collection
│   │   ├── adaptive_optimizer.rs    # Automatic profile switching
│   │   ├── dashboard.rs             # Real-time performance dashboard
│   │   └── alerts.rs                # Performance alerting system
│   │
│   ├── quality/                     # Quality Assurance
│   │   ├── mod.rs                   # QA system coordination
│   │   ├── translation_qa.rs        # Translation quality (BLEU, confidence)
│   │   ├── audio_qa.rs              # Audio quality (SNR, distortion)
│   │   ├── metrics.rs               # Quality metrics collection
│   │   └── feedback.rs              # User feedback system
│   │
│   ├── native/                      # Native Optimizations
│   │   ├── mod.rs                   # FFI coordination
│   │   ├── engine_ffi.rs            # C++ engine bindings
│   │   └── simd_ffi.rs              # SIMD optimizations
│   │
│   └── tokenization/                # Tokenization Support
│       ├── mod.rs                   # Tokenization coordination
│       └── nllb_tokenizer.rs        # NLLB tokenizer implementation
│
├── tests/
│   ├── unit/                        # Unit Tests
│   │   ├── mod.rs                   # Test utilities
│   │   ├── audio_tests.rs           # Audio pipeline tests
│   │   ├── inference_tests.rs       # ML inference tests
│   │   └── streaming_tests.rs       # Streaming protocol tests
│   │
│   ├── integration/                 # Integration Tests
│   │   ├── mod.rs                   # Integration test utilities
│   │   ├── e2e_tests.rs             # End-to-end pipeline tests
│   │   ├── performance_tests.rs     # Performance validation tests
│   │   └── stress_tests.rs          # Stress and load tests
│   │
│   └── performance/                 # Performance Testing
│       ├── mod.rs                   # Performance test framework
│       ├── benchmarks.rs            # Criterion benchmarks
│       ├── latency_suite.rs         # Latency measurement suite
│       ├── memory_validator.rs      # Memory leak detection
│       └── regression_detector.rs   # Performance regression detection
│
├── docs/
│   ├── API.md                       # Complete API documentation
│   └── DEPLOYMENT.md                # Deployment guide (Docker, K8s)
│
├── scripts/
│   └── deploy.sh                    # Automated deployment script
│
├── .github/
│   └── workflows/
│       └── release.yml              # CI/CD release pipeline
│
├── Cargo.toml                       # Project dependencies
├── build.rs                         # Build script with profile detection
└── README.md                        # Project documentation
```

## Performance Profiles

### Low Profile (Resource-Constrained)
- **Target Hardware**: 4GB RAM, 2 CPU cores
- **Models**: Whisper-tiny INT8, MarianNMT INT8
- **Latency Target**: < 500ms
- **Memory Usage**: ~932MB
- **Key Features**: WebRTC VAD, SIMD linear resampling, RustFFT

### Medium Profile (Balanced)
- **Target Hardware**: 4GB RAM, 2GB VRAM, 2 CPU cores
- **Models**: Whisper-small FP16, M2M-100
- **Latency Target**: < 400ms
- **Memory Usage**: 713MB RAM + 2.85GB VRAM
- **Key Features**: TEN VAD, Cubic resampling, GPU acceleration

### High Profile (Maximum Quality)
- **Target Hardware**: 8GB RAM, 8GB VRAM, 6 CPU cores
- **Models**: Parakeet-TDT, NLLB-200
- **Latency Target**: < 250ms
- **Memory Usage**: 1.05GB RAM + 6.55GB VRAM
- **Key Features**: Silero VAD, libsoxr, Intel IPP, TensorRT

## Key Features

### Dynamic Profile Switching
- Automatic hardware detection
- Resource-based profile selection
- Seamless runtime profile transitions
- Graceful degradation under load

### One-Click Setup
- Automated hardware detection
- Optimal profile selection
- Model downloading and caching
- Configuration generation
- Performance validation

### Real-Time Processing
- Sub-500ms end-to-end latency
- Jitter buffer management
- Binary WebSocket protocol
- Opus audio compression
- Backpressure handling

### Quality Assurance
- Real-time BLEU score calculation
- Translation confidence scoring
- Audio quality monitoring (SNR, distortion)
- User feedback collection
- Quality trend analysis

### Monitoring & Optimization
- Real-time performance metrics
- Adaptive profile switching
- Resource utilization tracking
- Latency monitoring
- Performance alerts

## Technology Stack

### Core Languages
- **Rust**: Primary implementation language
- **C++**: Native optimizations (WebRTC VAD, libsoxr, Intel IPP)
- **Python**: Advanced ML model integration (via PyO3)

### ML Frameworks
- **ONNX Runtime**: Primary inference engine
- **TensorRT**: NVIDIA GPU optimization (high profile)
- **OpenVINO**: Intel optimization (optional)

### Audio Processing
- **RustFFT**: FFT processing
- **WebRTC**: Voice activity detection
- **Opus**: Audio compression
- **libsoxr**: High-quality resampling

### Web Technologies
- **Axum**: WebSocket server framework
- **Tokio**: Async runtime
- **Tungstenite**: WebSocket implementation

### Build & Deployment
- **Docker**: Container deployment
- **Kubernetes**: Orchestration support
- **GitHub Actions**: CI/CD pipeline

## Development Workflow

### Building
```bash
# Debug build
cargo build

# Release build with optimizations
cargo build --release

# Profile-specific build
cargo build --release --features profile-high
```

### Testing
```bash
# Run all tests
cargo test

# Run unit tests
cargo test --lib

# Run integration tests
cargo test --test '*'

# Run benchmarks
cargo bench
```

### Deployment
```bash
# One-click setup
./scripts/deploy.sh install medium

# Docker deployment
docker build -t obs-live-translator .
docker run -p 8080:8080 obs-live-translator

# Kubernetes deployment
kubectl apply -f k8s/deployment.yaml
```

## API Usage

### Basic Translation
```rust
use obs_live_translator::{Translator, TranslatorConfig, Profile};

// Initialize with automatic profile detection
let config = TranslatorConfig::from_profile(Profile::Medium);
let translator = Translator::new(config).await?;

// Process audio
let audio_data: Vec<f32> = get_audio_data();
let result = translator.process_audio(&audio_data).await?;

println!("Transcription: {}", result.transcription.unwrap());
println!("Translation: {}", result.translation.unwrap());
```

### WebSocket Streaming
```javascript
const ws = new WebSocket('ws://localhost:8080');
ws.binaryType = 'arraybuffer';

ws.onmessage = (event) => {
  const result = JSON.parse(event.data);
  console.log('Translation:', result.translation);
};

// Send audio chunks
ws.send(audioChunk);
```

## Performance Metrics

### Latency Breakdown (Medium Profile)
- VAD: 50ms
- Resampling: 8ms
- Feature Extraction: 20ms
- ASR: 150ms
- Translation: 120ms
- Network Overhead: 15ms
- **Total: 363ms**

### Resource Usage
- **CPU**: 27% (2 cores)
- **Memory**: 713MB RAM + 2.85GB VRAM
- **Network**: 6-64 kbps (Opus compression)

## Contributing
Please see CONTRIBUTING.md for development guidelines and code style requirements.

## License
Apache 2.0 License - See LICENSE file for details.
