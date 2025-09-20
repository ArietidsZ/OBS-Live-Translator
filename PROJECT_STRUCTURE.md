# OBS Live Translator - Project Structure

## Directory Layout

```
obs-live-translator/
├── src/
│   ├── audio/             # Audio processing modules
│   │   ├── mod.rs         # Audio module exports
│   │   ├── processor.rs   # Real-time audio processing
│   │   ├── features.rs    # FFT and mel-spectrogram extraction
│   │   ├── resampler.rs   # Audio resampling
│   │   └── vad.rs         # Voice Activity Detection
│   │
│   ├── inference/         # Machine learning inference
│   │   ├── mod.rs         # Inference module exports
│   │   ├── engine.rs      # Core ONNX Runtime engine
│   │   ├── whisper.rs     # Whisper ASR model
│   │   ├── translation.rs # Translation model
│   │   └── batch.rs       # Batch processing optimization
│   │
│   ├── config/            # Configuration management
│   │   └── mod.rs         # TOML-based configuration
│   │
│   ├── streaming/         # WebSocket streaming server
│   │   ├── mod.rs         # Streaming module exports
│   │   ├── server.rs      # Axum HTTP/WebSocket server
│   │   ├── session.rs     # Session management
│   │   └── websocket.rs   # WebSocket message handling
│   │
│   ├── native/            # FFI for potential C++ optimizations
│   │   ├── mod.rs         # Native module exports
│   │   ├── engine_ffi.rs  # FFI bindings
│   │   └── engine_stub.cpp # C++ stub implementation
│   │
│   ├── bin/               # Executable binaries
│   │   ├── server.rs      # Main WebSocket server
│   │   └── benchmark.rs   # Performance benchmarking
│   │
│   └── lib.rs             # Library root with main Translator
│
├── Cargo.toml             # Rust dependencies
├── build.rs               # C++ compilation script
└── README.md              # Project documentation
```

## Module Descriptions

### Audio Processing (`src/audio/`)
- **processor.rs**: Handles windowing, pre-emphasis, and buffering
- **features.rs**: Implements FFT using RustFFT and mel-spectrogram computation
- **resampler.rs**: Linear interpolation for sample rate conversion
- **vad.rs**: Energy-based Voice Activity Detection with zero-crossing rate

### Inference (`src/inference/`)
- **engine.rs**: ONNX Runtime session management with execution providers
- **whisper.rs**: Whisper ASR model for speech-to-text
- **translation.rs**: Neural machine translation
- **batch.rs**: Batched inference for improved throughput

### Configuration (`src/config/`)
- Centralized TOML configuration for audio, models, server, and translation settings
- Platform-specific paths for Windows, macOS, and Linux

### Streaming Server (`src/streaming/`)
- **server.rs**: Axum-based HTTP server with WebSocket endpoints
- **session.rs**: Per-client session state and model management
- **websocket.rs**: Binary and JSON message protocol handling

### Native FFI (`src/native/`)
- Minimal FFI layer for potential C++ optimizations
- Stub implementation for testing

## Key Features

1. **Real-time Processing**: Lock-free audio buffering with minimal latency
2. **Multi-client Support**: WebSocket-based streaming with session management
3. **Flexible Deployment**: CPU, CUDA, CoreML, and DirectML support
4. **Modular Architecture**: Clean separation between audio, inference, and networking

## Dependencies

- **Audio**: rustfft, hound
- **Web**: axum, tokio, serde
- **ML**: ONNX Runtime (system dependency)
- **Utils**: anyhow, tracing, mimalloc

## Build Requirements

- Rust 1.70+
- C++ compiler with C++17 support
- ONNX Runtime (via Homebrew on macOS: `brew install onnxruntime`)