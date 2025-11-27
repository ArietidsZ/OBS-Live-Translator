# Performance Guide

## Benchmarks

**End-to-End Latency** (Measured on reference hardware)

| Profile | Hardware | VAD | ASR | Translation | Total Latency |
|---------|----------|-----|-----|-------------|---------------|
| **Low** | Intel i7-12700H (CPU) | 0.5ms | 150ms | 180ms | ~330ms |
| **Medium** | NVIDIA RTX 3060 (TensorRT) | 0.5ms | 45ms | 80ms | ~125ms |
| **High** | NVIDIA RTX 4090 (TensorRT) | 0.5ms | 30ms | 60ms | ~90ms |

*Note: Latency includes network overhead for WebSocket transmission.*

## Optimizations

### 1. Zero-Copy Audio Buffering
- **Implementation**: `src/audio/buffer.rs`
- **Mechanism**: The circular buffer is designed to provide contiguous memory slices (`read_chunk`) whenever possible. This avoids allocating new vectors or copying data during the critical path of feeding audio to the VAD and ASR engines.

### 2. Model Caching
- **Implementation**: `src/models/session_cache.rs`
- **Benefit**: 
  - Eliminates redundant model loading times (which can take 1-5 seconds per model).
  - Reduces memory footprint by sharing immutable ONNX Runtime sessions across multiple engine instances (e.g., if multiple streams use the same language pair).

### 3. Hardware Acceleration
- **Strategy**: The system automatically selects the best available execution provider.
- **Priority**:
  1. **TensorRT** (NVIDIA GPUs) - Fastest inference.
  2. **CoreML** (Apple Silicon) - Optimized for macOS.
  3. **OpenVINO** (Intel CPUs/GPUs) - Optimized for Intel hardware.
  4. **CUDA** (NVIDIA GPUs) - General purpose GPU acceleration.
  5. **CPU** - Universal fallback.

### 4. Asynchronous Pipeline
- **Concurrency**: All major components (VAD, ASR, Translation) run in separate `tokio` tasks.
- **Non-blocking**: Audio ingestion does not block inference; inference does not block transmission.

## Profiling

To run the built-in benchmarks:

```bash
cargo run --release --bin benchmark
```

To view real-time metrics during operation:

1. Start the server: `cargo run --release --bin obs-translator-server`
2. Access metrics: `http://localhost:3000/metrics`
