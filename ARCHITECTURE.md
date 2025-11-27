# System Architecture

## Overview

OBS Live Translator v5.0 is designed as a high-performance, low-latency real-time speech translation system. It employs a cascaded architecture where audio flows through a pipeline of specialized components, each optimized for specific hardware profiles.

## Core Pipeline

```mermaid
graph LR
    Audio[Audio Input] --> VAD[Voice Activity Detection]
    VAD --> Buffer[Circular Audio Buffer]
    Buffer --> ASR[Automatic Speech Recognition]
    ASR --> LangDetect[Language Detection]
    LangDetect --> Translation[Neural Machine Translation]
    Translation --> Output[WebSocket/WebTransport Output]
```

## Key Components

### 1. Audio Processing & Buffering
- **Circular Buffer**: Implemented in `src/audio/buffer.rs`. Uses a fixed-size `Vec<f32>` with read/write pointers to handle streaming audio efficiently.
- **Zero-Copy Optimization**: `read_chunk` attempts to return contiguous slices to minimize memory copying.
- **Resampling**: `rubato` is used for high-quality asynchronous resampling to 16kHz (required by most models).

### 2. Voice Activity Detection (VAD)
- **Engine**: Silero VAD v5 (default), with support for TEN VAD and Cobra.
- **Strategy**: 
  - Detects speech segments with high precision (0.5ms latency).
  - Filters out silence and noise to reduce load on ASR.

### 3. Automatic Speech Recognition (ASR)
- **Profiles**:
  - **Low**: Distil-Whisper Large v3 (INT8) - CPU optimized.
  - **Medium**: Parakeet TDT (FP16/INT8) - GPU optimized.
  - **High**: Canary Qwen (BF16/FP8) - High-end GPU.
- **Inference**: Powered by ONNX Runtime with platform-specific execution providers (TensorRT, CoreML, OpenVINO).

### 4. Language Detection
- **Hybrid Strategy**: Combines text-based detection (CLD3/FastText) with ASR-provided language tokens.
- **Fallback**: Uses `lingua` or `whatlang` if primary detection fails.

### 5. Translation
- **Engines**:
  - **MADLAD-400**: Massive multilingual support (419 languages).
  - **NLLB-200**: High-quality translation for low-resource languages.
  - **Claude 3.5**: Optional cloud-based fallback for premium quality.

## Advanced Features

### Model Caching & Lazy Loading
- **SessionCache**: `src/models/session_cache.rs` implements an in-memory cache (`HashMap<PathBuf, Arc<Session>>`) for ONNX Runtime sessions.
- **Mechanism**: Models are loaded only when first requested. Sessions are shared across engine instances to save memory.

### Error Handling & Recovery
- **Graceful Degradation**: `ExecutionProviderConfig` implements a fallback chain (e.g., TensorRT -> CUDA -> CPU).
- **Retry Logic**: Transient failures in inference are handled with exponential backoff retries.

### Monitoring & Metrics
- **MetricsCollector**: `src/monitoring/mod.rs` collects latency histograms and counters.
- **Prometheus Export**: Server exposes `/metrics` endpoint for real-time monitoring via Grafana/Prometheus.

## Directory Structure

```
src/
├── asr/                # ASR engine implementations
├── audio/              # Audio processing & buffering
├── bin/                # Binary entry points (server, tools)
├── config/             # Configuration management
├── execution_provider/ # Hardware acceleration logic
├── inference/          # Generic ONNX inference wrapper
├── language_detection/ # Language ID modules
├── models/             # Model management & caching
├── monitoring/         # Metrics & telemetry
├── translation/        # Translation engines
├── vad/                # Voice Activity Detection
└── lib.rs              # Library root
```
