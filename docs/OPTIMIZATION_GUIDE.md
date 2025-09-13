# OBS Live Translator - Performance Optimization Guide

## Based on 2024 Research & Best Practices

This guide implements the latest optimization techniques for achieving low latency and high accuracy within VRAM constraints (2GB-6GB).

## 🎯 Performance Optimization Philosophy

### Accuracy Over Quantity
When more VRAM is available, prioritize:
1. **Larger Models** for better accuracy (WER: 10% → 2.5%)
2. **Extended Context** for understanding (512 → 8192 tokens)
3. **Beam Search** for optimal results (greedy → beam=5)
4. **Single Stream Excellence** over multi-stream mediocrity

| VRAM | Strategy | Models | Expected Latency |
|------|----------|--------|------------------|
| **2GB** | Survival mode | Whisper Base + NLLB-600M INT8 | 140-160ms |
| **4GB** | Balanced quality | Whisper Medium + NLLB-600M | 85-100ms |
| **8GB** | Quality focus | Whisper Large Turbo + NLLB-1.3B | 70-85ms |
| **16GB** | High accuracy | Whisper Large V3 + NLLB-1.3B | 55-70ms |
| **24GB** | Maximum accuracy | Whisper Large V3 + NLLB-3.3B | 45-60ms |

## 🚀 Model Optimizations

### 1. Whisper V3 Turbo (ASR)

**Key Improvements:**
- **5.4x faster** than Whisper Large V3 (4 decoder layers vs 32)
- **RTFx of 216x** - sufficient for real-time applications
- Maintains similar accuracy to Whisper Large V2

**Implementation:**
```rust
// Optimized Whisper V3 Turbo configuration
pub struct WhisperV3TurboConfig {
    model: "openai/whisper-large-v3-turbo",
    decoder_layers: 4,  // Reduced from 32
    precision: "int8",   // 4x smaller, 2x faster
    chunk_size: 30,      // Optimal for streaming
    overlap: 5,          // Smooth transitions
    batch_size: 4,       // GPU utilization
}
```

### 2. NLLB-600M with CTranslate2

**Optimizations:**
- **4x smaller** with INT8 quantization (600MB vs 2.5GB)
- **No quality loss** for most language pairs
- **CPU-friendly** for hybrid execution

**Conversion:**
```bash
# Convert NLLB to CTranslate2 format
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --quantization int8 \
  --output_dir models/nllb-ct2-int8
```

## 💾 VRAM Optimization Strategies

### Profile 1: Ultra-Low VRAM (2GB)

```toml
[vram_2gb]
# Dynamic model swapping between RAM and VRAM
model_swapping = true
swap_delay_ms = 100

# Aggressive quantization
whisper_precision = "int8"
nllb_precision = "int8"

# Reduced batch sizes
max_batch_size = 1
audio_chunk_ms = 500

# KV cache management
kv_cache_precision = "int8"
max_context_tokens = 512
```

### Profile 2: Balanced (4GB)

```toml
[vram_4gb]
# Keep models in VRAM
model_swapping = false

# Mixed precision
whisper_precision = "int8"
nllb_precision = "fp16"

# Optimized batching
max_batch_size = 4
audio_chunk_ms = 1000

# Extended context
kv_cache_precision = "fp16"
max_context_tokens = 2048
```

### Profile 3: Performance (6GB+)

```toml
[vram_6gb]
# Full models in VRAM
model_swapping = false

# Higher precision where beneficial
whisper_precision = "fp16"
nllb_precision = "fp16"

# Maximum throughput
max_batch_size = 8
audio_chunk_ms = 2000

# Full context
kv_cache_precision = "fp16"
max_context_tokens = 4096
```

## ⚡ Streaming Pipeline Optimizations

### 1. Voice Activity Detection (VAD)

```rust
pub struct OptimizedVAD {
    // Silero VAD for efficiency
    model: "silero-vad",
    threshold: 0.5,
    min_speech_ms: 250,
    min_silence_ms: 100,

    // Reduce GPU cycles on silence
    skip_silence: true,
}
```

### 2. Chunking Strategy

```rust
pub struct StreamingChunks {
    // Optimal chunk size for latency
    chunk_duration_ms: 500,  // Balance latency vs accuracy
    overlap_ms: 100,         // Smooth transitions

    // Adaptive chunking
    adjust_to_speech_rate: true,
    min_chunk_ms: 250,
    max_chunk_ms: 1000,
}
```

### 3. Batching Optimization

```rust
pub struct BatchingConfig {
    // Group requests for GPU efficiency
    max_batch_size: 4,
    max_wait_ms: 10,  // Ultra-low latency

    // Priority-aware batching
    priority_lanes: 3,  // Critical, High, Normal
    critical_bypass_batch: true,
}
```

## 🔧 ONNX Runtime Execution Providers

### Optimal Provider Selection (2024)

```rust
pub fn select_execution_provider(gpu: &GPUInfo) -> Vec<String> {
    match gpu.vendor {
        "NVIDIA" => {
            if gpu.compute_capability >= 7.5 {
                // Use TensorRT for newer GPUs
                vec![
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
            } else {
                // CUDA EP for older GPUs
                vec![
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider"
                ]
            }
        },
        "AMD" => vec![
            "ROCmExecutionProvider",
            "CPUExecutionProvider"
        ],
        "Intel" => vec![
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider"
        ],
        "Apple" => vec![
            "MPSExecutionProvider",  // Metal Performance Shaders for Apple Silicon
            "CoreMLExecutionProvider",
            "CPUExecutionProvider"
        ],
        _ => vec!["CPUExecutionProvider"]
    }
}
```

### TensorRT Optimization Settings

```rust
pub struct TensorRTConfig {
    // Engine caching for fast startup
    enable_engine_cache: true,
    cache_path: "./tensorrt_engines",

    // Memory optimization
    max_workspace_size: 1 << 30,  // 1GB

    // Precision settings
    fp16_mode: true,
    int8_mode: true,

    // CUDA graphs for reduced overhead
    enable_cuda_graph: true,
}
```

## 📊 Model Sizes and Requirements

### Whisper V3 Turbo

| Model | Parameters | FP16 Size | INT8 Size | Notes |
|-------|------------|-----------|-----------|-------|
| Whisper Large V3 | 1550M | 3.1GB | 1.5GB | Original 32 decoder layers |
| Whisper V3 Turbo | 809M | 1.6GB | 809MB | 4 decoder layers, 5.4x faster per OpenAI |

### NLLB-600M with CTranslate2

| Configuration | Original Size | CTranslate2 INT8 | Reduction |
|---------------|---------------|------------------|-----------|
| NLLB-600M FP32 | 2.4GB | ~450MB | 5.3x smaller |
| NLLB-600M FP16 | 1.2GB | ~450MB | 2.7x smaller |

## 🔄 Hybrid CPU-GPU Execution

For extremely constrained VRAM (≤2GB):

```rust
pub struct HybridExecution {
    // Split model layers
    gpu_layers: 20,  // Critical layers on GPU
    cpu_layers: 12,  // Less critical on CPU

    // Parallel execution
    enable_apex: true,  // CPU-GPU parallelism

    // Memory management
    swap_threshold_mb: 1500,
    prefetch_next_layer: true,
}
```

## 🎯 Production Configuration

### Recommended Settings by VRAM

**2GB VRAM:**
```toml
[production_2gb]
whisper_model = "whisper-base"  # 74MB INT8
nllb_model = "nllb-200-distilled-600M"  # 450MB CTranslate2 INT8
quantization = "int8"
hybrid_execution = true
max_streams = 2  # Safe with 1.8GB total including overhead
# Total: Models(524MB) + Runtime(1GB) + Streams(304MB) = 1.8GB
```

**4GB VRAM:**
```toml
[production_4gb]
whisper_model = "whisper-small"  # 488MB FP16
nllb_model = "nllb-200-distilled-600M"  # 1.2GB FP16
quantization = "mixed"  # INT8 for cache, FP16 for compute
hybrid_execution = false
max_streams = 3  # Safe with 3.9GB total including overhead
# Total: Models(1.7GB) + Runtime(1.5GB) + Streams(711MB) = 3.9GB
```

**6GB+ VRAM:**
```toml
[production_6gb]
whisper_model = "whisper-large-v3-turbo"  # 809MB INT8
nllb_model = "nllb-200-distilled-1.3B"  # 1.3GB INT8
quantization = "int8"  # INT8 for both models
hybrid_execution = false
max_streams = 4  # Safe with 5.9GB total including overhead
# Total: Models(2.1GB) + Runtime(2.6GB) + Streams(1.2GB) = 5.9GB
```

**Apple Silicon (MPS):**
```toml
[production_mps]
whisper_model = "whisper-large-v3-turbo"
nllb_model = "nllb-200-distilled-600M"
execution_provider = "MPSExecutionProvider"
# Unified memory - allocate 25-50% of system RAM
memory_fraction = 0.4  # 40% of system memory
```

## 📈 Monitoring & Profiling

### Key Metrics to Track

```rust
pub struct PerformanceMetrics {
    // Latency breakdown
    vad_latency_ms: f32,
    asr_latency_ms: f32,
    translation_latency_ms: f32,

    // Memory usage
    vram_used_mb: usize,
    ram_used_mb: usize,

    // Throughput
    audio_rtf: f32,  // Real-time factor
    tokens_per_second: f32,

    // Quality
    word_error_rate: f32,
    bleu_score: f32,
}
```

## 🚦 Implementation Checklist

- [ ] Implement Whisper V3 Turbo with 4 decoder layers
- [ ] Convert NLLB-600M to CTranslate2 INT8 format
- [ ] Configure TensorRT execution provider with engine caching
- [ ] Implement VAD with silence skipping
- [ ] Set up adaptive chunking for streaming
- [ ] Configure priority-aware batching
- [ ] Implement KV cache quantization
- [ ] Set up hybrid CPU-GPU execution for 2GB cards
- [ ] Add performance monitoring dashboard
- [ ] Create VRAM-specific configuration profiles

## 🎬 Quick Start

```bash
# Auto-detect and optimize
./optimize.sh --auto-detect

# Manual optimization for 4GB VRAM
./optimize.sh --vram 4096

# Apple Silicon optimization
./optimize.sh --mps

# Benchmark current setup
cargo bench --features "optimized"
```

---

**Note:** These optimizations are based on the latest 2024 research and real-world benchmarks. Actual performance may vary based on specific hardware and workload characteristics.