# VRAM Usage Guide for OBS Live Translator

> Comprehensive memory requirements and optimization strategies

## Critical Components for Full Process

### 1. Model Weights (Static)
These are loaded once and remain in memory:
- **Whisper Model**: Varies by choice
- **NLLB Model**: Varies by choice
- **Silero VAD**: 32MB (usually on CPU)

### 2. Runtime Overhead (Dynamic)
Memory that varies during execution:

#### ONNX Runtime Overhead
- **Session initialization**: 150-300MB
- **Graph optimization**: 50-100MB
- **Kernel workspace**: 100-200MB
- **Total ONNX overhead**: 300-600MB

#### TensorRT Overhead (if enabled)
- **Engine cache**: 200-400MB
- **Workspace allocation**: 512MB-1GB
- **CUDA context**: 100-200MB
- **Total TensorRT overhead**: 812MB-1.6GB

### 3. Inference Memory (Per-Stream)
Memory needed during actual inference:

#### KV Cache (Attention Memory)
- **Whisper Base**: ~50MB per stream (512 tokens)
- **Whisper Small**: ~75MB per stream (1024 tokens)
- **Whisper V3 Turbo**: ~100MB per stream (2048 tokens)

#### Audio Buffers
- **Input buffer**: 5MB per stream (30s @ 16kHz)
- **Processing buffer**: 5MB per stream
- **Output buffer**: 2MB per stream
- **Total per stream**: 12MB

#### Intermediate Tensors
- **Feature extraction**: 20-50MB
- **Attention maps**: 30-80MB
- **Decoder states**: 40-100MB
- **Total per inference**: 90-230MB

### 4. System Reserved
GPU system requirements:
- **CUDA context**: 200-400MB
- **Driver overhead**: 100-200MB
- **Display compositor**: 200-300MB (if GPU drives display)
- **Total system reserved**: 500-900MB

## Complete VRAM Calculations

### 2GB VRAM Scenario (Ultra-Conservative)

**Models:**
- Whisper Base INT8: 74MB
- NLLB-600M CTranslate2 INT8: 450MB
- **Model Total**: 524MB

**Runtime Overhead:**
- ONNX Runtime: 300MB
- CUDA context: 200MB
- System reserved: 500MB
- **Runtime Total**: 1,000MB

**Per-Stream Requirements:**
- KV Cache: 50MB
- Audio buffers: 12MB
- Intermediate tensors: 90MB
- **Per-Stream Total**: 152MB

**Maximum Configuration:**
- Models + Runtime: 1,524MB
- Available for streams: 524MB (2048 - 1524)
- **Max streams**: 3 (524 / 152 ≈ 3)
- **Actual safe streams**: 2 (with buffer)

**Total with 2 streams**: 1,524 + (152 × 2) = **1,828MB**
**Safety margin (10%)**: 1,828 × 1.1 = **2,011MB**

⚠️ **TIGHT FIT - Requires careful management**

### 4GB VRAM Scenario (Balanced)

**Models:**
- Whisper Small FP16: 488MB
- NLLB-600M FP16: 1,200MB
- **Model Total**: 1,688MB

**Runtime Overhead:**
- ONNX Runtime: 400MB
- TensorRT cache: 200MB
- CUDA context: 300MB
- System reserved: 600MB
- **Runtime Total**: 1,500MB

**Per-Stream Requirements:**
- KV Cache: 75MB
- Audio buffers: 12MB
- Intermediate tensors: 150MB
- **Per-Stream Total**: 237MB

**Maximum Configuration:**
- Models + Runtime: 3,188MB
- Available for streams: 908MB (4096 - 3188)
- **Max streams**: 3 (908 / 237 ≈ 3.8)
- **Actual safe streams**: 3

**Total with 3 streams**: 3,188 + (237 × 3) = **3,899MB**
**Safety margin (5%)**: 3,899 × 1.05 = **4,094MB**

✅ **FITS in 4GB with minimal buffer**

### 6GB VRAM Scenario (Performance)

**Models:**
- Whisper V3 Turbo INT8: 809MB
- NLLB-1.3B INT8: 1,300MB
- **Model Total**: 2,109MB

**Runtime Overhead:**
- ONNX Runtime: 500MB
- TensorRT workspace: 1,000MB
- CUDA context: 400MB
- System reserved: 700MB
- **Runtime Total**: 2,600MB

**Per-Stream Requirements:**
- KV Cache: 100MB
- Audio buffers: 12MB
- Intermediate tensors: 200MB
- **Per-Stream Total**: 312MB

**Maximum Configuration:**
- Models + Runtime: 4,709MB
- Available for streams: 1,435MB (6144 - 4709)
- **Max streams**: 4 (1435 / 312 ≈ 4.6)
- **Actual safe streams**: 4

**Total with 4 streams**: 4,709 + (312 × 4) = **5,957MB**
**Safety margin (3%)**: 5,957 × 1.03 = **6,136MB**

✅ **FITS in 6GB but at limit**

## Revised Safe Configurations

### 2GB VRAM Profile
```toml
[vram_2gb_safe]
# Models: 524MB total
whisper_model = "whisper-base"
whisper_precision = "int8"
nllb_model = "nllb-600m-ctranslate2"
nllb_precision = "int8"

# Conservative settings
max_concurrent_streams = 2  # Reduced from 3
kv_cache_tokens = 512       # Minimum viable
enable_model_swapping = true # Critical for 2GB
tensorrt_enabled = false    # Save memory

# Total usage: ~1.8GB (leaves 200MB buffer)
```

### 4GB VRAM Profile
```toml
[vram_4gb_safe]
# Models: 1,688MB total
whisper_model = "whisper-small"
whisper_precision = "fp16"
nllb_model = "nllb-600m"
nllb_precision = "fp16"

# Balanced settings
max_concurrent_streams = 3  # Reduced from 4
kv_cache_tokens = 1024
enable_model_swapping = false
tensorrt_enabled = true
tensorrt_workspace_mb = 256  # Limited workspace

# Total usage: ~3.9GB (leaves 100MB buffer)
```

### 6GB VRAM Profile
```toml
[vram_6gb_safe]
# Models: 2,109MB total
whisper_model = "whisper-v3-turbo"
whisper_precision = "int8"
nllb_model = "nllb-1.3b"
nllb_precision = "int8"

# Performance settings
max_concurrent_streams = 4  # Reduced from 6
kv_cache_tokens = 2048
enable_model_swapping = false
tensorrt_enabled = true
tensorrt_workspace_mb = 1024

# Total usage: ~5.9GB (leaves 200MB buffer)
```

## Critical Insights

1. **TensorRT adds significant overhead** (800MB-1.6GB) but provides speed benefits
2. **System reserved memory** (500-900MB) is often overlooked but critical
3. **Intermediate tensors** during inference can consume 90-230MB per stream
4. **Safety margins are essential** - GPU OOM errors can crash the entire system

## Recommendations

### For 2GB Cards
- **DO**: Use model swapping, INT8 quantization, minimize streams
- **DON'T**: Enable TensorRT, use FP16, run more than 2 streams
- **RISK**: Very tight fit, consider CPU-only fallback

### For 4GB Cards
- **DO**: Use balanced settings, 3 streams max, limited TensorRT
- **DON'T**: Use large models, exceed 3 streams
- **SAFE**: Reasonable headroom with proper configuration

### For 6GB+ Cards
- **DO**: Use INT8 for models to leave room for inference
- **DON'T**: Assume unlimited streams, use FP16 for both models
- **NOTE**: 4 streams is safe maximum, not 6 as previously stated

## Memory Profiling Commands

```bash
# NVIDIA GPUs
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 1

# Monitor during inference
watch -n 0.5 nvidia-smi

# Detailed memory breakdown
nvidia-smi -q -d MEMORY

# Apple Silicon
sudo powermetrics --samplers gpu_power -i 1000
```

## Final Notes

These calculations include ALL overhead for the complete process. Previous calculations only considered model weights and basic overhead, missing:
- TensorRT workspace allocation
- Intermediate inference tensors
- System reserved memory
- CUDA context overhead
- Safety margins for stability

The revised configurations are conservative but ensure stable operation without OOM errors.