# Accurate VRAM Requirements & Calculations

## Model Sizes (Based on Research)

### Whisper Models
| Model | Parameters | FP32 Size | FP16 Size | INT8 Size | Source |
|-------|-----------|-----------|-----------|-----------|---------|
| Whisper Base | 74M | 296MB | 148MB | 74MB | OpenAI |
| Whisper Small | 244M | 976MB | 488MB | 244MB | OpenAI |
| Whisper Medium | 769M | 3.1GB | 1.5GB | 769MB | OpenAI |
| Whisper Large V3 | 1550M | 6.2GB | 3.1GB | 1.5GB | OpenAI |
| **Whisper Large V3 Turbo** | 809M | 3.2GB | 1.6GB | 809MB | OpenAI (4 decoder layers) |

### NLLB Models
| Model | Parameters | FP32 Size | FP16 Size | INT8 Size | CTranslate2 INT8 |
|-------|-----------|-----------|-----------|-----------|------------------|
| NLLB-200-distilled-600M | 600M | 2.4GB | 1.2GB | 600MB | ~450MB |
| NLLB-200-distilled-1.3B | 1.3B | 5.2GB | 2.6GB | 1.3GB | ~975MB |
| NLLB-200-3.3B | 3.3B | 13.2GB | 6.6GB | 3.3GB | ~2.5GB |

### Additional Components
| Component | Memory Required | Notes |
|-----------|----------------|-------|
| Silero VAD | ~32MB | Runs on CPU |
| Audio Buffer (30s) | ~5MB | 16kHz, mono, float32 |
| KV Cache (per stream) | 50-200MB | Depends on context length |
| ONNX Runtime Overhead | ~200-500MB | Execution provider dependent |
| TensorRT Engine Cache | ~100-300MB | If using TensorRT |

## Total VRAM Calculations

### Scenario 1: 2GB VRAM Limit

**Configuration:**
- Whisper Base (INT8): 74MB
- NLLB-600M (CTranslate2 INT8): 450MB
- KV Cache (512 tokens): 50MB × 2 streams = 100MB
- Audio Buffers: 5MB × 2 = 10MB
- ONNX Runtime overhead: 200MB
- **Safety margin**: 20% overhead

**Total:** 74 + 450 + 100 + 10 + 200 = **834MB**
**With overhead:** 834 × 1.2 = **1,001MB**

✅ **Fits in 2GB with 1GB free for system**

### Scenario 2: 4GB VRAM Limit

**Configuration:**
- Whisper Small (FP16): 488MB
- NLLB-600M (FP16): 1,200MB
- KV Cache (2048 tokens): 100MB × 4 streams = 400MB
- Audio Buffers: 5MB × 4 = 20MB
- ONNX Runtime overhead: 300MB
- TensorRT Engine Cache: 200MB
- **Safety margin**: 20% overhead

**Total:** 488 + 1,200 + 400 + 20 + 300 + 200 = **2,608MB**
**With overhead:** 2,608 × 1.2 = **3,130MB**

✅ **Fits in 4GB with 870MB free**

### Scenario 3: 6GB VRAM Limit

**Configuration:**
- Whisper Large V3 Turbo (FP16): 1,600MB
- NLLB-1.3B (FP16): 2,600MB
- KV Cache (4096 tokens): 150MB × 6 streams = 900MB
- Audio Buffers: 5MB × 6 = 30MB
- ONNX Runtime overhead: 400MB
- TensorRT Engine Cache: 300MB
- **Safety margin**: 20% overhead

**Total:** 1,600 + 2,600 + 900 + 30 + 400 + 300 = **5,830MB**
**With overhead:** 5,830 × 1.2 = **6,996MB**

❌ **Exceeds 6GB - Need adjustment**

**Adjusted 6GB Configuration:**
- Whisper Large V3 Turbo (INT8): 809MB
- NLLB-1.3B (INT8): 1,300MB
- KV Cache (2048 tokens): 100MB × 6 streams = 600MB
- Audio Buffers: 5MB × 6 = 30MB
- ONNX Runtime overhead: 400MB
- TensorRT Engine Cache: 300MB

**Total:** 809 + 1,300 + 600 + 30 + 400 + 300 = **3,439MB**
**With overhead:** 3,439 × 1.2 = **4,127MB**

✅ **Fits in 6GB with 1.9GB free**

## Memory Management Strategies

### Dynamic Model Loading
For 2GB cards, we need to implement model swapping:
1. Load Whisper for ASR phase (~74MB)
2. Unload Whisper, load NLLB for translation (~450MB)
3. Keep only active model in VRAM

### KV Cache Management
```
Per-token memory = hidden_size × num_layers × 2 (K+V) × precision_bytes
For Whisper: 1024 × 32 × 2 × 2 (FP16) = 131KB per token
For 1024 tokens: ~134MB per stream
```

### Batch Size Limits
```
Available VRAM for batching = Total VRAM - Models - KV Cache - Overhead
2GB scenario: 2048 - 1001 = 1047MB free → batch_size = 1-2
4GB scenario: 4096 - 3130 = 966MB free → batch_size = 2-4
6GB scenario: 6144 - 4127 = 2017MB free → batch_size = 4-8
```

## Hybrid CPU-GPU Execution (2GB cards)

When model doesn't fit in VRAM:
1. **Split layers**: Keep attention layers on GPU, FFN layers on CPU
2. **Memory mapping**: Use unified memory on systems that support it
3. **Pinned memory**: Pre-allocate pinned CPU memory for fast transfers

## Apple Silicon (MPS) Considerations

Apple Silicon uses unified memory architecture:
- No dedicated VRAM, shares with system RAM
- MPS (Metal Performance Shaders) backend
- Recommended allocation: 25-50% of total system memory

For a 16GB M1/M2 Mac:
- Safe to allocate 4-8GB for models
- MPS automatically manages memory paging
- Use `torch.backends.mps.is_available()` to detect

## Formulas

### Total VRAM Required
```
VRAM_total = Model_weights + KV_cache + Audio_buffers + Runtime_overhead + Safety_margin

Where:
- Model_weights = Sum of all model sizes in selected precision
- KV_cache = num_streams × tokens_per_stream × memory_per_token
- Audio_buffers = num_streams × buffer_seconds × sample_rate × 4 bytes
- Runtime_overhead = ~200-500MB depending on framework
- Safety_margin = 20% of total
```

### Maximum Concurrent Streams
```
Max_streams = (VRAM_available - Model_weights - Runtime_overhead) / (KV_cache_per_stream + Audio_buffer_per_stream)
```

## Validation

These calculations are based on:
- OpenAI Whisper model cards
- Meta NLLB documentation
- CTranslate2 quantization benchmarks
- ONNX Runtime memory profiling
- Real-world deployment reports from 2024