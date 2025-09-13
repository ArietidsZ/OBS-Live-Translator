# GPU Performance Analysis & Realistic Latency Estimates

## Modern GPU Specifications (2024)

### High-End GPUs (82-52 TFLOPS)
| GPU | TFLOPS | VRAM | Bandwidth | Whisper RTF* | Est. Latency |
|-----|--------|------|-----------|--------------|--------------|
| RTX 4090 | 82.6 | 24GB | 1010 GB/s | 0.052 (large) | 15-20ms per chunk |
| RTX 4080 Super | 52.3 | 16GB | 736 GB/s | 0.065 | 20-25ms |
| RTX 4080 | 48.7 | 16GB | 717 GB/s | 0.070 | 22-28ms |
| AMD RX 7900 XTX | ~61 | 24GB | 960 GB/s | 0.060 | 18-24ms |
| AMD RX 7900 XT | ~51 | 20GB | 800 GB/s | 0.075 | 23-30ms |

### Mid-Range GPUs (29-40 TFLOPS)
| GPU | TFLOPS | VRAM | Bandwidth | Whisper RTF* | Est. Latency |
|-----|--------|------|-----------|--------------|--------------|
| RTX 4070 Ti | 40.1 | 12GB | 504 GB/s | 0.090 | 28-35ms |
| AMD RX 7800 XT | 32.6 | 16GB | 624 GB/s | 0.095 | 30-38ms |
| RTX 4070 | 29.1 | 12GB | 504 GB/s | 0.110 | 35-45ms |
| Apple M3 Max | ~30 | 48GB† | 400 GB/s | 0.100 | 32-40ms |

### Entry-Level GPUs (15-22 TFLOPS)
| GPU | TFLOPS | VRAM | Bandwidth | Whisper RTF* | Est. Latency |
|-----|--------|------|-----------|--------------|--------------|
| RTX 4060 Ti | 22.0 | 8/16GB | 288 GB/s | 0.150 | 48-60ms |
| RTX 4060 | 15.0 | 8GB | 272 GB/s | 0.200 | 65-80ms |
| AMD RX 7600 | ~21 | 8GB | 288 GB/s | 0.180 | 58-72ms |
| Intel Arc A770 | ~19 | 8/16GB | 512 GB/s | 0.170 | 55-68ms |

### Integrated GPUs
| GPU | TFLOPS | Memory | Bandwidth | Whisper RTF* | Est. Latency |
|-----|--------|--------|-----------|--------------|--------------|
| Apple M3 Pro | ~18 | 36GB† | 150 GB/s | 0.250 | 80-100ms |
| Apple M2 Pro | ~13 | 32GB† | 200 GB/s | 0.300 | 95-120ms |
| Intel Iris Xe | 2.4 | Shared | 68 GB/s | 1.500 | 500-600ms |
| AMD RDNA2 iGPU | 3.4 | Shared | 88 GB/s | 1.200 | 400-500ms |

*RTF = Real-Time Factor (lower is better, 0.1 = 10x faster than real-time)
†Unified memory shared with CPU

## Realistic Latency Breakdown

### Processing Pipeline Components

#### 1. Audio Capture & Preprocessing
- **Buffer filling**: 20-30ms (for 30ms chunks @ 16kHz)
- **VAD processing**: 2-5ms
- **Resampling/normalization**: 1-2ms

#### 2. ASR (Whisper) Inference
Based on GPU tier and model size:

**Whisper Large V3 Turbo (4 decoder layers)**
- RTX 4090: 15-20ms per 30ms chunk
- RTX 4080: 22-28ms
- RTX 4070: 35-45ms
- RTX 4060: 65-80ms

**Whisper Base (for low VRAM)**
- RTX 4090: 5-8ms
- RTX 4080: 7-10ms
- RTX 4070: 10-15ms
- RTX 4060: 18-25ms

#### 3. Translation (NLLB)
- NLLB-1.3B on RTX 4090: 8-12ms
- NLLB-1.3B on RTX 4070: 15-20ms
- NLLB-600M on RTX 4060: 20-30ms
- CTranslate2 INT8: 40-60% faster

#### 4. Post-processing & Display
- Text formatting: 1-2ms
- WebSocket transmission: 1-3ms
- Browser rendering: 5-10ms

### Total End-to-End Latency Estimates

#### High-End Setup (RTX 4090/4080, 16-24GB VRAM)
- **Best case**: 45ms (Whisper Turbo + NLLB-1.3B FP16)
- **Average**: 55-65ms
- **Worst case**: 80ms (complex audio, multiple speakers)

#### Mid-Range Setup (RTX 4070/RX 7800 XT, 12-16GB VRAM)
- **Best case**: 70ms (Whisper Small + NLLB-600M)
- **Average**: 85-100ms
- **Worst case**: 130ms

#### Entry-Level Setup (RTX 4060/RX 7600, 8GB VRAM)
- **Best case**: 110ms (Whisper Base + NLLB-600M INT8)
- **Average**: 140-160ms
- **Worst case**: 200ms

#### Integrated Graphics (Apple Silicon/Intel Iris)
- **Apple M3 Pro**: 120-150ms (optimized with MPS)
- **Apple M2**: 180-220ms
- **Intel Iris Xe**: 600-800ms (not recommended)

## Optimization Strategy: Accuracy Over Streams

### When More VRAM is Available

Instead of increasing concurrent streams, use extra VRAM for:

#### 1. Larger, More Accurate Models
```toml
# 8GB VRAM - Basic accuracy
whisper_model = "whisper-base"  # 74MB, WER ~10%
nllb_model = "nllb-600m"         # 450MB, BLEU ~28

# 16GB VRAM - Enhanced accuracy
whisper_model = "whisper-large-v3-turbo"  # 1.6GB, WER ~4%
nllb_model = "nllb-1.3b"                  # 2.6GB, BLEU ~32

# 24GB VRAM - Maximum accuracy
whisper_model = "whisper-large-v3"  # 3.1GB, WER ~2.5%
nllb_model = "nllb-3.3b"            # 6.6GB, BLEU ~35
```

#### 2. Extended Context Windows
```toml
# More VRAM = Longer context for better understanding
[vram_8gb]
max_context_tokens = 512   # ~50MB
chunk_duration_ms = 500

[vram_16gb]
max_context_tokens = 2048  # ~200MB
chunk_duration_ms = 1000

[vram_24gb]
max_context_tokens = 4096  # ~400MB
chunk_duration_ms = 2000
```

#### 3. Enhanced Processing Features
```toml
[accuracy_features]
# Enable with more VRAM
enable_speaker_diarization = true     # +500MB
enable_punctuation_restoration = true  # +200MB
enable_emotion_detection = true        # +300MB
enable_noise_suppression = true        # +150MB
enable_multi_language_detection = true # +400MB
```

#### 4. Quality-Focused Batching
```toml
# Single high-quality stream instead of multiple
[vram_24gb]
max_concurrent_streams = 1  # Focus on ONE stream
batch_size = 1              # No batching delays
use_beam_search = true      # Better accuracy (+30% time)
beam_size = 5               # More search paths
temperature = 0.0           # Deterministic output
```

## Real-World Performance Benchmarks

### RTX 4090 (Actual Measurements)
- Whisper Large: 186s for 1 hour audio (RTF: 0.052)
- Whisper Tiny: 34s for 1 hour audio (RTF: 0.009)
- With optimization: 8s for 10 minutes (RTF: 0.013)
- Translation throughput: >3000 words/minute

### RTX 4080 (Actual Measurements)
- 28.5% slower than RTX 4090
- Translation throughput: ~2900 words/minute
- Batch processing: 15s for 50x30s clips (batch_size≥16)

### Apple M3 Pro (Metal Performance Shaders)
- 65% faster than M1 for graphics
- Cinebench GPU: 5,512 (14-core GPU)
- Limited to 150 GB/s memory bandwidth
- Best with smaller models due to bandwidth constraints

## Recommended Configurations by GPU

### RTX 4090 (24GB) - Maximum Accuracy
```toml
whisper_model = "whisper-large-v3"     # Full model
nllb_model = "nllb-3.3b"               # Largest model
max_concurrent_streams = 1             # Single stream focus
use_beam_search = true
beam_size = 5
chunk_duration_ms = 2000               # Longer context
max_context_tokens = 8192              # Maximum context
```

### RTX 4070 (12GB) - Balanced Accuracy
```toml
whisper_model = "whisper-large-v3-turbo"  # Optimized large
nllb_model = "nllb-1.3b"                  # Good quality
max_concurrent_streams = 1                # Single stream
use_beam_search = true
beam_size = 3
chunk_duration_ms = 1000
max_context_tokens = 2048
```

### RTX 4060 (8GB) - Optimized Performance
```toml
whisper_model = "whisper-small"        # Balanced size
nllb_model = "nllb-600m"               # Efficient
max_concurrent_streams = 1             # Single stream
use_beam_search = false               # Speed priority
chunk_duration_ms = 500
max_context_tokens = 1024
```

### Apple M3 Pro - MPS Optimized
```toml
whisper_model = "whisper-base.en"     # English-optimized
nllb_model = "nllb-600m"              # Memory efficient
execution_provider = "MPSExecutionProvider"
max_concurrent_streams = 1
use_coreml_if_available = true
chunk_duration_ms = 750
```

## Key Insights

1. **TFLOPS ≠ Direct Performance**: Memory bandwidth often more critical
   - RTX 4090: 1010 GB/s enables smooth large model inference
   - RTX 4060: 272 GB/s creates bottleneck despite 15 TFLOPS

2. **Diminishing Returns**: Beyond RTX 4070, latency improvements are marginal
   - 4090 vs 4080: Only 20-30% faster despite 70% more TFLOPS
   - Better to invest in model quality than raw compute

3. **Memory Bandwidth Critical**:
   - Apple M3 Pro has 18 TFLOPS but only 150 GB/s bandwidth
   - Intel Arc A770 has good bandwidth (512 GB/s) despite lower TFLOPS

4. **Real-World vs Theoretical**:
   - Actual Whisper RTF: 0.052-0.2 (not the <0.01 often claimed)
   - Network/display adds 10-20ms unavoidable latency
   - Total system latency rarely below 50ms

## Conclusion

For OBS Live Translation, prioritize:
1. **Model Quality** over stream count when VRAM available
2. **Single Stream Excellence** rather than multi-stream mediocrity
3. **Realistic Expectations**: 50-150ms total latency is excellent
4. **Hardware Sweet Spot**: RTX 4070/RX 7800 XT offers best value

The focus should be on delivering ONE highly accurate, low-latency translation stream rather than multiple lower-quality streams.