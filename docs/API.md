# OBS Live Translator API Documentation

## Overview

The OBS Live Translator provides a comprehensive API for real-time audio translation with multiple performance profiles and deployment options.

## Core API

### Translator

The main entry point for translation services.

```rust
use obs_live_translator::{Translator, TranslatorConfig, Profile};

// Create translator with profile
let config = TranslatorConfig::from_profile(Profile::Medium);
let translator = Translator::new(config).await?;

// Process audio
let audio_data: Vec<f32> = get_audio_data();
let result = translator.process_audio(&audio_data).await?;

println!("Transcription: {}", result.transcription.unwrap_or_default());
println!("Translation: {}", result.translation.unwrap_or_default());
```

### Configuration

#### Profile-Based Configuration

```rust
// Low Profile - Optimized for resource-constrained environments
let low_config = TranslatorConfig::from_profile(Profile::Low);

// Medium Profile - Balanced performance
let medium_config = TranslatorConfig::from_profile(Profile::Medium);

// High Profile - Maximum accuracy
let high_config = TranslatorConfig::from_profile(Profile::High);
```

#### Custom Configuration

```rust
let config = TranslatorConfig {
    profile: Profile::Medium,
    audio: AudioProcessorConfig {
        sample_rate: 16000,
        frame_size_ms: 30.0,
        enable_vad: true,
        enable_noise_reduction: true,
        enable_agc: true,
    },
    asr: AsrConfig {
        model_type: "whisper_small".to_string(),
        model_path: "models/whisper_small.onnx".to_string(),
        device: "cpu".to_string(),
        batch_size: 1,
        precision: ModelPrecision::Fp16,
        enable_timestamps: true,
    },
    translation: TranslationConfig {
        model_type: "nllb".to_string(),
        model_path: "models/nllb.onnx".to_string(),
        source_language: "auto".to_string(),
        target_language: "en".to_string(),
        device: "cpu".to_string(),
        max_length: 512,
    },
};
```

## Audio Processing API

### Voice Activity Detection (VAD)

```rust
use obs_live_translator::audio::vad::{VadEngine, VadConfig};

let config = VadConfig {
    engine_type: "webrtc".to_string(),
    threshold: 0.5,
    min_voice_duration_ms: 100,
    min_silence_duration_ms: 200,
};

let mut vad = VadEngine::new(config).await?;
let result = vad.process_frame(&audio_frame).await?;

if result.is_voice {
    println!("Speech detected with confidence: {}", result.confidence);
}
```

### Audio Resampling

```rust
use obs_live_translator::audio::resampling::{ResamplerEngine, ResamplerConfig};

let config = ResamplerConfig {
    input_sample_rate: 48000,
    output_sample_rate: 16000,
    quality: ResamplingQuality::High,
    channels: 1,
};

let mut resampler = ResamplerEngine::new(config).await?;
let resampled = resampler.resample(&input_audio).await?;
```

## Machine Learning Inference API

### ASR Engine

```rust
use obs_live_translator::asr::{AsrEngine, AsrConfig, ModelPrecision};

let config = AsrConfig {
    model_type: "whisper_small".to_string(),
    model_path: "models/whisper_small.onnx".to_string(),
    device: "cuda".to_string(), // Use GPU if available
    batch_size: 4,
    precision: ModelPrecision::Fp16,
    enable_timestamps: true,
};

let mut asr = AsrEngine::new(config).await?;
let transcription = asr.transcribe(&audio).await?;

println!("Text: {}", transcription.text.unwrap_or_default());
println!("Confidence: {}", transcription.confidence);
```

### Translation Engine

```rust
use obs_live_translator::translation::{TranslationEngine, TranslationConfig};

let config = TranslationConfig {
    model_type: "nllb".to_string(),
    model_path: "models/nllb.onnx".to_string(),
    source_language: "es".to_string(),
    target_language: "en".to_string(),
    device: "cpu".to_string(),
    max_length: 512,
};

let mut translator = TranslationEngine::new(config).await?;
let result = translator.translate("Hola, ¿cómo estás?").await?;

println!("Translation: {}", result.translated_text);
```

## Streaming API

### WebSocket Server

```rust
use obs_live_translator::streaming::{StreamingServer, StreamingConfig};

let config = StreamingConfig {
    buffer_size: 8192,
    max_latency_ms: 200,
    reconnect_attempts: 3,
    heartbeat_interval_ms: 30000,
};

let server = StreamingServer::new("127.0.0.1:8080", config).await?;
server.run().await?;
```

### WebSocket Client

```rust
use obs_live_translator::streaming::WebSocketClient;

let client = WebSocketClient::connect("ws://localhost:8080").await?;

// Send audio data
client.send_audio_chunk(&audio_chunk).await?;

// Receive translation
let response = client.receive_translation().await?;
println!("Received: {}", response.text);
```

## Performance Monitoring API

### Performance Monitor

```rust
use obs_live_translator::monitoring::{PerformanceMonitor, MonitorConfig};

let config = MonitorConfig {
    sample_interval_ms: 100,
    history_size: 1000,
    enable_detailed_metrics: true,
};

let monitor = PerformanceMonitor::new(config).await?;
monitor.start().await?;

// Get performance snapshot
let snapshot = monitor.get_performance_snapshot().await?;
println!("CPU Usage: {:.1}%", snapshot.current_metrics.resources.cpu_utilization_percent);
println!("Memory: {:.0} MB", snapshot.current_metrics.resources.memory_usage_mb);
println!("Avg Latency: {:.2}ms", snapshot.current_metrics.latency.avg_latency_ms);

monitor.stop().await?;
```

### Metrics Collection

```rust
use obs_live_translator::monitoring::MetricsCollector;

let collector = MetricsCollector::new(Default::default()).await?;

// Track component operation
collector.start_component_operation("translation", "translate_text").await;
// ... perform operation ...
collector.end_component_operation("translation", "translate_text", true).await;

// Get metrics
let metrics = collector.get_current_metrics().await;
println!("Operations/sec: {:.0}", metrics.throughput.operations_per_second);
```

## Error Handling

All API methods return `Result<T, Error>` types. Handle errors appropriately:

```rust
use obs_live_translator::{Translator, TranslatorConfig, Error};

match Translator::new(config).await {
    Ok(translator) => {
        // Use translator
    },
    Err(Error::ModelLoadError(msg)) => {
        eprintln!("Failed to load model: {}", msg);
    },
    Err(Error::ConfigError(msg)) => {
        eprintln!("Configuration error: {}", msg);
    },
    Err(e) => {
        eprintln!("Unexpected error: {}", e);
    }
}
```

## Async/Await Support

All API methods are async and should be called within an async context:

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TranslatorConfig::from_profile(Profile::Medium);
    let translator = Translator::new(config).await?;
    
    // Process audio asynchronously
    let result = translator.process_audio(&audio_data).await?;
    
    Ok(())
}
```

## Thread Safety

Most components are thread-safe and can be shared across threads:

```rust
use std::sync::Arc;
use tokio::sync::Mutex;

let translator = Arc::new(Translator::new(config).await?);

// Share translator across multiple tasks
let translator_clone = Arc::clone(&translator);
tokio::spawn(async move {
    let result = translator_clone.process_audio(&audio).await;
    // Handle result
});
```

## Best Practices

1. **Profile Selection**: Choose the appropriate profile based on your hardware and latency requirements
2. **Resource Management**: Properly dispose of resources using drop or explicit cleanup methods
3. **Error Handling**: Always handle errors gracefully, especially for model loading and network operations
4. **Monitoring**: Use the performance monitoring API to track system health in production
5. **Batching**: When processing multiple audio streams, use batching for better throughput
6. **Configuration**: Store configuration in external files for easy deployment changes

## Examples

See the `examples/` directory for complete working examples:

- `basic_translation.rs` - Simple audio translation
- `streaming_server.rs` - WebSocket streaming server
- `batch_processing.rs` - Batch audio processing
- `performance_monitoring.rs` - Performance tracking
- `custom_pipeline.rs` - Custom processing pipeline