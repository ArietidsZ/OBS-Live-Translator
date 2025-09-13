//! High-performance OBS Live Translator server

use anyhow::Result;
use obs_live_translator_core::{HighPerformanceTranslator, TranslatorConfig};
use std::env;
use tracing::{info, error};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    if args.len() > 1 && args[1] == "--benchmark" {
        return run_benchmark().await;
    }
    
    info!("🚀 Starting OBS Live Translator v2.0 (Rust Edition)");
    
    // Create high-performance configuration
    let config = TranslatorConfig {
        target_latency_ms: 25,
        max_gpu_memory_mb: 8192,
        thread_count: num_cpus::get(),
        aggressive_mode: true,
        batch_size: 8,
        buffer_size_samples: 1024,
    };
    
    // Initialize translator
    let translator = match HighPerformanceTranslator::new(config).await {
        Ok(translator) => translator,
        Err(e) => {
            error!("Failed to initialize translator: {}", e);
            return Err(e);
        }
    };
    
    info!("✅ Translator initialized successfully");
    
    // Start translation pipeline
    let mut translation_stream = translator.start_translation_pipeline().await?;
    
    info!("🎤 Translation pipeline started - listening for audio");
    info!("📊 Dashboard: http://localhost:8080/dashboard");
    info!("📺 OBS Overlay: http://localhost:8080/overlay");
    
    // Process translations
    while let Some(result) = translation_stream.recv().await {
        info!(
            "Translation: '{}' -> '{}' ({:.1}ms)", 
            result.original_text,
            result.translated_text,
            result.processing_time_ms
        );
    }
    
    Ok(())
}

async fn run_benchmark() -> Result<()> {
    info!("🏁 Running performance benchmark...");
    
    // TODO: Implement comprehensive benchmarking
    // - Audio processing latency
    // - ASR throughput
    // - Translation speed
    // - Memory usage
    // - GPU utilization
    
    println!("System Information:");
    println!("==================");
    println!("Models:");
    println!("  • Whisper V3 Turbo: 5.4x faster than V3");
    println!("  • NLLB-600M with CTranslate2 INT8");
    println!("  • Silero VAD for voice detection");
    println!("Configuration:");
    println!("  • Execution provider: ONNX Runtime");
    println!("  • Quantization: INT8/FP16 mixed");
    println!("  • Memory optimization: Adaptive");
    println!("==================");
    
    Ok(())
}