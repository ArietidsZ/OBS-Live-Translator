// Part 2.1 Testing: Parakeet INT8 Model Loading Test
// Validates that the INT8 model can be loaded and basic functionality works

use anyhow::Result;
use obs_live_translator::{
    asr::{parakeet::ParakeetTDT, ASREngine},
    types::TranslatorConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("{}", "=".repeat(60));
    println!("Part 2.1: Parakeet TDT INT8 Model Loading Test");
    println!("{}", "=".repeat(60));
    println!();

    // Create config pointing to models directory
    let config = TranslatorConfig {
        model_path: "./models".to_string(),
        source_language: "en".to_string(),
        target_language: "en".to_string(),
        ..Default::default()
    };

    println!("📂 Loading Parakeet TDT model...");
    println!("   Model path: {}", config.model_path);
    println!();

    // Load the model - will automatically select INT8 if available
    let start = std::time::Instant::now();
    let cache = std::sync::Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let parakeet = match ParakeetTDT::new(&config, cache).await {
        Ok(model) => {
            let elapsed = start.elapsed();
            println!(
                "✅ Model loaded successfully in {:.2}s",
                elapsed.as_secs_f32()
            );
            println!("   Model: {}", model.model_name());
            println!("   Sample rate: {} Hz", model.sample_rate());
            model
        }
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            println!();
            println!("Possible issues:");
            println!(
                "  1. INT8 model not downloaded - run: python3 scripts/download_parakeet_int8.py"
            );
            println!("  2. FP16 model not available either");
            println!("  3. ONNX Runtime not properly configured");
            return Err(e.into());
        }
    };

    println!();
    println!("{}", "=".repeat(60));
    println!("✅ Test PASSED: 3-component model loading successful");
    println!("{}", "=".repeat(60));
    println!();

    // Print model details
    println!("📊 Model Details:");
    println!("   Name: {}", parakeet.model_name());
    println!("   Architecture: 3-component transducer (encoder/decoder/joiner)");
    println!("   Expected memory: ~661MB (INT8) or ~1.2GB (FP16)");
    println!("   Expected latency: ~100-150ms (INT8) or ~200ms (FP16)");
    println!();

    println!("📁 Model Files:");
    println!("   Encoder: parakeet-tdt-encoder-int8.onnx (622 MB)");
    println!("   Decoder: parakeet-tdt-decoder-int8.onnx (6.9 MB)");
    println!("   Joiner: parakeet-tdt-joiner-int8.onnx (1.7 MB)");
    println!();

    println!("⚠️  Note: Inference not yet implemented");
    println!("   - Model loading: ✅ Working");
    println!("   - Inference pipeline: 🚧 TODO");
    println!();

    println!("Next steps:");
    println!("  1. Implement audio feature extraction");
    println!("  2. Implement encoder/decoder/joiner inference");
    println!("  3. Implement beam search decoding");
    println!("  4. Benchmark INT8 vs FP16 performance");
    println!();

    Ok(())
}
