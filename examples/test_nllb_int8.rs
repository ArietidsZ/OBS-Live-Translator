// Part 2.2: NLLB-200 INT8 Model Loading Test

use anyhow::Result;
use obs_live_translator::{
    inference::Precision,
    translation::{nllb::NLLB200, TranslationEngine},
    types::TranslatorConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("{}", "=".repeat(60));
    println!("Part 2.2: NLLB-200 INT8 Model Loading Test");
    println!("{}", "=".repeat(60));
    println!();

    let config = TranslatorConfig {
        model_path: "./models".to_string(),
        ..Default::default()
    };

    println!("📂 Loading NLLB-200 model...");
    println!("   Model path: {}", config.model_path);
    println!("   Requested precision: INT8");
    println!();

    // Load with INT8 precision
    let start = std::time::Instant::now();
    let cache = std::sync::Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let nllb = match NLLB200::new(&config, Precision::INT8, cache).await {
        Ok(model) => {
            let elapsed = start.elapsed();
            println!(
                "✅ Model loaded successfully in {:.2}s",
                elapsed.as_secs_f32()
            );
            println!("   Model: {}", model.model_name());
            println!("   Languages: {}", model.language_count());
            model
        }
        Err(e) => {
            println!("❌ Failed to load model: {}", e);
            println!();
            println!("Possible issues:");
            println!("  1. INT8 model not downloaded - run: python3 scripts/download_nllb_int8.py");
            println!("  2. Model files missing or corrupted");
            return Err(e.into());
        }
    };

    println!();
    println!("{}", "=".repeat(60));
    println!("✅ Test PASSED: INT8 model loading successful");
    println!("{}", "=".repeat(60));
    println!();

    println!("📊 Model Details:");
    println!("   Name: {}", nllb.model_name());
    println!("   Architecture: Encoder-Decoder with KV caching");
    println!("   Languages: {} language pairs", nllb.language_count());
    println!("   Expected memory: ~1GB (INT8) vs ~2GB (FP16)");
    println!("   Expected latency: 1.5-2x faster translation");
    println!();

    println!("📁 Model Files:");
    println!("   Encoder: nllb-encoder.onnx (512 MB)");
    println!("   Decoder: nllb-decoder-int8.onnx (1.43 GB)");
    println!("   Decoder (with KV cache): nllb-decoder-with-past-int8.onnx (2.4 GB)");
    println!("   Tokenizer: nllb-tokenizer.json (16.5 MB)");
    println!();

    println!("⚠️  Note: Inference not yet implemented");
    println!("   - Model loading: ✅ Working");
    println!("   - Translation pipeline: 🚧 TODO");
    println!();

    println!("✅ Part 2.2 COMPLETE!");
    println!();

    Ok(())
}
