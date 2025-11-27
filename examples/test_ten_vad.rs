use anyhow::Result;
use obs_live_translator::types::TranslatorConfig;
use obs_live_translator::vad::{ten_vad::TenVad, VadEngine};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Testing TEN VAD Integration...");

    // Create config
    let mut config = TranslatorConfig::default();
    config.model_path = "./models".to_string();

    // Ensure model exists (or placeholder)
    if !std::path::Path::new("./models/ten_vad.onnx").exists() {
        println!(
            "Model not found, skipping inference test. Run scripts/download_ten_vad.py first."
        );
        return Ok(());
    }

    // Initialize VAD
    let cache = std::sync::Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let vad = TenVad::new(&config, cache).await?;

    // Create dummy audio (1 second of silence)
    let silence = vec![0.0f32; 16000];

    // Create dummy audio (1 second of "speech" - high energy noise)
    let speech: Vec<f32> = (0..16000).map(|i| (i as f32).sin() * 0.5).collect();

    // Benchmark Silence
    let start = Instant::now();
    let is_speech = vad.detect(&silence).await?;
    let duration = start.elapsed();
    println!("Silence Detection: {} (took {:?})", is_speech, duration);
    assert!(!is_speech, "Silence should not be detected as speech");

    // Benchmark Speech
    let start = Instant::now();
    let is_speech = vad.detect(&speech).await?;
    let duration = start.elapsed();
    println!("Speech Detection: {} (took {:?})", is_speech, duration);
    assert!(is_speech, "High energy audio should be detected as speech");

    println!("TEN VAD Verification Passed!");

    Ok(())
}
