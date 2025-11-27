use anyhow::Result;
use obs_live_translator::types::TranslatorConfig;
use obs_live_translator::vad::{silero::SileroVad, VadEngine};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Testing Silero VAD v5 Integration...");

    // Create config
    let mut config = TranslatorConfig::default();
    config.model_path = "./models".to_string();

    // Ensure model exists (or placeholder)
    if !std::path::Path::new("./models/silero_vad_v5.onnx").exists() {
        println!(
            "Model not found, skipping inference test. Run scripts/download_silero_v5.py first."
        );
        // We'll run the script in the command line, but good to check here.
    }

    // Initialize VAD
    let cache = std::sync::Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let vad = SileroVad::new(&config, cache).await?;

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

    println!("Silero VAD v5 Verification Passed!");

    Ok(())
}
