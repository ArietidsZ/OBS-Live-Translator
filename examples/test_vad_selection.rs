use anyhow::Result;
use obs_live_translator::types::{TranslatorConfig, VadType};
use obs_live_translator::vad::create_vad_engine;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    println!("Testing VAD Selection Strategy...");

    let mut config = TranslatorConfig::default();
    config.model_path = "./models".to_string();

    // Ensure directory exists
    std::fs::create_dir_all("./models")?;

    // Create dummy models if they don't exist
    let models = vec!["silero_vad.onnx", "silero_vad_v5.onnx", "ten_vad.onnx"];
    for model in models {
        let path = format!("./models/{}", model);
        if !std::path::Path::new(&path).exists() {
            std::fs::write(&path, b"DUMMY_MODEL_CONTENT")?;
            println!("Created dummy model: {}", path);
        }
    }

    // Test SileroV5 (Default)
    config.vad_type = VadType::SileroV5;
    let cache = std::sync::Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let engine = create_vad_engine(&config, cache.clone()).await?;
    println!("Selected: {}", engine.name());
    assert_eq!(engine.name(), "SileroVadV5");

    // Test TenVad
    config.vad_type = VadType::TenVad;
    let engine = create_vad_engine(&config, cache.clone()).await?;
    println!("Selected: {}", engine.name());
    assert_eq!(engine.name(), "TenVad");

    // Test Cobra (Stub)
    config.vad_type = VadType::Cobra;
    let engine = create_vad_engine(&config, cache.clone()).await?;
    println!("Selected: {}", engine.name());
    assert_eq!(engine.name(), "CobraVad");

    // Test Legacy Silero
    config.vad_type = VadType::Silero;
    let engine = create_vad_engine(&config, cache.clone()).await?;
    println!("Selected: {}", engine.name());
    assert_eq!(engine.name(), "SileroVAD (Legacy)");

    println!("VAD Selection Verification Passed!");

    Ok(())
}
