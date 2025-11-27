use obs_live_translator::{
    asr::{canary::Canary180M, ASREngine},
    types::{Profile, TranslatorConfig},
    Result,
};
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    tracing::info!("Testing Canary 180M INT8 loading...");

    let config = TranslatorConfig {
        profile: Profile::High, // Maps to Canary
        source_language: "en".to_string(),
        target_language: "es".to_string(),
        model_path: "models".to_string(),
        ..Default::default()
    };

    let start = Instant::now();
    let cache = std::sync::Arc::new(obs_live_translator::models::session_cache::SessionCache::new());
    let canary = Canary180M::new(&config, cache).await?;
    let duration = start.elapsed();

    tracing::info!(
        "Successfully loaded Canary 180M in {:.2}s",
        duration.as_secs_f64()
    );
    tracing::info!("Model Name: {}", canary.model_name());
    tracing::info!("Sample Rate: {} Hz", canary.sample_rate());

    Ok(())
}
