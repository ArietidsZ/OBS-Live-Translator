//! Model Download Utility
//!
//! Downloads required models for a specific profile

use obs_live_translator::{models, profile::ProfileDetector, Profile};
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "download_models=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("OBS Live Translator - Model Downloader");

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let profile = if args.len() > 1 {
        match args[1].to_lowercase().as_str() {
            "low" => Profile::Low,
            "medium" => Profile::Medium,
            "high" => Profile::High,
            _ => {
                eprintln!("Invalid profile. Use: low, medium, or high");
                eprintln!("Auto-detecting profile...");
                ProfileDetector::detect()?
            }
        }
    } else {
        tracing::info!("No profile specified, auto-detecting...");
        ProfileDetector::detect()?
    };

    tracing::info!("Downloading models for profile: {:?}", profile);

    // Create models directory
    let models_dir = PathBuf::from("./models");
    tokio::fs::create_dir_all(&models_dir).await?;

    // Get models for profile
    let model_list = models::get_profile_models(profile);

    tracing::info!("Found {} models to download", model_list.len());

    // Download each model
    for model in &model_list {
        let destination = models_dir.join(model.name.replace(" ", "_").to_lowercase() + ".onnx");

        if destination.exists() {
            tracing::info!("Model already exists, skipping: {}", model.name);
            continue;
        }

        match models::download_model(model, &destination).await {
            Ok(_) => tracing::info!("✓ Successfully downloaded: {}", model.name),
            Err(e) => {
                tracing::warn!("✗ Failed to download {}: {}", model.name, e);
                tracing::info!(
                    "  Note: Model URLs are placeholders. You'll need to manually download from:"
                );
                tracing::info!("  {}", model.url);
            }
        }
    }

    tracing::info!("Model download complete!");
    tracing::info!("Models directory: {}", models_dir.display());

    Ok(())
}
