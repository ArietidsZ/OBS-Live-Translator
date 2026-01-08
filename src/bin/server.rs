//! OBS Translator Server
//!
//! Main server binary with WebSocket and WebTransport support

use obs_live_translator::{
    config,
    profile::ProfileDetector,
    server::{self, AppState},
    Translator,
    TranslatorConfig,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "obs_live_translator=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("OBS Live Translator v5.0 - Starting");

    // Load configuration
    let config = match config::load_config("config.toml") {
        Ok(config) => config,
        Err(_) => {
            tracing::info!("No config file found, using defaults with auto-detected profile");
            TranslatorConfig {
                profile: ProfileDetector::detect()?,
                ..Default::default()
            }
        }
    };

    tracing::info!("Configuration loaded: profile={:?}", config.profile);

    // Initialize Prometheus recorder
    let recorder_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder");

    // Initialize translator
    let translator = match Translator::new(config).await {
        Ok(t) => {
            tracing::info!("Translator initialized successfully");
            std::sync::Arc::new(t)
        }
        Err(e) => {
            tracing::error!("Failed to initialize translator: {}", e);
            tracing::info!("This is expected if models are not yet downloaded");
            tracing::info!("Run 'cargo run --bin download-models' to download required models");
            return Err(e.into());
        }
    };

    let state = std::sync::Arc::new(AppState::new(translator));
    state.mark_ready();

    // Define Axum app
    let app = server::build_router(state, recorder_handle);

    // Start server
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], 3000));
    tracing::info!("Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(server::shutdown_signal())
        .await?;

    Ok(())
}
