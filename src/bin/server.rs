//! OBS Live Translator Server Binary

use obs_live_translator::config::AppConfig;
use obs_live_translator::streaming::{StreamingConfig, StreamingServer};
use anyhow::Result;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG")
                .unwrap_or_else(|_| "obs_live_translator=info,tower_http=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = AppConfig::load()?;

    // Create streaming config
    let streaming_config = StreamingConfig {
        buffer_size: config.server.websocket_buffer_size,
        max_latency_ms: 500,
        reconnect_attempts: 3,
        heartbeat_interval_ms: 30000,
    };

    // Create and start server
    let server = StreamingServer::new(streaming_config, config);

    tracing::info!("Starting OBS Live Translator Server");

    server.start().await?;

    Ok(())
}