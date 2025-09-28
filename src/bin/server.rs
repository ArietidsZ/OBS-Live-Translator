//! OBS Live Translator Server Binary

use obs_live_translator::config::AppConfig;
use obs_live_translator::streaming::{StreamingConfig, WebSocketHandler, SessionManager};
use anyhow::Result;
use axum::{Router, routing::get, extract::ws::WebSocketUpgrade};
use std::net::SocketAddr;
use std::sync::Arc;
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

    // Create session manager
    let session_manager = Arc::new(SessionManager::new());

    // Create router with WebSocket endpoint
    let app = Router::new()
        .route("/ws", get({
            let session_manager = session_manager.clone();
            move |ws: WebSocketUpgrade| async move {
                ws.on_upgrade(move |socket| async move {
                    let handler = WebSocketHandler::new(socket, session_manager);
                    if let Err(e) = handler.handle().await {
                        tracing::error!("WebSocket handler error: {}", e);
                    }
                })
            }
        }));

    // Start server
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port)
        .parse()?;
    tracing::info!("🚀 OBS Live Translator Server listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}