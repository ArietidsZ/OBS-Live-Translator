//! Streaming server implementation

use super::{StreamingConfig, WebSocketHandler, SessionManager};
use crate::config::AppConfig;
use anyhow::Result;
use axum::{
    routing::get,
    Router,
    extract::ws::{WebSocket, WebSocketUpgrade},
};
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};

/// Main streaming server
pub struct StreamingServer {
    config: StreamingConfig,
    session_manager: Arc<SessionManager>,
    app_config: AppConfig,
}

impl StreamingServer {
    pub fn new(config: StreamingConfig, app_config: AppConfig) -> Self {
        let session_manager = Arc::new(SessionManager::new());

        Self {
            config,
            session_manager,
            app_config,
        }
    }

    /// Start the server
    pub async fn start(&self) -> Result<()> {
        let app = self.create_app();
        let addr = format!("{}:{}", self.app_config.server.host, self.app_config.server.port);

        info!("Starting streaming server on {}", addr);

        let listener = TcpListener::bind(&addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// Create the Axum application
    fn create_app(&self) -> Router {
        let session_manager = Arc::clone(&self.session_manager);

        Router::new()
            .route("/ws", get(move |ws: WebSocketUpgrade| {
                let session_manager = Arc::clone(&session_manager);
                async move {
                    ws.on_upgrade(move |socket| {
                        handle_websocket(socket, session_manager)
                    })
                }
            }))
            .route("/health", get(|| async { "OK" }))
            .route("/stats", get({
                let session_manager = Arc::clone(&self.session_manager);
                move || async move {
                    let stats = session_manager.get_stats().await;
                    serde_json::to_string(&stats).unwrap_or_else(|_| "{}".to_string())
                }
            }))
    }

    /// Get server statistics
    pub async fn get_stats(&self) -> ServerStats {
        let session_stats = self.session_manager.get_stats().await;

        ServerStats {
            active_sessions: session_stats.active_sessions,
            total_connections: session_stats.total_connections,
            total_messages: session_stats.total_messages,
            uptime_seconds: session_stats.uptime_seconds,
        }
    }
}

/// Handle WebSocket connections
async fn handle_websocket(socket: WebSocket, session_manager: Arc<SessionManager>) {
    let handler = WebSocketHandler::new(socket, session_manager);

    if let Err(e) = handler.handle().await {
        error!("WebSocket error: {}", e);
    }
}

/// Server statistics
#[derive(Debug, serde::Serialize)]
pub struct ServerStats {
    pub active_sessions: usize,
    pub total_connections: u64,
    pub total_messages: u64,
    pub uptime_seconds: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_server_creation() {
        let config = StreamingConfig::default();
        let app_config = AppConfig::default();
        let server = StreamingServer::new(config, app_config);

        // Server should be created successfully
        assert_eq!(server.session_manager.active_sessions_count(), 0);
    }
}