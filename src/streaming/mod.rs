//! Real-time streaming and WebSocket server

pub mod server;
pub mod websocket;
pub mod session;

pub use server::StreamingServer;
pub use websocket::{WebSocketHandler, WebSocketMessage};
pub use session::{StreamingSession, SessionManager};

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub buffer_size: usize,
    pub max_latency_ms: u32,
    pub reconnect_attempts: u32,
    pub heartbeat_interval_ms: u32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 8192,
            max_latency_ms: 500,
            reconnect_attempts: 3,
            heartbeat_interval_ms: 30000,
        }
    }
}