//! Streaming server with WebSocket and WebTransport support

pub mod protocol;
pub mod websocket;
pub mod webtransport;

use crate::types::TranslationResult;

/// Streaming message types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type")]
pub enum StreamMessage {
    /// Configuration message from client
    #[serde(rename = "config")]
    Config {
        source_language: Option<String>,
        target_language: String,
        profile: Option<String>,
    },

    /// Audio data from client (binary)
    #[serde(rename = "audio")]
    Audio {
        #[serde(skip)]
        samples: Vec<f32>,
    },

    /// Translation result to client
    #[serde(rename = "result")]
    Result(TranslationResult),

    /// Error message
    #[serde(rename = "error")]
    Error { message: String },

    /// Ping/pong for keepalive
    #[serde(rename = "ping")]
    Ping,
    #[serde(rename = "pong")]
    Pong,
}
